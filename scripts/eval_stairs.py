"""Automated four-step-staircase policy evaluation CLI.

The success/failure **logic** lives in
``src/tasks/velocity/mdp/stair_eval.py`` (pure, CPU-testable).  This module is
only the runtime adapter: it loads a checkpoint exactly like
``scripts/play.py`` (``load_env_cfg(..., play=True)``, ``load_rl_cfg``,
``load_runner_cls``, ``get_checkpoint_path``, ``RslRlVecEnvWrapper``) and feeds
one :class:`TrialSample` per control step into :class:`TrialEvaluator`.

Example::

  python scripts/eval_stairs.py Quadruped-3320-Stairs \
      --checkpoint logs/rsl_rl/quadruped_3320_velocity/<run>/model_2999.pt \
      --trials 5 --out results.csv --seed 0

The CLI is headless by default.  Pass ``--video-dir DIR`` to also record one
MP4 per trial with the repository's ``VideoRecorder``.  ``results.csv`` always
has exactly these columns::

  trial_id, seed, command_speed, success, elapsed_time,
  max_abs_roll, max_abs_pitch, failure_reason, notes

``max_abs_roll`` / ``max_abs_pitch`` are written in **degrees** so they can be
compared directly with the assessment's 45 deg limit; the pure logic stores
radians.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from src.tasks.velocity.mdp.stair_eval import (
  FAIL_TIMEOUT,
  SuccessCriteria,
  TrialEvaluator,
  TrialResult,
  TrialSample,
  assert_mandated_geometry,
  quat_wxyz_to_roll_pitch,
)

#: ``results.csv`` columns, in order (assessment section 6).
CSV_COLUMNS = [
  "trial_id",
  "seed",
  "command_speed",
  "success",
  "elapsed_time",
  "max_abs_roll",
  "max_abs_pitch",
  "failure_reason",
  "notes",
]

#: Command speed of the assessment's hard "达标" tier.
PASS_SPEED_MPS = 0.4
#: Required success fraction at the pass tier: 4 of 5 ("至少成功 4 次").
PASS_FRACTION = 0.8

#: Peak non-foot contact force treated as a real touch by the adapter.  The
#: environment's hard ``illegal_contact`` termination uses 10 N; the evaluator's
#: ``body_contact_grace_s`` absorbs brief low-force grazes.
BODY_CONTACT_FORCE_N = 1.0


class _BaseState(NamedTuple):
  x: float
  z: float
  roll: float
  pitch: float


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="eval_stairs.py",
    description=(
      "Evaluate a trained staircase policy over independent episodes and "
      "write an assessment results.csv."
    ),
  )
  parser.add_argument(
    "task_id",
    help="Registered mjlab task id, e.g. Quadruped-3320-Stairs.",
  )
  parser.add_argument(
    "--checkpoint",
    required=True,
    help=(
      "Run directory containing model_*.pt, an experiment log root, or a "
      "direct path to a .pt checkpoint."
    ),
  )
  parser.add_argument(
    "--trials",
    type=int,
    default=5,
    help="Number of independent episodes to run (default: 5).",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("results.csv"),
    help="Output CSV path (default: results.csv).",
  )
  parser.add_argument(
    "--video-dir",
    type=Path,
    default=None,
    help="If given, record one MP4 per trial into this directory.",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Base random seed. Trial i uses seed + i (default: random).",
  )
  parser.add_argument(
    "--video-width",
    type=int,
    default=None,
    help="Recorded frame width in pixels (default: env viewer width).",
  )
  parser.add_argument(
    "--video-height",
    type=int,
    default=None,
    help="Recorded frame height in pixels (default: env viewer height).",
  )
  parser.add_argument(
    "--no-render",
    action="store_true",
    help="Force headless mode, ignoring --video-dir.",
  )
  parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Torch device, e.g. cuda:0 or cpu (default: cuda:0 if available).",
  )
  return parser


# ---------------------------------------------------------------------------
# Checkpoint resolution (mirrors scripts/play.py / mjlab.utils.os)
# ---------------------------------------------------------------------------


def normalize_play_cfg(env_cfg) -> None:
  """Make the copied play config self-consistent without changing its intent.

  The registered play config turns off heading commands but keeps the inherited
  ``ranges.heading`` range; mjlab's ``UniformVelocityCommand`` rejects that
  combination at construction.  The heading range is ignored once
  ``heading_command=False``, so clearing it changes nothing that is measured
  (terrain, collisions, command speed and criteria are untouched).
  """
  commands = getattr(env_cfg, "commands", None)
  twist = commands.get("twist") if commands else None
  if twist is None:
    return
  ranges = getattr(twist, "ranges", None)
  if not getattr(twist, "heading_command", False) and ranges is not None:
    if ranges.heading is not None:
      print(
        "[WARN] Play config has heading_command=False with ranges.heading set; "
        "clearing the unused heading range."
      )
      ranges.heading = None


def resolve_checkpoint(spec: str | Path) -> Path:
  """Resolve a checkpoint from a ``.pt`` file, a run dir, or a log root.

  * a file is used as-is (play.py's ``--checkpoint`` behaviour),
  * a directory containing ``model_*.pt`` is passed to
    :func:`get_checkpoint_path` as the run directory,
  * any other directory is treated as an experiment log root.
  """
  path = Path(spec).expanduser()
  if not path.exists():
    raise FileNotFoundError(f"Checkpoint path not found: {path}")
  if path.is_file():
    return path
  if any(path.glob("model_*.pt")):
    return get_checkpoint_path(path.parent, re.escape(path.name), r"model_.*\.pt$")
  return get_checkpoint_path(path, ".*", r"model_.*\.pt$")


# ---------------------------------------------------------------------------
# Runtime state adapters
# ---------------------------------------------------------------------------


def _base_state(env: RslRlVecEnvWrapper) -> _BaseState:
  """Read the base-link pose from the live environment in world coordinates."""
  robot = env.unwrapped.scene["robot"]
  pose = robot.data.root_link_pose_w[0]
  roll, pitch = quat_wxyz_to_roll_pitch(
    (float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
  )
  return _BaseState(
    x=float(pose[0]), z=float(pose[2]), roll=roll, pitch=pitch
  )


def _body_contact(env: RslRlVecEnvWrapper) -> tuple[bool, float]:
  """Return ``(touched, peak_force_n)`` for non-foot ground/step contacts."""
  try:
    sensor = env.unwrapped.scene["nonfoot_ground_touch"]
  except KeyError:
    return False, 0.0
  data = sensor.data
  touched = False
  peak = 0.0
  if data.found is not None:
    touched = bool(data.found.gt(0).any().item())
  # history_length=4 records every physics substep of the control step, so a
  # contact that ended before the step boundary is still detected.
  if data.force_history is not None:
    peak = float(torch.linalg.vector_norm(data.force_history, dim=-1).max().item())
    touched = touched or peak > BODY_CONTACT_FORCE_N
  elif data.force is not None:
    peak = float(torch.linalg.vector_norm(data.force, dim=-1).max().item())
    touched = touched or peak > BODY_CONTACT_FORCE_N
  return touched, peak


def _termination_reason(
  env: RslRlVecEnvWrapper, terminated: bool, truncated: bool
) -> str | None:
  """Map the environment's termination terms to stair_eval reason tokens."""
  manager = env.unwrapped.termination_manager

  def fired(name: str) -> bool:
    try:
      return bool(manager.get_term(name)[0].item())
    except (KeyError, ValueError):
      return False

  if terminated:
    if fired("illegal_contact"):
      return "body_contact"
    if fired("fell_over"):
      return "orientation"
    return None  # evaluator treats an unknown termination as a reset
  if truncated:
    return "timeout"
  return None


def _make_sample(
  t: float,
  state: _BaseState,
  origin_x: float,
  body_contact: bool,
  contact_force_n: float,
  env_done: bool,
  env_reason: str | None,
) -> TrialSample:
  return TrialSample(
    t=t,
    x=state.x - origin_x,
    z=state.z,
    roll=state.roll,
    pitch=state.pitch,
    body_contact=body_contact,
    env_done=env_done,
    env_reason=env_reason,
    contact_force_n=contact_force_n,
  )


def _policy_actions(policy, obs) -> torch.Tensor:
  with torch.no_grad():
    return policy(obs)


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------


def _run_trial(
  env: RslRlVecEnvWrapper,
  policy,
  criteria: SuccessCriteria,
  *,
  trial_seed: int,
  video: VideoRecorder | None,
  video_starts: list[int] | None,
  global_steps: int,
) -> tuple[TrialResult, float, int]:
  """Run one independent episode and return ``(result, command_speed, steps)``."""
  env.unwrapped.seed(trial_seed)
  if video is not None and video_starts is not None:
    video_starts.append(global_steps)

  # The terrain patch is randomised by a reset event that we register BEFORE
  # ``reset_base`` (see quadruped_3320_stairs_env_cfg), so the terrain and its
  # origins are already final by the time reset_base places the robot.  Sample
  # the course origin AFTER the reset -- sampling it before would anchor the
  # course frame to the previous patch and offset x by whole patch spacings
  # (observed: max_x jumping to +/-13 m).
  obs, _ = env.reset()
  origin_x = float(env.unwrapped.scene.env_origins[0][0].item())
  command = env.unwrapped.command_manager.get_command("twist")
  command_speed = float(command[0, 0].item())
  step_dt = float(env.unwrapped.step_dt)

  evaluator = TrialEvaluator(criteria)
  max_steps = int(math.ceil(criteria.episode_timeout_s / step_dt)) + 5

  state = _base_state(env)
  touched, force = _body_contact(env)
  result = evaluator.update(
    _make_sample(0.0, state, origin_x, touched, force, False, None)
  )

  t = 0.0
  trial_steps = 0
  while result is None:
    if trial_steps >= max_steps:
      # The evaluator's own timeout fires at episode_timeout_s; this is a
      # belt-and-braces guard so a pathological env can never hang the CLI.
      result = evaluator.finalize(FAIL_TIMEOUT, t)
      break

    actions = _policy_actions(policy, obs)
    prev_state = _base_state(env)
    prev_touched, prev_force = _body_contact(env)

    obs, _reward, _dones, _extras = env.step(actions)
    global_steps += 1
    trial_steps += 1
    t += step_dt

    terminated = bool(env.unwrapped.reset_terminated[0].item())
    truncated = bool(env.unwrapped.reset_time_outs[0].item())
    if terminated or truncated:
      # env.step() already reset this env, so the live pose is the fresh spawn;
      # report the last state observed before the terminating step instead.
      state = prev_state
      touched, force = prev_touched, prev_force
      env_reason = _termination_reason(env, terminated, truncated)
      env_done = True
    else:
      state = _base_state(env)
      touched, force = _body_contact(env)
      env_reason = None
      env_done = False

    result = evaluator.update(
      _make_sample(t, state, origin_x, touched, force, env_done, env_reason)
    )

  if video is not None and video.is_recording:
    video._finish_recording()

  return result, command_speed, global_steps


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_summary(rows: list[dict]) -> None:
  header = (
    f"{'trial':>5}  {'success':>7}  {'speed':>6}  {'time[s]':>8}  "
    f"{'roll[deg]':>9}  {'pitch[deg]':>10}  {'reason':<14}  notes"
  )
  print()
  print(header)
  print("-" * len(header))
  for row in rows:
    print(
      f"{row['trial_id']:>5}  {row['success']:>7}  {row['command_speed']:>6}  "
      f"{row['elapsed_time']:>8}  {row['max_abs_roll']:>9}  "
      f"{row['max_abs_pitch']:>10}  {row['failure_reason']:<14}  {row['notes']}"
    )
  print("-" * len(header))


def _write_csv(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_eval(args: argparse.Namespace) -> int:
  if args.trials < 1:
    print("[ERROR] --trials must be >= 1", file=sys.stderr)
    return 2

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  resume_path = resolve_checkpoint(args.checkpoint)
  print(f"[INFO] Task:        {args.task_id}")
  print(f"[INFO] Checkpoint:  {resume_path}")
  print(f"[INFO] Device:      {device}")

  env_cfg = load_env_cfg(args.task_id, play=True)
  agent_cfg = load_rl_cfg(args.task_id)
  normalize_play_cfg(env_cfg)

  if args.video_width is not None:
    env_cfg.viewer.width = args.video_width
  if args.video_height is not None:
    env_cfg.viewer.height = args.video_height

  # One environment per trial keeps episodes fully independent.
  env_cfg.scene.num_envs = 1

  # The mandated-geometry guard runs on CPU and fails loudly if the assessment
  # dimensions ever drift; the criteria themselves are the existing logic.
  criteria = SuccessCriteria()
  assert_mandated_geometry(criteria.geometry)

  seed_base = args.seed if args.seed is not None else random.randint(0, 10**6)
  env_cfg.seed = seed_base
  print(f"[INFO] Base seed:   {seed_base} (trial i uses seed + i)")

  record = bool(args.video_dir) and not args.no_render
  if args.video_dir and args.no_render:
    print("[WARN] --no-render given: skipping video recording.")
  render_mode = "rgb_array" if record else None

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  video: VideoRecorder | None = None
  video_starts: list[int] | None = None
  if record:
    video_starts = []
    # Record one clip per trial: the trigger fires at the first step of each
    # trial, and the trial loop finalises the clip when the trial ends.
    raw_env = VideoRecorder(
      raw_env,
      video_folder=args.video_dir,
      step_trigger=lambda step: step in (video_starts or ()),
      video_length=None,
      disable_logger=False,
    )
    video = raw_env
    print(f"[INFO] Recording videos to {args.video_dir}")

  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  rows: list[dict] = []
  global_steps = 0
  try:
    for trial_id in range(args.trials):
      trial_seed = seed_base + trial_id
      try:
        result, command_speed, global_steps = _run_trial(
          env,
          policy,
          criteria,
          trial_seed=trial_seed,
          video=video,
          video_starts=video_starts,
          global_steps=global_steps,
        )
      except Exception as exc:  # noqa: BLE001 - keep the report reproducible
        import traceback

        traceback.print_exc()
        result = TrialResult(
          success=False,
          failure_reason="error",
          elapsed_time=0.0,
          max_abs_roll=0.0,
          max_abs_pitch=0.0,
          notes=f"runtime error: {exc}",
        )
        command_speed = float("nan")

      row = {
        "trial_id": trial_id,
        "seed": trial_seed,
        "command_speed": f"{command_speed:.4f}",
        "success": result.success,
        "elapsed_time": f"{result.elapsed_time:.4f}",
        "max_abs_roll": f"{result.max_abs_roll_deg:.4f}",
        "max_abs_pitch": f"{result.max_abs_pitch_deg:.4f}",
        "failure_reason": result.failure_reason,
        "notes": result.notes,
      }
      rows.append(row)
      print(
        f"[INFO] Trial {trial_id}: "
        f"{'SUCCESS' if result.success else 'FAILURE'} "
        f"({result.failure_reason}) after {result.elapsed_time:.2f} s"
      )
  finally:
    env.close()

  _print_summary(rows)
  _write_csv(args.out, rows)
  print(f"[INFO] Wrote {args.out}")

  successes = sum(1 for row in rows if row["success"])
  speed = rows[0]["command_speed"] if rows else "0"
  try:
    is_pass_tier = abs(float(speed) - PASS_SPEED_MPS) < 1e-3
  except ValueError:
    is_pass_tier = False
  required = max(1, math.ceil(PASS_FRACTION * len(rows)))
  print(
    f"[INFO] Successes: {successes}/{len(rows)} "
    f"(required >= {required} at {PASS_SPEED_MPS:.1f} m/s)"
  )

  if is_pass_tier and successes < required:
    print(
      f"[ERROR] Fewer than {required} of {len(rows)} trials succeeded at "
      f"{PASS_SPEED_MPS:.1f} m/s.",
      file=sys.stderr,
    )
    return 1
  return 0


def main(argv: list[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)

  # Import the task packages to populate the registry (as in play.py/train.py).
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  if args.task_id not in all_tasks:
    print(f"[ERROR] Unknown task '{args.task_id}'.", file=sys.stderr)
    print("Available tasks:", file=sys.stderr)
    for task in all_tasks:
      print(f"  {task}", file=sys.stderr)
    return 2

  return run_eval(args)


if __name__ == "__main__":
  raise SystemExit(main())
