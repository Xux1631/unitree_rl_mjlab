# Automated four-step-staircase evaluation (`scripts/eval_stairs.py`)

Automates the assessment's `play` procedure (§4.3) against a trained
checkpoint: five independent episodes at a constant forward speed on the
mandated four-step staircase, one `results.csv` row per episode, and a
non-zero exit code when the hard "达标" tier is missed.

The success/failure **logic** is **not** re-implemented here. It lives in
`src/tasks/velocity/mdp/stair_eval.py` as pure, CPU-testable code
(`CourseGeometry`, `SuccessCriteria`, `TrialSample`, `TrialEvaluator`,
`evaluate_trace`). The CLI is only the runtime adapter that reads the robot
state out of the live environment and feeds one `TrialSample` per control step
into `TrialEvaluator`.

## How it works (methodology)

1. **Environment.** `load_env_cfg(task_id, play=True)` builds exactly the
   configuration used by `play.py`: full (un-curriculised) 4-step flight,
   terrain randomisation off per-patch, `reset_base` yaw pinned to 0, episode
   length effectively infinite, and a **constant** twist command whose forward
   speed is fixed per task (`0.4` / `0.6` / `0.8` m/s). `num_envs` is forced to
   `1` so every trial is one independent episode.
2. **Checkpoint.** `--checkpoint` accepts a `.pt` file, a run directory that
   contains `model_*.pt`, or an experiment log root. Directories are resolved
   with `mjlab.utils.os.get_checkpoint_path` (newest run / newest alphabetical
   checkpoint), mirroring `play.py`/`train.py`.
3. **Policy.** `load_rl_cfg`, `load_runner_cls` and `RslRlVecEnvWrapper` are
   used exactly as in `play.py`; the runner is loaded with
   `load_cfg={"actor": True}, strict=True, map_location=device` and queried
   through `get_inference_policy`. Actions are deterministic (distribution
   mean) and taken under `torch.no_grad()`.
4. **Per-step samples.** For each control step the adapter builds a
   `TrialSample` in the **course frame**:
   * `x` = base-link world `x` minus the patchorigin (`env_origins[0][0]`),
     so the nominal spawn is `x = 0` and the first riser sits at
     `x = 0.9 m` regardless of where the randomised patch is;
   * `z` = base-link world height (approach ground is `z = 0`, but the base
     *link* stands ≈0.33 m above it — see the reach check below);
   * `roll` / `pitch` from the base quaternion via
     `quat_wxyz_to_roll_pitch`;
   * `body_contact` from the environment's `nonfoot_ground_touch` contact
     sensor (base/knee/thigh/calf geoms against the terrain, feet excluded),
     using `found` plus the 4-substep `force_history` so a contact that ended
     before the control-step boundary is still seen;
   * `env_done` / `env_reason` from the environment's termination terms
     (`illegal_contact → body_contact`, `fell_over → orientation`,
     `time_out → timeout`, anything else → `reset`).
5. **Verdict.** `TrialEvaluator.update()` is called once per step and returns
   the `TrialResult` exactly once, when the trial becomes a success or a
   failure. The CLI never overrides that verdict.
6. **Headless / video.** Rendering is off by default. With `--video-dir DIR`
   the environment is created with `render_mode="rgb_array"` and wrapped in the
   repository's `VideoRecorder` (as in `play.py`); one MP4 is written per trial.
   `--no-render` forces headless even if `--video-dir` is given.
7. **Determinism.** With `--seed S`, trial `i` uses seed `S + i`: the
   environment RNG is re-seeded before each reset, which reproduces terrain
   randomisation and `reset_base` sampling. Without `--seed` a random base seed
   is chosen and printed so the run can be repeated later.
8. **Robustness.** A trial that never reaches the platform cannot hang: the
   evaluator fails it on `stuck_timeout_s` (6 s without 0.05 m of new forward
   progress) or on `episode_timeout_s` (30 s), and a belt-and-braces step cap
   closes out anything pathological. Per-trial runtime exceptions are caught,
   reported in `notes`, and recorded as a failed row.

## Mandated geometry

`assert_mandated_geometry()` asserts these values on the criteria geometry at
start-up; the terrain is built by
`src/tasks/velocity/mdp/stairs_terrain.py::BoxStraightStairsTerrainCfg`.

| Quantity | Value |
| --- | --- |
| Number of continuous steps | **4** |
| Step rise | **0.13 m** |
| Step tread along `+x` | **0.30 m** |
| Total rise (`4 × 0.13`) | **0.52 m** |
| Flat approach in front of the first riser | 1.6 m |
| Flat top platform after the last step | 1.7 m |
| Single patch | 4.5 m × 4.5 m |
| Spawn | on the approach, 0.9 m in front of the first riser, yaw = 0, facing `+x` |

In the course frame the first riser is at `x = 0.9 m` and the platform starts
at `x = 2.1 m` and ends at `x = 3.8 m`.

## Success criteria (assessment §4.3)

A trial is a **success only if all** of the following hold:

1. the main body crosses the stairs and reaches the top platform
   (`x ≥ 2.1 m` **and** base height `z ≥ 0.52 − 0.10 = 0.42 m`, i.e. the
   platform-height threshold, not just crossing the platform's `x`);
2. after first reaching the platform it keeps moving forward at least **1.0 m**
   (measured from the platform-entry `x`, using the running maximum `x`);
3. body / knees / other non-permitted parts do **not** persistently contact the
   ground or steps (contact shorter than `body_contact_grace_s = 0.2 s` is
   tolerated);
4. no fall condition is triggered.

## Failure criteria (any one)

| Condition | `failure_reason` |
| --- | --- |
| Body/knee sustained contact with ground/steps beyond 0.2 s, or the environment's hard `illegal_contact` (>10 N) termination | `body_contact` |
| `\|roll\| > 45°` continuously for more than 0.5 s, or the environment's `fell_over` termination | `roll_exceeded` |
| `\|pitch\| > 45°` continuously for more than 0.5 s, or `fell_over` | `pitch_exceeded` |
| No forward progress beyond 0.05 m for more than 6 s | `stuck` |
| Simulated time reaches 30 s without a decision | `timeout` |
| Episode ends / resets before the course is completed | `reset` |
| Unexpected runtime exception in the CLI adapter | `error` |

The command is constant and the terrain/criteria are never changed mid-trial;
the harness never drags or teleports the robot.

## `results.csv`

Exactly these columns, in this order:

```text
trial_id, seed, command_speed, success, elapsed_time,
max_abs_roll, max_abs_pitch, failure_reason, notes
```

| Column | Meaning |
| --- | --- |
| `trial_id` | 0-based episode index |
| `seed` | seed actually used for this trial (`base + trial_id`) |
| `command_speed` | constant forward command read from the twist command (m/s) |
| `success` | `True` / `False` (the evaluator's verdict) |
| `elapsed_time` | simulated seconds until the verdict (s) |
| `max_abs_roll` | peak `\|roll\|` during the trial, **degrees** |
| `max_abs_pitch` | peak `\|pitch\|` during the trial, **degrees** |
| `failure_reason` | token from the table above (`ok` on success) |
| `notes` | `steps=`, `max_x=`, `platform_entry_x=`, `post_platform_travel=` summary |

> The pure logic stores roll/pitch in radians (`TrialResult.max_abs_roll`).
> The CSV deliberately writes **degrees** so the values are directly
> comparable with the assessment's 45° limit.

## Reproducing

Environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab
export MPLCONFIGDIR=/tmp/mpl
export WARP_CACHE_PATH=/home/xux/RC/.warpcache
```

Exact command for the hard "达标" tier (0.4 m/s, 5 trials):

```bash
python scripts/eval_stairs.py Quadruped-3320-Stairs \
  --checkpoint logs/rsl_rl/quadruped_3320_velocity/2026-10-13_00-09-50_stairs_768_cfg_v1/model_0.pt \
  --trials 5 --out results.csv --seed 0
```

Higher tiers use the identical terrain/criteria with only the constant command
speed changed:

```bash
python scripts/eval_stairs.py Quadruped-3320-Stairs-0.6 --checkpoint <run dir or .pt> \
  --trials 5 --out results_0_6.csv --seed 0
python scripts/eval_stairs.py Quadruped-3320-Stairs-0.8 --checkpoint <run dir or .pt> \
  --trials 5 --out results_0_8.csv --seed 0
```

Optional continuous video (one MP4 per trial):

```bash
python scripts/eval_stairs.py Quadruped-3320-Stairs \
  --checkpoint <ckpt> --trials 5 --out results.csv --seed 0 \
  --video-dir logs/rsl_rl/quadruped_3320_velocity/eval_videos
```

### Exit code

`0` on success. At the 0.4 m/s tier the process exits `1` when fewer than
`ceil(0.8 × trials)` (4 of 5) episodes succeed — the assessment's hard gate.
Tiers other than 0.4 m/s still print the summary and write the CSV but do not
fail the process. `2` is returned for argument / unknown-task errors.

## Verification status

Verified without a GPU / without building the environment:

* `python -m pytest tests/test_stair_eval.py -q` — 24 passed (pure logic).
* `python scripts/eval_stairs.py --help` — full CLI help.
* `python -c "import ast; ast.parse(open('scripts/eval_stairs.py').read())"`.
* Dry import of the module, checkpoint resolution for a `.pt`, a run directory
  and a log root, and unknown-task rejection.

Not runtime-verified here (a training run occupies the GPU): live environment
construction, policy inference, contact-sensor reads and video recording. See
the CLI's `[INFO]` output and `results.csv` for the first real run.
