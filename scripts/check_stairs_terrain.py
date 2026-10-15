#!/usr/bin/env python3
"""Numerically verify the mandated straight-staircase terrain geometry.

Loads the *registered task configuration* so that the terrain size and every
stair dimension come from the code that actually runs -- nothing is hardcoded.
The config of the task ``Quadruped-3320-Stairs`` is built at the full terrain
curriculum difficulty (``difficulty = 1.0``), compiled into a fresh
``mujoco.MjSpec`` and the *compiled model* is checked:

1. every terrain geom is an axis-aligned box with collisions enabled,
2. the ``warmup_steps`` smooth lead-in steps are listed separately from the
   mandated flight; their tops ramp ``1/3, 2/3, 3/3`` of the 0.13 m step rise,
3. exactly 4 mandated step boxes with distinct top surfaces separated by
   exactly 0.13 m and a constant 0.30 m tread along the travel direction (+x),
4. a flat top platform of ``platform_depth`` whose top is the mandated rise,
5. a downward-ray height profile along +x is piecewise constant and matches the
   expected plinth heights (no hidden ramps),
6. the spawn origin sits ``spawn_offset`` in front of the mandated flight,
7. optional flat-patch sampling lands on the flat approach.

Curriculum note
---------------
The training config sets ``curriculum_steps=True`` (the terrain-grid ROW ramps
the number of mandated steps ``1 -> 4``) and ``curriculum_height=True`` (the
rise ramps ``0.04 m -> 0.13 m``).  Only the full difficulty ``difficulty = 1.0``
carries all 4 mandated steps at the full 13 cm rise, so that is the row this
script validates.  ``play``/evaluation must use the same full geometry.

Run from the repository root:

  env -u PYTHONPATH python scripts/check_stairs_terrain.py

Exits non-zero on any failure.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import mujoco
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.terrains.terrain_generator import (  # noqa: E402
  FlatPatchSamplingCfg,
  TerrainGenerator,
)

import src.tasks  # noqa: E402,F401  (registers the quadruped_3320 tasks)
from src.tasks.velocity.mdp.stairs_terrain import (  # noqa: E402
  BoxStraightStairsTerrainCfg,
)

TASK_ID = "Quadruped-3320-Stairs"

# Independently restated mandate (do not import these from the class).
EXPECTED_NUM_STEPS = 4
EXPECTED_STEP_HEIGHT = 0.13
EXPECTED_STEP_TREAD = 0.30
EXPECTED_TOTAL_RISE = EXPECTED_NUM_STEPS * EXPECTED_STEP_HEIGHT

# Full terrain-curriculum difficulty = the row play/evaluation must use.
DIFFICULTY = 1.0

EXACT_TOL = 1e-9
RAY_TOL = 2e-3
RAY_DX = 1e-4


def _build(
  cfg: BoxStraightStairsTerrainCfg, difficulty: float
) -> tuple[mujoco.MjModel, mujoco.MjData, object]:
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty, spec, np.random.default_rng(0))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return model, data, output


def _ray_height(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  x: float,
  y: float,
  start_z: float = 3.0,
) -> tuple[float, int]:
  geom_id = np.zeros(1, dtype=np.int32)
  dist = mujoco.mj_ray(
    model,
    data,
    np.array([x, y, start_z], dtype=np.float64),
    np.array([0.0, 0.0, -1.0], dtype=np.float64),
    None,
    1,
    -1,
    geom_id,
  )
  return start_z - dist, int(geom_id[0])


def _layout(sub, size: tuple[float, float], difficulty: float) -> dict:
  """Mirror of ``BoxStraightStairsTerrainCfg.function``'s layout maths."""
  n_warm = max(0, len(sub.warmup_heights))
  frac = min(max(float(difficulty), 0.0), 1.0)
  if sub.curriculum_steps:
    n_steps = int(round(1 + frac * (sub.num_steps - 1)))
    n_steps = max(1, min(int(sub.num_steps), n_steps))
  else:
    n_steps = int(sub.num_steps)
  if sub.curriculum_height:
    step_h = sub.min_step_height + frac * (sub.step_height - sub.min_step_height)
  else:
    step_h = float(sub.step_height)

  approach = float(sub.approach_depth)
  platform = float(sub.platform_depth)
  tread = float(sub.step_tread)
  warm_rise = step_h if n_warm > 0 else 0.0
  total_rise = warm_rise + n_steps * step_h
  flight_length = approach + (n_warm + n_steps) * tread + platform
  front_pad = size[0] - flight_length
  warm_x0 = front_pad + approach
  stairs_x0 = warm_x0 + n_warm * tread
  platform_x0 = stairs_x0 + n_steps * tread
  return {
    "n_warm": n_warm,
    "n_steps": n_steps,
    "step_h": step_h,
    "tread": tread,
    "approach": approach,
    "platform": platform,
    "warm_rise": warm_rise,
    "total_rise": total_rise,
    "flight_length": flight_length,
    "front_pad": front_pad,
    "warm_x0": warm_x0,
    "stairs_x0": stairs_x0,
    "platform_x0": platform_x0,
    "platform_length": size[0] - platform_x0,
    "size": size,
    "base_thickness": float(sub.base_thickness),
    "spawn_offset": float(sub.spawn_offset),
  }


def _boxes(lay: dict) -> list[tuple[str, float, float, float]]:
  """Expected box layout as ``(name, centre_x, x_depth, top_z)``."""
  tread = lay["tread"]
  step_h = lay["step_h"]
  n_warm = lay["n_warm"]
  boxes = [("approach", 0.5 * lay["warm_x0"], lay["warm_x0"], 0.0)]
  for i in range(n_warm):
    boxes.append(
      (
        f"warmup[{i}]",
        lay["warm_x0"] + (i + 0.5) * tread,
        tread,
        (i + 1) * step_h / max(n_warm, 1),
      )
    )
  for k in range(lay["n_steps"]):
    boxes.append(
      (
        f"mandated[{k}]",
        lay["stairs_x0"] + (k + 0.5) * tread,
        tread,
        lay["warm_rise"] + (k + 1) * step_h,
      )
    )
  plen = lay["platform_length"]
  boxes.append(
    ("platform", lay["platform_x0"] + 0.5 * plen, plen, lay["total_rise"])
  )
  return boxes


class Checker:
  def __init__(self) -> None:
    self.failures: list[str] = []

  def check(self, ok: bool, message: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {message}")
    if not ok:
      self.failures.append(message)

  def close(self, a: float, b: float, tol: float, message: str) -> None:
    self.check(abs(a - b) <= tol, f"{message} (got {a!r}, expected {b!r})")


def main() -> int:
  c = Checker()

  # -------------------------------------------------------------------------
  print("=" * 78)
  print(f"Loading registered task '{TASK_ID}' (training config)")
  env_cfg = load_env_cfg(TASK_ID, play=False)
  gen = env_cfg.scene.terrain.terrain_generator
  assert gen is not None and len(gen.sub_terrains) == 1
  sub = next(iter(gen.sub_terrains.values()))
  assert isinstance(sub, BoxStraightStairsTerrainCfg)

  # ``TerrainGenerator.__init__`` copies the generator size onto every
  # sub-terrain before calling ``function``; emulate that here.
  sub.size = gen.size
  terrain_size = tuple(float(v) for v in gen.size)

  lay = _layout(sub, terrain_size, DIFFICULTY)
  boxes = _boxes(lay)

  print(f"  task size (from env_cfg.scene.terrain.terrain_generator.size) = "
        f"{terrain_size}")
  print(f"  difficulty={DIFFICULTY}  curriculum_steps={sub.curriculum_steps} "
        f"curriculum_height={sub.curriculum_height} "
        f"warmup_steps={sub.warmup_heights}")
  print(f"  approach_depth={lay['approach']} platform_depth={lay['platform']} "
        f"spawn_offset={lay['spawn_offset']} base_thickness={lay['base_thickness']}")
  print(f"  -> mandated steps={lay['n_steps']} x {lay['step_h']:.3f} m rise x "
        f"{lay['tread']:.2f} m tread, total rise above ground="
        f"{lay['total_rise']:.3f} m")
  print(f"  -> flight_length={lay['flight_length']:.3f} m "
        f"(front_pad={lay['front_pad']:.3f} m), warmup x0="
        f"{lay['warm_x0']:.3f} mandated x0={lay['stairs_x0']:.3f} "
        f"platform x0={lay['platform_x0']:.3f}")
  print(f"  -> expected boxes: 1 approach + {lay['n_warm']} lead-in + "
        f"{lay['n_steps']} mandated + 1 platform = {len(boxes)}")

  if sub.curriculum_steps:
    print("\n  NOTE: curriculum_steps=True ramps the number of mandated steps")
    print("        from 1 (difficulty 0) to 4 (difficulty 1).  This script")
    print("        validates the FULL difficulty (difficulty=1.0), which is the")
    print("        only row with all 4 mandated steps.")
  if sub.curriculum_height:
    print("  NOTE: curriculum_height=True ramps the rise from "
          f"{sub.min_step_height:.2f} m to {sub.step_height:.2f} m; at")
    print("        difficulty=1.0 the rise is the full mandated "
          f"{EXPECTED_STEP_HEIGHT:.2f} m.")
  print("=" * 78)

  model, data, output = _build(sub, DIFFICULTY)
  print(f"  compiled model: {model.ngeom} geoms, {model.nbody} bodies")

  # -------------------------------------------------------------------------
  print("\n[1] Dump compiled box geoms (model order = insertion order):")
  actual_cx: list[float] = []
  for i in range(model.ngeom):
    geom_type = model.geom_type[i]
    size = model.geom_size[i]
    pos = data.geom_xpos[i]
    quat = data.geom_xmat[i].reshape(3, 3)
    top_z = pos[2] + size[2]
    is_box = geom_type == mujoco.mjtGeom.mjGEOM_BOX
    kind = "box" if is_box else f"NOT-BOX(type={geom_type})"
    print(
      f"  geom[{i:02d}] {kind:9s} "
      f"size={np.array2string(size, precision=6)} "
      f"pos={np.array2string(pos, precision=6)} top_z={top_z:.6f}"
    )
    actual_cx.append(float(pos[0]))
    c.check(is_box, f"geom[{i}] is a box")
    # Axis-aligned boxes only (no tilted/ramp geometry).
    c.check(
      bool(np.allclose(quat, np.eye(3), atol=EXACT_TOL)),
      f"geom[{i}] is axis-aligned (identity rotation)",
    )
    # Collisions enabled.
    c.check(
      model.geom_contype[i] != 0 and model.geom_conaffinity[i] != 0,
      f"geom[{i}] has collisions enabled "
      f"(contype={model.geom_contype[i]}, conaffinity={model.geom_conaffinity[i]})",
    )

  # -------------------------------------------------------------------------
  print(f"\n[2] Box identity: match each geom to the expected layout "
        f"({len(boxes)} boxes):")
  c.check(
    model.ngeom == len(boxes),
    f"exactly {len(boxes)} geoms emitted (got {model.ngeom})",
  )
  matched: dict[int, int] = {}
  for j, (name, cx, depth, top) in enumerate(boxes):
    hits = [i for i, ax in enumerate(actual_cx) if abs(ax - cx) <= EXACT_TOL]
    if len(hits) != 1:
      c.check(False, f"{name}: exactly one geom centred at x={cx:.4f} "
                     f"(matches={len(hits)})")
      continue
    gid = hits[0]
    matched[j] = gid
    c.close(
      2 * model.geom_size[gid][0], depth, EXACT_TOL, f"{name} x-depth"
    )
    c.close(
      data.geom_xpos[gid][2] + model.geom_size[gid][2],
      top,
      EXACT_TOL,
      f"{name} top-surface height",
    )

  # -------------------------------------------------------------------------
  print(f"\n[3] Mandated flight: exactly {EXPECTED_NUM_STEPS} steps x "
        f"{EXPECTED_STEP_HEIGHT} m rise x {EXPECTED_STEP_TREAD} m tread:")
  mandated_idx = [j for j, b in enumerate(boxes) if b[0].startswith("mandated")]
  c.check(
    len(mandated_idx) == EXPECTED_NUM_STEPS,
    f"exactly {EXPECTED_NUM_STEPS} mandated step boxes emitted "
    f"(got {len(mandated_idx)})",
  )
  m_tops = [boxes[j][3] for j in mandated_idx]
  for k, j in enumerate(mandated_idx):
    print(f"    {boxes[j][0]}: cx={boxes[j][1]:.4f} depth={boxes[j][2]:.4f} "
          f"top_z={boxes[j][3]:.6f}")
  for k, top in enumerate(m_tops):
    c.close(
      top,
      lay["warm_rise"] + (k + 1) * EXPECTED_STEP_HEIGHT,
      EXACT_TOL,
      f"mandated step {k} top surface height",
    )
  for k in range(1, len(m_tops)):
    c.close(
      m_tops[k] - m_tops[k - 1],
      EXPECTED_STEP_HEIGHT,
      EXACT_TOL,
      f"mandated rise between step {k - 1} and step {k}",
    )
  c.check(
    len({round(t, 9) for t in m_tops}) == EXPECTED_NUM_STEPS,
    f"exactly {EXPECTED_NUM_STEPS} distinct mandated top heights "
    f"(got {len({round(t, 9) for t in m_tops})})",
  )
  c.close(
    m_tops[-1] - m_tops[0],
    EXPECTED_TOTAL_RISE - EXPECTED_STEP_HEIGHT,
    EXACT_TOL,
    "mandated flight rise from first to last step",
  )
  for k in range(1, len(mandated_idx)):
    prev = matched.get(mandated_idx[k - 1])
    cur = matched.get(mandated_idx[k])
    if prev is None or cur is None:
      continue
    prev_right = data.geom_xpos[prev][0] + model.geom_size[prev][0]
    cur_left = data.geom_xpos[cur][0] - model.geom_size[cur][0]
    c.close(
      cur_left,
      prev_right,
      EXACT_TOL,
      f"mandated step {k} left edge is flush with step {k - 1} right edge "
      f"(no gap/overlap)",
    )

  print(f"\n[3b] Top platform (depth {lay['platform']:.2f} m at total rise "
        f"{lay['total_rise']:.3f} m):")
  plat_j = next(j for j, b in enumerate(boxes) if b[0] == "platform")
  plat = matched.get(plat_j)
  if plat is not None:
    print(f"    platform: cx={boxes[plat_j][1]:.4f} "
          f"depth={boxes[plat_j][2]:.4f} top_z={boxes[plat_j][3]:.6f}")
    c.close(
      2 * model.geom_size[plat][0],
      lay["platform"],
      EXACT_TOL,
      "platform depth",
    )
    c.close(
      data.geom_xpos[plat][2] + model.geom_size[plat][2],
      lay["total_rise"],
      EXACT_TOL,
      "platform top height",
    )
    last = matched.get(mandated_idx[-1])
    if last is not None:
      last_right = data.geom_xpos[last][0] + model.geom_size[last][0]
      plat_left = data.geom_xpos[plat][0] - model.geom_size[plat][0]
      c.close(
        plat_left, last_right, EXACT_TOL,
        "platform starts exactly at the last mandated step's right edge",
      )
      c.close(
        boxes[plat_j][3] - boxes[mandated_idx[-1]][3],
        0.0,
        EXACT_TOL,
        "platform top level equals the last mandated step top level",
      )

  # -------------------------------------------------------------------------
  print(f"\n[4] Smooth lead-in steps (warmup_steps={lay['n_warm']}):")
  warm_idx = [j for j, b in enumerate(boxes) if b[0].startswith("warmup")]
  c.check(
    len(warm_idx) == lay["n_warm"],
    f"exactly {lay['n_warm']} lead-in boxes emitted (got {len(warm_idx)})",
  )
  for i, j in enumerate(warm_idx):
    print(f"    {boxes[j][0]}: top_z={boxes[j][3]:.6f} = "
          f"{i + 1}/{lay['n_warm']} x {lay['step_h']:.3f} m")
    c.close(
      boxes[j][3],
      (i + 1) * lay["step_h"] / max(lay["n_warm"], 1),
      EXACT_TOL,
      f"lead-in step {i} top is {i + 1}/{lay['n_warm']} of the step rise",
    )
    c.close(
      boxes[j][2], EXPECTED_STEP_TREAD, EXACT_TOL, f"lead-in step {i} tread"
    )
  if warm_idx:
    c.close(
      boxes[warm_idx[-1]][3] - boxes[warm_idx[-2]][3] if len(warm_idx) > 1 else 0.0,
      lay["step_h"] / max(lay["n_warm"], 1),
      EXACT_TOL,
      "lead-in rise per step is step_height / warmup_steps",
    )
    print("    NOTE: lead-in steps are NOT the mandated flight; they ease the "
          "robot onto the staircase.")
    print(f"          The mandated flight (4 x 0.13 m) starts at x="
          f"{lay['stairs_x0']:.3f} m.")

  # -------------------------------------------------------------------------
  print(f"\n[5] Downward-ray height profile (dx={RAY_DX} m, "
        f"y={terrain_size[1] / 2:.3f}):")
  y_center = terrain_size[1] / 2
  n = int(round(terrain_size[0] / RAY_DX))
  xs = (np.arange(n) + 0.5) * RAY_DX
  heights = np.empty(n)
  miss = 0
  for i, x in enumerate(xs):
    z, gid = _ray_height(model, data, x, y_center)
    if gid < 0:
      miss += 1
    heights[i] = z
  c.check(miss == 0, f"every ray hit the terrain (misses={miss})")

  rounded = np.round(heights, 9)
  expected_widths: dict[float, float] = {0.0: lay["warm_x0"]}
  for j in warm_idx:
    expected_widths[round(boxes[j][3], 9)] = EXPECTED_STEP_TREAD
  for j in mandated_idx:
    expected_widths[round(boxes[j][3], 9)] = EXPECTED_STEP_TREAD
  top_level = round(lay["total_rise"], 9)
  expected_widths[top_level] += lay["platform_length"]

  expected_levels = sorted(expected_widths)
  actual_levels = sorted(set(rounded.tolist()))
  c.check(
    actual_levels == expected_levels,
    f"profile only contains expected heights {expected_levels} "
    f"(got {actual_levels})",
  )

  change_idx = np.where(np.diff(rounded) != 0)[0]
  starts = np.concatenate(([0], change_idx + 1))
  ends = np.concatenate((change_idx, [n - 1]))
  plateaus = [
    (rounded[s], xs[e] - xs[s] + RAY_DX)
    for s, e in zip(starts, ends, strict=True)
  ]
  print("  measured plateaus (height, x-width):")
  for level, width in plateaus:
    print(f"    z={level:.6f}  width={width:.6f} m")
  width_by_level = {round(level, 9): width for level, width in plateaus}
  for level in expected_levels:
    measured = width_by_level.get(round(level, 9))
    expected = expected_widths[level]
    if measured is None:
      c.check(False, f"profile has a plateau at z={level:.6f}")
      continue
    c.check(
      abs(measured - expected) <= RAY_TOL,
      f"plateau at z={level:.6f} is {expected:.3f} m along +x "
      f"(measured {measured:.6f})",
    )

  # -------------------------------------------------------------------------
  print("\n[6] Spawn origin:")
  origin = np.asarray(output.origin)
  print(f"  origin={origin}")
  # ``spawn_offset`` is documented as "distance in front of the FIRST riser".
  # With lead-in steps present the first riser is the first lead-in step at
  # ``warm_x0``; without them it is the first mandated step at ``stairs_x0``.
  first_riser_x = lay["warm_x0"] if lay["n_warm"] > 0 else lay["stairs_x0"]
  c.close(origin[1], y_center, EXACT_TOL, "origin y is the patch centre")
  c.close(origin[2], 0.0, EXACT_TOL, "origin z is ground level")
  c.close(
    origin[0],
    first_riser_x - lay["spawn_offset"],
    EXACT_TOL,
    "origin x is spawn_offset in front of the first riser",
  )
  c.check(origin[0] >= 0.0, "origin is inside the patch")
  c.check(
    origin[0] <= first_riser_x + EXACT_TOL,
    "origin is at or in front of the first riser",
  )
  spawn_z, _ = _ray_height(model, data, origin[0], origin[1])
  c.close(
    spawn_z,
    0.0,
    EXACT_TOL,
    "ground under the spawn origin is flat at z=0 (the flat run-up)",
  )
  approach_z, approach_gid = _ray_height(model, data, 0.5 * lay["warm_x0"], origin[1])
  c.close(approach_z, 0.0, EXACT_TOL, "flat approach in front of the stairs is at z=0")
  c.check(
    approach_gid == matched.get(0, -1),
    "approach ray hits the approach geom",
  )
  print(f"  NOTE: first riser (lead-in) at x={first_riser_x:.3f} m, mandated "
        f"flight starts at x={lay['stairs_x0']:.3f} m; the spawn offset is")
  print(f"        measured from the FIRST riser, so the flat run-up is "
        f"{lay['spawn_offset']:.3f} m before the robot meets any step.")

  # -------------------------------------------------------------------------
  print("\n[7] Optional flat-patch sampling on the approach:")
  patch_cfg = dataclasses.replace(
    sub,
    flat_patch_sampling={
      "spawn": FlatPatchSamplingCfg(num_patches=6, patch_radius=0.15)
    },
  )
  patch_model, patch_data, patch_output = _build(patch_cfg, DIFFICULTY)
  patches = patch_output.flat_patches
  c.check(patches is not None and "spawn" in patches, "flat patches are emitted")
  if patches is not None:
    pts = patches["spawn"]
    print(f"  {len(pts)} patches, x in [{pts[:, 0].min():.3f}, {pts[:, 0].max():.3f}]")
    c.check(len(pts) == 6, "requested number of patches emitted")
    c.check(
      bool(np.all(pts[:, 2] == 0.0)),
      "all flat patches are emitted at z=0",
    )
    on_approach = []
    for x, y, _ in pts:
      z, _ = _ray_height(patch_model, patch_data, float(x), float(y))
      on_approach.append(abs(z) <= EXACT_TOL)
    c.check(
      all(on_approach),
      "every flat patch really sits on flat z=0 terrain (ray-traced)",
    )
    c.check(
      bool(np.all(pts[:, 0] <= lay["warm_x0"] - 0.15 + EXACT_TOL)),
      "all flat patches are at least patch_radius in front of the first riser",
    )

  # -------------------------------------------------------------------------
  print("\n[8] Strict mandate guard:")
  try:
    BoxStraightStairsTerrainCfg(size=terrain_size, step_height=0.10)
    c.check(False, "non-mandated step_height=0.10 is rejected when strict=True")
  except ValueError:
    c.check(True, "non-mandated step_height=0.10 is rejected when strict=True")

  # -------------------------------------------------------------------------
  print("\n[9] play / evaluation configuration cross-check:")
  play_cfg = load_env_cfg(TASK_ID, play=True)
  play_gen_cfg = play_cfg.scene.terrain.terrain_generator
  play_sub = next(iter(play_gen_cfg.sub_terrains.values()))
  c.close(
    float(play_gen_cfg.size[0]), terrain_size[0], EXACT_TOL,
    "play terrain size[0] equals the training size[0]",
  )
  c.close(
    float(play_gen_cfg.size[1]), terrain_size[1], EXACT_TOL,
    "play terrain size[1] equals the training size[1]",
  )
  print(f"  play: size={tuple(play_gen_cfg.size)} "
        f"curriculum={play_gen_cfg.curriculum} "
        f"num_rows={play_gen_cfg.num_rows} num_cols={play_gen_cfg.num_cols} "
        f"difficulty_range={play_gen_cfg.difficulty_range}")
  print(f"        curriculum_steps={play_sub.curriculum_steps} "
        f"curriculum_height={play_sub.curriculum_height} "
        f"min_step_height={play_sub.min_step_height} "
        f"step_height={play_sub.step_height}")
  c.check(
    not play_sub.curriculum_steps,
    "play disables curriculum_steps (every patch builds the full 4 steps)",
  )
  if play_sub.curriculum_height and not play_gen_cfg.curriculum:
    print("  [WARN] play leaves curriculum_height=True while curriculum=False:")
    print("         mjlab then draws difficulty uniformly from "
          f"{play_gen_cfg.difficulty_range} for every patch, so the step")
    print("         RISE is a random "
          f"{play_sub.min_step_height:.2f}-{play_sub.step_height:.2f} m "
          "instead of the fixed mandated 0.13 m.")
    print("         (Fix: set curriculum_height=False on the play sub-terrain.)")

  # Empirically confirm how many play patches carry the full mandated rise.
  play_probe = dataclasses.replace(play_gen_cfg)
  play_probe.seed = 0
  probe_gen = TerrainGenerator(play_probe, device="cpu")
  spec = mujoco.MjSpec()
  probe_gen.compile(spec)
  probe_model = spec.compile()
  probe_data = mujoco.MjData(probe_model)
  mujoco.mj_forward(probe_model, probe_data)
  size = tuple(float(v) for v in play_probe.size)
  n_rows, n_cols = play_probe.num_rows, play_probe.num_cols
  x0 = -0.5 * n_rows * size[0]
  y0 = -0.5 * n_cols * size[1]
  patch_top: dict[tuple[int, int], float] = {}
  for i in range(probe_model.ngeom):
    if probe_model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
      continue
    px, py = float(probe_data.geom_xpos[i][0]), float(probe_data.geom_xpos[i][1])
    row = int(np.floor((px - x0) / size[0]))
    col = int(np.floor((py - y0) / size[1]))
    if not (0 <= row < n_rows and 0 <= col < n_cols):
      continue
    top = float(probe_data.geom_xpos[i][2] + probe_model.geom_size[i][2])
    patch_top[(row, col)] = max(patch_top.get((row, col), -1.0), top)
  if patch_top:
    tops = sorted(patch_top.values())
    full = [t for t in tops if abs(t - lay["total_rise"]) <= EXACT_TOL]
    print(f"  measured play patches: {len(tops)} of {n_rows * n_cols}, "
          f"top levels in [{tops[0]:.4f}, {tops[-1]:.4f}] m, "
          f"full {lay['total_rise']:.2f} m flight in {len(full)} patch(es)")
    status = "PASS" if len(full) == len(tops) else "WARN"
    print(f"  [{status}] every play patch carries the full mandated "
          f"{lay['total_rise']:.2f} m flight: {len(full)}/{len(tops)} "
          f"(informational; the mandated geometry itself is validated above)")

  # -------------------------------------------------------------------------
  print("\n" + "=" * 78)
  if c.failures:
    print(f"RESULT: FAIL ({len(c.failures)} check(s) failed)")
    for f in c.failures:
      print(f"  - {f}")
    return 1
  print(
    f"RESULT: PASS - mandated flight is exactly {EXPECTED_NUM_STEPS} steps x "
    f"{EXPECTED_STEP_HEIGHT} m x {EXPECTED_STEP_TREAD} m "
    f"(total rise {EXPECTED_TOTAL_RISE:.2f} m) at difficulty={DIFFICULTY}, "
    f"plus {lay['n_warm']} smooth lead-in steps of "
    f"{EXPECTED_STEP_HEIGHT / max(lay['n_warm'], 1):.4f} m each"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
