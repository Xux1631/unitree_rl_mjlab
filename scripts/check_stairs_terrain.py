#!/usr/bin/env python3
"""Numerically verify the mandated straight-staircase terrain geometry.

Builds :class:`BoxStraightStairsTerrainCfg` into a fresh ``mujoco.MjSpec``,
compiles it, and checks the *compiled model* rather than the source config:

1. every terrain geom is an axis-aligned box with collisions enabled,
2. exactly 4 step boxes with distinct top surfaces separated by exactly 0.13 m,
3. each step's tread depth along the travel direction (+x) is exactly 0.30 m,
4. a downward-ray height profile along +x is piecewise constant and matches the
   expected plinth heights (no hidden ramps),
5. the spawn origin sits on the flat approach in front of the first riser, and
   the optional flat-patch sampling lands on the approach too.

Run from the repository root:

  python scripts/check_stairs_terrain.py

Exits non-zero on any failure.
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg  # noqa: E402

from src.tasks.velocity.mdp.stairs_terrain import (  # noqa: E402
  BoxStraightStairsTerrainCfg,
)

# Independently restated mandate (do not import these from the class).
EXPECTED_NUM_STEPS = 4
EXPECTED_STEP_HEIGHT = 0.13
EXPECTED_STEP_TREAD = 0.30
EXPECTED_TOTAL_RISE = EXPECTED_NUM_STEPS * EXPECTED_STEP_HEIGHT

# Sub-terrain the A2 rough task uses (env_cfgs.py sets generator size to this).
TERRAIN_SIZE = (4.5, 4.5)
APPROACH_DEPTH = 1.6
PLATFORM_DEPTH = 1.7
SPAWN_OFFSET = 0.9

EXACT_TOL = 1e-9
RAY_TOL = 2e-3
RAY_DX = 1e-4


def _build(
  cfg: BoxStraightStairsTerrainCfg,
) -> tuple[mujoco.MjModel, mujoco.MjData, object]:
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(0))
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

  print("=" * 78)
  print("Building BoxStraightStairsTerrainCfg")
  print(f"  size={TERRAIN_SIZE} approach_depth={APPROACH_DEPTH} "
        f"platform_depth={PLATFORM_DEPTH} spawn_offset={SPAWN_OFFSET}")
  cfg = BoxStraightStairsTerrainCfg(
    size=TERRAIN_SIZE,
    approach_depth=APPROACH_DEPTH,
    platform_depth=PLATFORM_DEPTH,
    spawn_offset=SPAWN_OFFSET,
  )
  model, data, output = _build(cfg)
  print(f"  compiled model: {model.ngeom} geoms, {model.nbody} bodies")
  print("=" * 78)

  # -------------------------------------------------------------------------
  print("\n[1] Dump compiled box geoms (model order = insertion order):")
  approach_id = None
  step_ids: list[int] = []
  platform_id = None
  step_top_by_id: dict[int, float] = {}
  for i in range(model.ngeom):
    geom_type = model.geom_type[i]
    size = model.geom_size[i]
    pos = data.geom_xpos[i]
    quat = data.geom_xmat[i].reshape(3, 3)
    top_z = pos[2] + size[2]
    is_box = geom_type == mujoco.mjtGeom.mjGEOM_BOX
    kind = "box" if is_box else f"NOT-BOX(type={geom_type})"
    label = "?"
    if is_box and abs(2 * size[0] - EXPECTED_STEP_TREAD) < EXACT_TOL:
      label = "STEP"
      step_ids.append(i)
      step_top_by_id[i] = top_z
    elif is_box and abs(top_z - EXPECTED_TOTAL_RISE) < EXACT_TOL:
      label = "PLATFORM"
      platform_id = i
    elif is_box and abs(top_z) < EXACT_TOL:
      label = "APPROACH"
      approach_id = i
    print(
      f"  geom[{i:02d}] {kind:9s} {label:8s} "
      f"size={np.array2string(size, precision=6)} "
      f"pos={np.array2string(pos, precision=6)} top_z={top_z:.6f}"
    )
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
  print(f"\n[2] Step count and top-surface heights (expected {EXPECTED_NUM_STEPS} "
        f"steps, rise {EXPECTED_STEP_HEIGHT}):")
  c.check(
    len(step_ids) == EXPECTED_NUM_STEPS,
    f"exactly {EXPECTED_NUM_STEPS} step boxes of tread "
    f"{EXPECTED_STEP_TREAD} m found (got {len(step_ids)})",
  )

  steps_sorted = sorted(step_ids, key=lambda i: data.geom_xpos[i][0])
  step_tops = [step_top_by_id[i] for i in steps_sorted]
  for k, (gid, top) in enumerate(zip(steps_sorted, step_tops, strict=True)):
    print(f"    step {k}: geom[{gid}] top_z={top:.9f}")
  expected_tops = [
    round((k + 1) * EXPECTED_STEP_HEIGHT, 9) for k in range(len(step_tops))
  ]
  for k, (top, exp) in enumerate(zip(step_tops, expected_tops, strict=False)):
    c.close(top, exp, EXACT_TOL, f"step {k} top surface height")
  distinct = sorted({round(t, 9) for t in step_tops})
  c.check(
    len(distinct) == EXPECTED_NUM_STEPS,
    f"exactly {EXPECTED_NUM_STEPS} distinct step top heights (got {distinct})",
  )
  for k in range(1, len(distinct)):
    c.close(
      distinct[k] - distinct[k - 1],
      EXPECTED_STEP_HEIGHT,
      EXACT_TOL,
      f"rise between step height {distinct[k - 1]:.3f} and {distinct[k]:.3f}",
    )
  c.close(
    distinct[-1] - 0.0,
    EXPECTED_TOTAL_RISE,
    EXACT_TOL,
    "total rise above approach",
  )

  # -------------------------------------------------------------------------
  print("\n[3] Tread depth along the travel direction (+x):")
  for k, gid in enumerate(steps_sorted):
    size = model.geom_size[gid]
    c.close(2 * size[0], EXPECTED_STEP_TREAD, EXACT_TOL, f"step {k} tread depth")
  for k in range(1, len(steps_sorted)):
    prev, cur = steps_sorted[k - 1], steps_sorted[k]
    prev_right = data.geom_xpos[prev][0] + model.geom_size[prev][0]
    cur_left = data.geom_xpos[cur][0] - model.geom_size[cur][0]
    c.close(
      cur_left,
      prev_right,
      EXACT_TOL,
      f"step {k} left edge is flush with step {k - 1} right edge (no gap/overlap)",
    )
  # The platform must start exactly at the last step's right edge.
  last = steps_sorted[-1]
  last_right = data.geom_xpos[last][0] + model.geom_size[last][0]
  platform_left = data.geom_xpos[platform_id][0] - model.geom_size[platform_id][0]
  c.close(
    platform_left,
    last_right,
    EXACT_TOL,
    "platform starts exactly at the last step's right edge",
  )
  c.close(
    2 * model.geom_size[platform_id][0],
    PLATFORM_DEPTH,
    EXACT_TOL,
    "platform depth",
  )
  c.close(
    data.geom_xpos[platform_id][2] + model.geom_size[platform_id][2],
    EXPECTED_TOTAL_RISE,
    EXACT_TOL,
    "platform top height",
  )

  # -------------------------------------------------------------------------
  print(f"\n[4] Downward-ray height profile (dx={RAY_DX} m, y={TERRAIN_SIZE[1] / 2}):")
  y_center = TERRAIN_SIZE[1] / 2
  n = int(round(TERRAIN_SIZE[0] / RAY_DX))
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
  expected_levels = {0.0} | set(expected_tops)
  actual_levels = sorted(set(rounded.tolist()))
  c.check(
    actual_levels == sorted(expected_levels),
    f"profile only contains expected heights {sorted(expected_levels)} "
    f"(got {actual_levels})",
  )
  positive_stair_levels = [h for h in actual_levels if h > 0.0]
  c.check(
    len(positive_stair_levels) == EXPECTED_NUM_STEPS,
    f"profile has exactly {EXPECTED_NUM_STEPS} distinct stair-top heights "
    f"(got {positive_stair_levels})",
  )

  # Segment the profile into plateaus and measure their widths.
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
  for level in expected_tops[:-1]:
    c.check(
      abs(width_by_level[round(level, 9)] - EXPECTED_STEP_TREAD) <= RAY_TOL,
      f"tread at z={level:.3f} is {EXPECTED_STEP_TREAD} m along +x "
      f"(measured {width_by_level[round(level, 9)]:.6f})",
    )
  # The top step and the platform share z=0.52; their combined run is exactly
  # one tread plus the platform depth.
  top_level = round(expected_tops[-1], 9)
  c.check(
    abs(width_by_level[top_level] - (EXPECTED_STEP_TREAD + PLATFORM_DEPTH)) <= RAY_TOL,
    f"top level z={top_level:.3f} run is tread+platform = "
    f"{EXPECTED_STEP_TREAD + PLATFORM_DEPTH} m "
    f"(measured {width_by_level[top_level]:.6f})",
  )
  c.check(
    abs(width_by_level[0.0] - (APPROACH_DEPTH + TERRAIN_SIZE[0] - cfg.flight_length))
    <= RAY_TOL,
    f"approach run at z=0.000 is {APPROACH_DEPTH} m "
    f"(measured {width_by_level[0.0]:.6f})",
  )

  # -------------------------------------------------------------------------
  print("\n[5] Spawn origin:")
  origin = np.asarray(output.origin)
  stairs_x0 = TERRAIN_SIZE[0] - cfg.flight_length + APPROACH_DEPTH
  print(f"  origin={origin}")
  c.close(origin[1], y_center, EXACT_TOL, "origin y is the patch centre")
  c.close(origin[2], 0.0, EXACT_TOL, "origin z is ground level")
  c.close(
    origin[0],
    stairs_x0 - SPAWN_OFFSET,
    EXACT_TOL,
    "origin x is spawn_offset in front of the first riser",
  )
  c.check(origin[0] < stairs_x0, "origin is on the approach, not on the stairs")
  spawn_z, spawn_gid = _ray_height(model, data, origin[0], origin[1])
  c.close(spawn_z, 0.0, EXACT_TOL, "ground under the spawn origin is flat at z=0")
  c.check(spawn_gid == approach_id, "spawn origin ray hits the approach geom")

  # -------------------------------------------------------------------------
  print("\n[6] Optional flat-patch sampling on the approach:")
  patch_cfg = BoxStraightStairsTerrainCfg(
    size=TERRAIN_SIZE,
    approach_depth=APPROACH_DEPTH,
    platform_depth=PLATFORM_DEPTH,
    spawn_offset=SPAWN_OFFSET,
    flat_patch_sampling={
      "spawn": FlatPatchSamplingCfg(num_patches=6, patch_radius=0.15)
    },
  )
  _, _, patch_output = _build(patch_cfg)
  patches = patch_output.flat_patches
  c.check(patches is not None and "spawn" in patches, "flat patches are emitted")
  if patches is not None:
    pts = patches["spawn"]
    print(f"  {len(pts)} patches, x in [{pts[:, 0].min():.3f}, {pts[:, 0].max():.3f}]")
    c.check(len(pts) == 6, "requested number of patches emitted")
    c.check(
      bool(np.all(pts[:, 2] == 0.0)),
      "all flat patches lie at z=0 on the approach",
    )
    c.check(
      bool(np.all(pts[:, 0] <= stairs_x0 - 0.15 + EXACT_TOL)),
      "all flat patches are at least patch_radius in front of the first riser",
    )

  # -------------------------------------------------------------------------
  print("\n[7] Strict mandate guard:")
  try:
    BoxStraightStairsTerrainCfg(size=TERRAIN_SIZE, step_height=0.10)
    c.check(False, "non-mandated step_height=0.10 is rejected when strict=True")
  except ValueError:
    c.check(True, "non-mandated step_height=0.10 is rejected when strict=True")

  # -------------------------------------------------------------------------
  print("\n" + "=" * 78)
  if c.failures:
    print(f"RESULT: FAIL ({len(c.failures)} check(s) failed)")
    for f in c.failures:
      print(f"  - {f}")
    return 1
  print("RESULT: PASS - geometry is exactly 4 steps x 0.13 m rise x 0.30 m tread, "
        "total rise 0.52 m")
  return 0


if __name__ == "__main__":
  sys.exit(main())
