"""Fixed-geometry straight staircase terrain for the stairs assessment.

Geometry (every dimension is a dataclass field; the defaults are the values
mandated by the assessment):

* a flat approach area on the ground in front of the stairs,
* exactly ``num_steps`` (default 4) continuous steps ascending along ``+x``,
* each step has a constant ``step_height`` (default 0.13 m) rise and a constant
  ``step_tread`` (default 0.30 m) tread measured along the direction of travel,
* a flat top platform of ``platform_depth`` (default 1.7 m) after the last step.

Total rise is ``num_steps * step_height`` (4 * 0.13 = 0.52 m by default).

The flight is built from :data:`mujoco.mjtGeom.mjGEOM_BOX` geoms added to
``spec.body("terrain")``, mirroring ``mjlab.terrains.primitive_terrains``.
Because each step is a single solid axis-aligned box whose top face is exactly
at ``(k + 1) * step_height`` and whose ``+x`` extent is exactly ``step_tread``,
there are no hidden ramps, no reduced heights and no missing collisions (geoms
keep MuJoCo's default ``contype``/``conaffinity``, so they all participate in
contact).

Spawn semantics
---------------

:class:`~mjlab.terrains.terrain_generator.TerrainGenerator` interprets
``TerrainOutput.origin`` as a position in the sub-terrain's *local* frame
(origin at the patch's minimum corner); it adds the patch's world position and
stores the result in ``terrain_origins``, which is what
:class:`~mjlab.terrains.terrain_entity.TerrainEntity` uses for
``env_origins``. The origin returned here therefore lies on the flat approach,
``spawn_offset`` metres **in front of the first riser**, centred in ``y`` and at
``z = 0`` (ground level). Unitree A2's front hips are at ``+x`` and its default
root orientation is the identity quaternion, so an environment reset with zero
yaw spawns facing the stairs.

``TerrainOutput.flat_patches`` is filled only when ``flat_patch_sampling`` is
configured (see :meth:`BoxStraightStairsTerrainCfg._make_flat_patches`). The
default spawn path (``terrain_origins`` + ``reset_root_state_uniform``) does
not need flat patches; they exist so ``reset_root_state_from_flat_patches``
also works.

The flight is intentionally independent of ``difficulty``: the assessment
mandates fixed dimensions, so every curriculum row produces identical steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.utils.color import brand_ramp, darken_rgba

__all__ = ["BoxStraightStairsTerrainCfg"]

# Mandated geometry (assessment). Kept as module constants so the verification
# script and downstream users can reference the exact expected numbers.
MANDATED_NUM_STEPS = 4
MANDATED_STEP_HEIGHT = 0.13
MANDATED_STEP_TREAD = 0.30
MANDATED_TOTAL_RISE = MANDATED_NUM_STEPS * MANDATED_STEP_HEIGHT

# Palette (mirrors mjlab.terrains.primitive_terrains).
_MUJOCO_BLUE = (0.20, 0.45, 0.95)
_MUJOCO_GREEN = (0.25, 0.80, 0.45)


@dataclass(kw_only=True)
class BoxStraightStairsTerrainCfg(SubTerrainCfg):
  """A single straight flight of fixed-dimension stairs.

  Unlike :class:`~mjlab.terrains.BoxPyramidStairsTerrainCfg` (stairs rising in
  both ``x`` and ``y``, with the rise interpolated by ``difficulty``), this is
  one straight flight with constant rise and constant tread. ``difficulty`` is
  ignored on purpose: the geometry is fixed.
  """

  num_steps: int = MANDATED_NUM_STEPS
  """Number of continuous steps. Mandated value: 4."""
  step_height: float = MANDATED_STEP_HEIGHT
  """Rise of each step, in meters. Mandated value: 0.13."""
  step_tread: float = MANDATED_STEP_TREAD
  """Tread depth of each step measured along the travel direction (+x), in
  meters. Mandated value: 0.30."""
  approach_depth: float = 1.6
  """Depth of the flat approach area in front of the first riser, in meters."""
  platform_depth: float = 1.7
  """Depth of the flat top platform after the last step, in meters."""
  base_thickness: float = 0.5
  """Thickness of the solid base below ground level (z = 0), in meters."""
  spawn_offset: float = 0.9
  """Distance from the spawn origin to the first riser, in meters. The origin
  is placed on the approach at ``stairs_x0 - spawn_offset``."""
  stair_width: float | None = None
  """Width of the steps/platform along ``y``, in meters. ``None`` (default)
  uses the full sub-terrain width, so there are no gaps beside the stairs."""
  curriculum_steps: bool = False
  """If True, ``difficulty`` ramps the NUMBER of steps from 1 to ``num_steps``.

  Every step keeps the mandated 0.13 m rise and 0.30 m tread, so the terrain
  curriculum never weakens the required geometry: it only starts the robot on a
  shorter flight.  At ``difficulty == 1`` the terrain is exactly the mandated
  4 x 0.13 m x 0.30 m staircase.  Leave False (default) for ``play``/evaluation,
  where the full flight is always built."""
  strict: bool = True
  """If True (default), assert that ``num_steps``/``step_height``/``step_tread``
  equal the mandated values and that each emitted box encodes them exactly. Set
  to False to reuse the class with other step dimensions."""

  def __post_init__(self) -> None:
    if self.strict:
      if self.num_steps != MANDATED_NUM_STEPS:
        raise ValueError(
          f"Mandated num_steps is {MANDATED_NUM_STEPS}, got {self.num_steps}. "
          "Set strict=False to use other step counts."
        )
      if self.step_height != MANDATED_STEP_HEIGHT:
        raise ValueError(
          f"Mandated step_height is {MANDATED_STEP_HEIGHT}, got "
          f"{self.step_height}. Set strict=False to use other heights."
        )
      if self.step_tread != MANDATED_STEP_TREAD:
        raise ValueError(
          f"Mandated step_tread is {MANDATED_STEP_TREAD}, got "
          f"{self.step_tread}. Set strict=False to use other treads."
        )
    if self.num_steps < 1:
      raise ValueError(f"num_steps must be >= 1, got {self.num_steps}.")
    if self.step_height <= 0.0:
      raise ValueError(f"step_height must be > 0, got {self.step_height}.")
    if self.step_tread <= 0.0:
      raise ValueError(f"step_tread must be > 0, got {self.step_tread}.")
    if self.approach_depth <= 0.0:
      raise ValueError(f"approach_depth must be > 0, got {self.approach_depth}.")
    if self.platform_depth <= 0.0:
      raise ValueError(f"platform_depth must be > 0, got {self.platform_depth}.")
    if self.base_thickness <= 0.0:
      raise ValueError(f"base_thickness must be > 0, got {self.base_thickness}.")
    if self.spawn_offset <= 0.0:
      raise ValueError(f"spawn_offset must be > 0, got {self.spawn_offset}.")
    if self.stair_width is not None and self.stair_width <= 0.0:
      raise ValueError(f"stair_width must be > 0, got {self.stair_width}.")

  @property
  def total_rise(self) -> float:
    """Height of the top platform above the approach, in meters."""
    return self.num_steps * self.step_height

  @property
  def flight_length(self) -> float:
    """Total length along ``x`` of approach + steps + platform, in meters."""
    return self.approach_depth + self.num_steps * self.step_tread + self.platform_depth

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    # Difficulty ramps the NUMBER of steps (1 -> num_steps) when
    # `curriculum_steps` is set.  Each step always keeps the mandated 0.13 m rise
    # and 0.30 m tread, so the curriculum never weakens the required geometry.
    if self.curriculum_steps:
      frac = min(max(float(difficulty), 0.0), 1.0)
      n_steps = int(round(1 + frac * (self.num_steps - 1)))
      n_steps = max(1, min(self.num_steps, n_steps))
    else:
      n_steps = self.num_steps
    del rng  # Deterministic layout.

    body = spec.body("terrain")

    total_rise = n_steps * self.step_height
    flight_length = (
      self.approach_depth + n_steps * self.step_tread + self.platform_depth
    )
    if flight_length > self.size[0] + 1e-9:
      raise ValueError(
        f"Terrain length {self.size[0]:.3f} m is too short for the requested "
        f"staircase ({flight_length:.3f} m = approach {self.approach_depth} + "
        f"{n_steps} * {self.step_tread} tread + platform "
        f"{self.platform_depth}). Increase size[0] or reduce the depths."
      )

    # Pad the *front* of the approach so the platform always ends flush with the
    # far edge of the patch (never a cliff behind the platform).
    front_pad = self.size[0] - flight_length
    stairs_x0 = front_pad + self.approach_depth
    platform_x0 = stairs_x0 + n_steps * self.step_tread

    width = self.size[1] if self.stair_width is None else self.stair_width
    if width > self.size[1] + 1e-9:
      raise ValueError(
        f"stair_width {width:.3f} exceeds terrain width {self.size[1]:.3f}."
      )
    y_center = 0.5 * self.size[1]
    half_base = 0.5 * self.base_thickness

    geometries: list[TerrainGeometry] = []

    # Flat approach, spanning the front of the sub-terrain at ground level.
    approach = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.5 * stairs_x0, 0.5 * self.size[1], half_base),
      pos=(0.5 * stairs_x0, y_center, -half_base),
    )
    approach_rgba = darken_rgba(brand_ramp(_MUJOCO_BLUE, 0.0), 0.85)
    geometries.append(TerrainGeometry(geom=approach, color=approach_rgba))

    # Continuous steps. Step k (0-indexed) has its top at (k + 1) * step_height
    # and its exposed tread spans exactly one step_tread along +x.
    step_geoms: list[mujoco.MjsGeom] = []
    for k in range(n_steps):
      top_z = (k + 1) * self.step_height
      box_height = top_z + self.base_thickness
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.5 * self.step_tread, 0.5 * width, 0.5 * box_height),
        pos=(
          stairs_x0 + (k + 0.5) * self.step_tread,
          y_center,
          top_z - 0.5 * box_height,
        ),
      )
      t = k / max(n_steps - 1, 1)
      geometries.append(TerrainGeometry(geom=box, color=brand_ramp(_MUJOCO_BLUE, t)))
      step_geoms.append(box)

    # Flat top platform, flush with the far edge of the patch.
    platform_length = self.size[0] - platform_x0
    platform_height = total_rise + self.base_thickness
    platform = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.5 * platform_length, 0.5 * width, 0.5 * platform_height),
      pos=(
        platform_x0 + 0.5 * platform_length,
        y_center,
        total_rise - 0.5 * platform_height,
      ),
    )
    geometries.append(
      TerrainGeometry(geom=platform, color=brand_ramp(_MUJOCO_GREEN, 0.5))
    )

    self._assert_emitted_geometry(step_geoms, platform, n_steps)

    # Spawn on the approach, facing +x (towards the stairs).
    origin_x = stairs_x0 - self.spawn_offset
    if origin_x < 0.0:
      raise ValueError(
        f"spawn_offset {self.spawn_offset:.3f} m does not fit on the approach "
        f"(approach spans [0, {stairs_x0:.3f}])."
      )
    origin = np.array([origin_x, y_center, 0.0])

    return TerrainOutput(
      origin=origin,
      geometries=geometries,
      flat_patches=self._make_flat_patches(stairs_x0, origin),
    )

  def _assert_emitted_geometry(
    self, step_geoms: list[mujoco.MjsGeom], platform: mujoco.MjsGeom, n_steps: int
  ) -> None:
    """Assert the emitted boxes encode the configured dimensions exactly."""
    assert len(step_geoms) == n_steps
    for k, box in enumerate(step_geoms):
      assert abs(2.0 * box.size[0] - self.step_tread) < 1e-12, (
        f"step {k} tread is {2.0 * box.size[0]}, expected {self.step_tread}."
      )
      top_z = box.pos[2] + box.size[2]
      assert abs(top_z - (k + 1) * self.step_height) < 1e-12, (
        f"step {k} top is {top_z}, expected {(k + 1) * self.step_height}."
      )
    platform_top = platform.pos[2] + platform.size[2]
    assert abs(platform_top - n_steps * self.step_height) < 1e-12, (
      f"platform top is {platform_top}, expected {self.total_rise}."
    )

  def _make_flat_patches(
    self, approach_x1: float, origin: np.ndarray
  ) -> dict[str, np.ndarray] | None:
    """Sample flat patches on the approach when ``flat_patch_sampling`` is set.

    ``TerrainOutput.flat_patches`` is optional: the framework's normal spawn
    path uses ``terrain_origins``, and the generator back-fills missing patches
    with ``origin``. When the user configures
    ``flat_patch_sampling={"spawn": FlatPatchSamplingCfg(...)}`` we instead emit
    a deterministic grid of patches that is guaranteed to lie on the flat
    approach, at least ``patch_radius`` away from the first riser.
    """
    if self.flat_patch_sampling is None:
      return None
    cfg: FlatPatchSamplingCfg | None = self.flat_patch_sampling.get("spawn")
    if cfg is None:
      return None

    x_lo = max(cfg.x_range[0], cfg.patch_radius)
    x_hi = min(cfg.x_range[1], approach_x1 - cfg.patch_radius)
    y_lo = max(cfg.y_range[0], cfg.patch_radius)
    y_hi = min(cfg.y_range[1], self.size[1] - cfg.patch_radius)
    if x_hi < x_lo or y_hi < y_lo:
      # Approach too small for the requested footprint: fall back to the spawn.
      return {"spawn": np.tile(origin, (cfg.num_patches, 1))}

    side = int(np.ceil(np.sqrt(cfg.num_patches)))
    xs = np.linspace(x_lo, x_hi, side)
    ys = np.linspace(y_lo, y_hi, side)
    grid = np.array([[x, y, 0.0] for x in xs for y in ys])
    return {"spawn": grid[: cfg.num_patches]}
