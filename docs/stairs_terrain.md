# Straight staircase terrain (`BoxStraightStairsTerrainCfg`)

Fixed-geometry staircase for the A2 stairs assessment. It replaces
`BoxPyramidStairsTerrainCfg`, whose rise is interpolated by `difficulty` and
whose stairs rise in both `x` and `y`, with a single straight flight that has
constant, exact dimensions.

Module: `src/tasks/velocity/mdp/stairs_terrain.py`
Verification: `scripts/check_stairs_terrain.py`

## Geometry

Travel direction is **`+x`**. The robot spawns with default (identity)
orientation, which is `+x` forward for the Unitree A2 (front hips at `+x`), and
therefore faces the stairs.

```
 z
 ^                    +--------+  top platform (z = 0.52)
 |                    |        |
0.52             +----+        |
 |               | 4  |        |
0.39        +----+    |        |
 |          | 3  |    |        |
0.26     +--+    |    |        |
 |       | 2|    |    |        |
0.13  +--+  |    |    |        |
 |    | 1|  |    |    |        |
0.00 -+--+--+----+----+--------+------------------> x
      |<--->|<--->|<--->|
       0.30  0.30  0.30
      |<----- approach ----->|<-- steps -->|<-- platform -->|
             1.60 m           4 x 0.30 m        1.70 m
```

| Quantity | Value | Field |
| --- | --- | --- |
| Number of continuous steps | **4** | `num_steps` |
| Step rise | **0.13 m** | `step_height` |
| Step tread along travel (`+x`) | **0.30 m** | `step_tread` |
| Total rise (`4 x 0.13`) | **0.52 m** | `num_steps * step_height` |
| Flat approach in front of first riser | 1.6 m | `approach_depth` |
| Flat top platform after last step | 1.7 m | `platform_depth` |
| Solid base below `z = 0` | 0.5 m | `base_thickness` |
| Spawn distance in front of first riser | 0.9 m | `spawn_offset` |
| Stair width along `y` | full patch width | `stair_width` (`None` = `size[1]`) |

With the default A2 patch size `(4.5, 4.5)` the flight exactly fills the patch:
`1.6 + 4*0.30 + 1.7 = 4.5 m`. For longer patches the extra length is added to
the front of the approach (`front_pad`) so the platform always stays flush with
the far edge and there is never a cliff behind it.

Every dimension is a dataclass field. `strict=True` (default) validates that
`num_steps`, `step_height` and `step_tread` equal the mandated values; set
`strict=False` to reuse the class with other dimensions.

## Collision parameters

- Seven `mujoco.mjtGeom.mjGEOM_BOX` geoms are added to `spec.body("terrain")`:
  one approach slab, four steps, one top platform (order: approach, step 0..3,
  platform).
- All geoms keep MuJoCo's defaults `contype = 1`, `conaffinity = 1`, so every
  step participates in contact. Nothing is visual-only.
- Each step is a *solid* box from `z = -base_thickness` up to its top surface,
  so there are no cavities and no hidden ramps. `+x` extents are exactly
  `step_tread` and are flush with each other; the platform starts exactly at the
  last step's right edge.
- Geoms are unnamed, matching `mjlab.terrains.primitive_terrains`;
  `TerrainGenerator.compile()` renames them to `terrain_0..N` after generation.

## Spawn origin and facing

`TerrainGenerator._create_terrain_geom` treats `TerrainOutput.origin` as a
position in the sub-terrain's local frame (origin at the patch's minimum corner)
and adds the patch's world position, storing the result in `terrain_origins`
(used by `TerrainEntity` for `env_origins`). This class returns

```
origin = (front_pad + approach_depth - spawn_offset, size[1] / 2, 0.0)
```

i.e. on the flat approach, `spawn_offset` metres in front of the first riser,
centred in `y`, at ground level. It is never on the stairs.

`origin` carries no yaw, so the facing direction comes from the reset pose. The
default root orientation of the A2 is the identity quaternion (`+x` forward), so
a zero-yaw reset faces the stairs. The stock velocity reset samples
`yaw: (-3.14, 3.14)`; override it to `(0.0, 0.0)` to guarantee facing (see the
wiring snippet in the task report / below).

`flat_patch_sampling` is **not required** by the spawn path: `randomize_terrain`
and terrain-level assignment use `terrain_origins` (and the generator back-fills
missing flat patches with the spawn origin). Configuring
`flat_patch_sampling={"spawn": FlatPatchSamplingCfg(...)}` additionally emits a
deterministic grid of patches on the approach, at least `patch_radius` away from
the first riser, for use with `reset_root_state_from_flat_patches`.

## Wiring into `unitree_a2_rough_env_cfg`

Replace the current `stairs` block in
`src/tasks/velocity/config/a2/env_cfgs.py` with:

```python
from src.tasks.velocity.mdp.stairs_terrain import BoxStraightStairsTerrainCfg

# ...

cfg.scene.terrain.terrain_generator.curriculum = True
cfg.scene.terrain.terrain_generator.size = (4.5, 4.5)

# Option A (recommended for the assessment): only the mandated staircase, so
# every column / every environment is the straight stairs.
cfg.scene.terrain.terrain_generator.sub_terrains = {
  "straight_stairs": BoxStraightStairsTerrainCfg(
    proportion=1.0,
    approach_depth=1.6,
    platform_depth=1.7,
    spawn_offset=0.9,
  ),
}

# Face the stairs on reset (default root orientation is +x = up the stairs).
# The stock x range (-0.5, 0.5) already keeps the robot on the 1.6 m approach.
cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
```

With `curriculum=True`, mjlab normalises `proportion` over the sub-terrains and
assigns each column by the cumulative distribution
(`TerrainGenerator._generate_curriculum_terrains`); the first key owns the first
columns. A single sub-terrain therefore owns all 20 columns, and every
environment spawns on the approach. If you instead keep the stock rough
sub-terrains and append `straight_stairs` last, its normalised share
`p / (1.0 + p)` gives it the *last* columns: `p=1.0` -> 10/20 columns,
`p=0.8` -> 8/20. Those environments spawn on other terrain types too, so the
zero-yaw override would not be appropriate for all of them.

## Reproduce

```bash
cd /home/xux/RC/mjlab/unitree_rl_mjlab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab
export MPLCONFIGDIR=/tmp/mpl
python scripts/check_stairs_terrain.py     # exits non-zero on failure
```

The script builds the config into a fresh `mujoco.MjSpec`, compiles it, dumps
every box geom, and verifies from the *compiled model* (not from the config)
that:

1. every terrain geom is an axis-aligned box with `contype`/`conaffinity != 0`;
2. exactly 4 distinct step top heights exist, each separated by 0.13 m, with a
   total rise of 0.52 m;
3. each step's tread along `+x` is exactly 0.30 m and adjacent steps are flush;
4. a downward-ray height profile (`dx = 1e-4 m`) along `y = size[1]/2` is
   piecewise constant with plateaus `0.00 -> 0.13 -> 0.26 -> 0.39 -> 0.52 m`
   (no hidden ramps or intermediate heights);
5. the spawn origin lands on the flat approach (ray hits `z = 0`), and optional
   flat patches also land on the approach;
6. `strict=True` rejects non-mandated step dimensions.
