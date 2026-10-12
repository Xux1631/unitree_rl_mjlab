from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  quadruped_3320_flat_env_cfg,
  quadruped_3320_stairs_env_cfg,
)
from .rl_cfg import quadruped_3320_ppo_runner_cfg

# Training task: terrain curriculum over the mandated four-step staircase.
register_mjlab_task(
  task_id="Quadruped-3320-Stairs",
  env_cfg=quadruped_3320_stairs_env_cfg(),
  play_env_cfg=quadruped_3320_stairs_env_cfg(play=True, command_speed=0.4),
  rl_cfg=quadruped_3320_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Flat-terrain task: used for the "平地连续稳定行走 30 s" requirement and as a
# curriculum stage baseline.
register_mjlab_task(
  task_id="Quadruped-3320-Flat",
  env_cfg=quadruped_3320_flat_env_cfg(),
  play_env_cfg=quadruped_3320_flat_env_cfg(play=True, command_speed=0.4),
  rl_cfg=quadruped_3320_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Play-only tiers for the assessment's performance table.  Identical terrain,
# collision and success criteria -- only the constant command speed differs, so
# the tiered results stay comparable.
for _tag, _speed in (("0.6", 0.6), ("0.8", 0.8)):
  register_mjlab_task(
    task_id=f"Quadruped-3320-Stairs-{_tag}",
    env_cfg=quadruped_3320_stairs_env_cfg(),
    play_env_cfg=quadruped_3320_stairs_env_cfg(play=True, command_speed=_speed),
    rl_cfg=quadruped_3320_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
