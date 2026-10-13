"""Climb-progress shaping reward.

The velocity-tracking reward only sees the HORIZONTAL speed, so a robot that
walks flat around the staircase scores full marks: walking 0.4 m/s on the level
is a local optimum and the staircase is never climbed (v15 walks 9.2 m in 30 s
but its base height never leaves the approach).  This term rewards the base
height ABOVE the standing height, which is only achievable by actually climbing
the steps, giving the policy a direct gradient up the staircase.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def climb_progress(
  env,
  base_height: float = 0.33,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the base being above the standing height.

  The robot spawns at z = 0.33 (``base_height``).  Every step rises 0.13 m, so
  on the top platform the base is at 0.33 + 4 * 0.13 = 0.85 m.  Returning
  ``clamp(z - base_height, 0)`` yields 0 on the approach and grows linearly with
  each step climbed; the weight is applied by the RewardTermCfg.
  """
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.root_link_pos_w[:, 2]
  return torch.clamp(z - base_height, min=0.0)


def base_height_penalty(
  env,
  base_height: float = 0.33,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise the base dropping below the standing height (ground-hugging).

  ``climb_progress`` only rewards height ABOVE the stance, so a robot that
  crouches below it is neither rewarded nor punished -- and since crawling flat
  still collects the full velocity reward, the policy had no reason to stop
  doing it.  Visual inspection of the v15 rollout showed the robot dragging
  almost on the ground, which also makes it unable to lift a leg over a 13 cm
  step.  This returns ``clamp(base_height - z, 0)``, i.e. the depth below the
  stance, to be scaled by a NEGATIVE weight by the RewardTermCfg.
  """
  asset: Entity = env.scene[asset_cfg.name]
  z = asset.data.root_link_pos_w[:, 2]
  return torch.clamp(base_height - z, min=0.0)


__all__ = ["climb_progress", "base_height_penalty"]
