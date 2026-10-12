"""Gait-reference reward used to bootstrap locomotion.

Attempts 1-6 of this task all converged to standing still: PPO optimised the
velocity-tracking reward to roughly the "do nothing" ceiling and never
discovered a gait, even though an open-loop trot drives this exact model
forward at ~0.32 m/s while staying upright (verified in a standalone MuJoCo
rollout, 12/12 parameter sets).

This term supplies that missing gradient by rewarding the joints for following a
known-good trot, as a function of the gait phase.  It is a shaping term only:
the policy is still plain PPO on the task reward, the reference is our own
open-loop gait (not a reference package or pretrained policy), and the weight is
annealed to zero by a curriculum stage so the final policy is free to improve on
it.
"""

from __future__ import annotations

import math

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Diagonal pairs move together, matching the open-loop trot that was measured to
# walk the model: RL and FR in phase, FL and RR half a period later.
TROT_PHASE = {"RL": 0.0, "FR": 0.0, "FL": 0.5, "RR": 0.5}


def trot_gait_reference(
  env,
  period: float = 0.6,
  amplitude: float = 0.30,
  std: float = 0.35,
  command_name: str | None = "twist",
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the joints for tracking a diagonal trot around the default stance.

  swing amplitude ``amplitude`` (rad) is applied to the thigh (both directions)
  and to the calf (swing half only, so the knee never hyper-extends under the
  body).  Returns ``exp(-mean_j (q - q_ref)^2 / std^2)`` per environment, gated
  by the commanded speed so an idle command does not force a gait.
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos  # [B, njoints] in the policy/MJCF joint order
  q_default = asset.data.default_joint_pos  # [B, njoints]

  # Gait phase advances with simulated time; env.step_dt is the control period.
  t = env.episode_length_buf.to(torch.float32) * env.step_dt
  phase = (t / period) % 1.0  # [B]

  names = asset.joint_names
  q_ref = q_default.clone()
  batch = q.shape[0]
  for j, name in enumerate(names):
    leg = name.split("_")[0]
    family = name.split("_")[1]
    off = TROT_PHASE.get(leg)
    if off is None:
      continue
    s = ((phase + off) % 1.0) * 2.0 * math.pi
    if family == "thigh":
      q_ref[:, j] = q_default[:, j] + amplitude * torch.sin(s)
    elif family == "calf":
      # swing half only: lift the knee while the foot is in the air
      q_ref[:, j] = q_default[:, j] - amplitude * torch.clamp(torch.sin(s), min=0.0)

  err = torch.mean(torch.square(q - q_ref), dim=1)
  reward = torch.exp(-err / (std * std))
  del batch

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      reward = reward * (speed > command_threshold).float()
  return reward


__all__ = ["trot_gait_reference", "TROT_PHASE"]
