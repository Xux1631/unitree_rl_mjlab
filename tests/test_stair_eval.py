"""Unit tests for the pure staircase success/failure logic.

CPU-only: ``src.tasks.velocity.mdp.stair_eval`` deliberately imports nothing
from mjlab / mujoco / torch, so these tests run on any machine.

Run from the repository root::

  python -m pytest tests/test_stair_eval.py -q
"""

from __future__ import annotations

import math

import pytest

from src.tasks.velocity.mdp.stair_eval import (
  FAIL_BODY_CONTACT,
  FAIL_OK,
  FAIL_PITCH,
  FAIL_RESET,
  FAIL_ROLL,
  FAIL_STUCK,
  FAIL_TIMEOUT,
  MANDATED_TOTAL_RISE,
  CourseGeometry,
  SuccessCriteria,
  TrialEvaluator,
  TrialSample,
  assert_mandated_geometry,
  evaluate_trace,
  quat_wxyz_to_roll_pitch,
)

DT = 0.02  # 50 Hz control loop (mjlab: timestep 0.005 * decimation 4).

# Base-link heights used by the synthetic traces (metres).
APPROACH_Z = 0.33
PLATFORM_Z = APPROACH_Z + MANDATED_TOTAL_RISE
RISER_X = CourseGeometry().first_riser_x  # 0.9 m in front of the origin
PLATFORM_X = CourseGeometry().platform_start_x  # 2.1 m


def criteria(**overrides) -> SuccessCriteria:
  return SuccessCriteria(**overrides)


def sample(
  t: float,
  x: float,
  z: float = APPROACH_Z,
  roll: float = 0.0,
  pitch: float = 0.0,
  body_contact: bool = False,
  env_done: bool = False,
  env_reason: str | None = None,
) -> TrialSample:
  return TrialSample(
    t=t,
    x=x,
    z=z,
    roll=roll,
    pitch=pitch,
    body_contact=body_contact,
    env_done=env_done,
    env_reason=env_reason,
  )


def stairs_z(x: float) -> float:
  """Base height for a robot that climbs the flight as ``x`` advances.

  Linear interpolation over the flight is fine for the tests: it only needs to
  cross the platform-height threshold once the base is past the last riser.
  """
  u = (x - RISER_X) / (PLATFORM_X - RISER_X)
  u = max(0.0, min(1.0, u))
  return APPROACH_Z + MANDATED_TOTAL_RISE * u


def trace_forward(
  x0: float,
  x1: float,
  v: float,
  **kwargs,
) -> list[TrialSample]:
  """Uniform forward walk from ``x0`` to ``x1`` at ``v`` m/s.

  Positions are snapped to a 1e-9 grid and to the exact endpoints so the
  boundary tests are not perturbed by float accumulation.
  """
  dx = v * DT
  n = max(1, int(round((x1 - x0) / dx)))
  out: list[TrialSample] = []
  for k in range(n + 1):
    x = x1 if k == n else round(x0 + k * dx, 9)
    out.append(sample(k * DT, x, stairs_z(x), **kwargs))
  return out


def trace_hold_last(
  samples: list[TrialSample],
  extra_s: float,
  **kwargs,
) -> list[TrialSample]:
  """Append ``extra_s`` seconds of samples frozen at the last state."""
  last = samples[-1]
  out = list(samples)
  n = int(round(extra_s / DT))
  for k in range(1, n + 1):
    out.append(
      sample(
        last.t + k * DT,
        last.x,
        last.z,
        roll=last.roll,
        pitch=last.pitch,
        **kwargs,
      )
    )
  return out


# ---------------------------------------------------------------------------
# Mandated geometry guard
# ---------------------------------------------------------------------------


def test_mandated_geometry_accepts_the_assessment_values() -> None:
  geom = CourseGeometry()
  assert_mandated_geometry(geom)
  assert geom.total_rise == pytest.approx(0.52)
  assert geom.platform_start_x == pytest.approx(2.1)
  assert geom.platform_end_x == pytest.approx(3.8)


@pytest.mark.parametrize(
  "overrides",
  [
    {"num_steps": 3},
    {"step_height": 0.10},
    {"step_tread": 0.25},
    {"num_steps": 5, "step_height": 0.10},  # same 0.50 rise is still rejected
  ],
)
def test_mandated_geometry_rejects_deviations(overrides: dict) -> None:
  with pytest.raises(ValueError):
    assert_mandated_geometry(CourseGeometry(**overrides))


# ---------------------------------------------------------------------------
# Roll / pitch extraction
# ---------------------------------------------------------------------------


def test_quat_roll_pitch_roundtrip() -> None:
  roll_in, pitch_in = math.radians(20.0), math.radians(-15.0)
  # Half-angle quaternion for Rz(0) * Ry(pitch) * Rx(roll).
  cr, sr = math.cos(roll_in / 2), math.sin(roll_in / 2)
  cp, sp = math.cos(pitch_in / 2), math.sin(pitch_in / 2)
  qw = cr * cp
  qx = sr * cp
  qy = cr * sp
  qz = -sr * sp
  roll, pitch = quat_wxyz_to_roll_pitch((qw, qx, qy, qz))
  assert roll == pytest.approx(roll_in, abs=1e-9)
  assert pitch == pytest.approx(pitch_in, abs=1e-9)


# ---------------------------------------------------------------------------
# 1. Clean success
# ---------------------------------------------------------------------------


def test_clean_success_run() -> None:
  # Walk the whole course at 0.5 m/s and stop 1.0 m after the platform edge.
  samples = trace_forward(0.0, PLATFORM_X + 1.2, v=0.5)
  result = evaluate_trace(samples, criteria())

  assert result.success is True
  assert result.failure_reason == FAIL_OK
  assert result.platform_entry_x == pytest.approx(PLATFORM_X, abs=1e-6)
  assert result.post_platform_travel >= 1.0
  assert result.elapsed_time > 0.0
  assert result.max_abs_roll == 0.0
  assert result.max_abs_pitch == 0.0


# ---------------------------------------------------------------------------
# 2. Body contact
# ---------------------------------------------------------------------------


def test_body_contact_fails() -> None:
  samples = trace_forward(0.0, PLATFORM_X + 1.2, v=0.5)
  # A knee/body drags on a step from t = 1.0 s onwards for 1 s.
  samples = [
    sample(
      s.t,
      s.x,
      s.z,
      roll=s.roll,
      pitch=s.pitch,
      body_contact=(1.0 <= s.t < 2.0),
    )
    for s in samples
  ]
  result = evaluate_trace(samples, criteria())
  assert result.success is False
  assert result.failure_reason == FAIL_BODY_CONTACT
  # Failure must fire shortly after the 0.2 s grace period elapses.
  assert result.elapsed_time == pytest.approx(1.2, abs=0.05)


def test_instant_body_contact_termination_fails() -> None:
  """The environment's ``illegal_contact`` termination maps to body_contact."""
  samples = trace_forward(0.0, 1.0, v=0.4)
  samples.append(
    sample(samples[-1].t + DT, 1.0, APPROACH_Z, env_done=True, env_reason="body_contact")
  )
  result = evaluate_trace(samples, criteria())
  assert result.success is False
  assert result.failure_reason == FAIL_BODY_CONTACT


def test_brief_body_contact_does_not_fail() -> None:
  """A single-step graze shorter than the grace period must not fail."""
  samples = trace_forward(0.0, PLATFORM_X + 1.2, v=0.5)
  samples = [
    TrialSample(
      t=s.t,
      x=s.x,
      z=s.z,
      roll=s.roll,
      pitch=s.pitch,
      body_contact=abs(s.t - 1.0) < DT / 2,
    )
    for s in samples
  ]
  result = evaluate_trace(samples, criteria())
  assert result.success is True


# ---------------------------------------------------------------------------
# 3. Orientation limit: sustained roll / pitch fails, short spike does not
# ---------------------------------------------------------------------------


def test_sustained_roll_over_45_deg_fails() -> None:
  samples = trace_forward(0.0, PLATFORM_X + 1.2, v=0.5)
  # 50 deg roll held from t = 1.0 s to t = 2.5 s (> 0.5 s).
  samples = [
    TrialSample(
      t=s.t,
      x=s.x,
      z=s.z,
      roll=math.radians(50.0) if 1.0 <= s.t < 2.5 else s.roll,
      pitch=s.pitch,
    )
    for s in samples
  ]
  result = evaluate_trace(samples, criteria())
  assert result.success is False
  assert result.failure_reason == FAIL_ROLL
  assert result.max_abs_roll == pytest.approx(math.radians(50.0))
  # 0.5 s after the excursion begins.
  assert result.elapsed_time == pytest.approx(1.5, abs=0.05)


def test_sustained_pitch_over_45_deg_fails() -> None:
  samples = trace_forward(0.0, PLATFORM_X + 1.2, v=0.5)
  samples = [
    TrialSample(
      t=s.t,
      x=s.x,
      z=s.z,
      roll=s.roll,
      pitch=math.radians(-55.0) if 1.0 <= s.t < 2.0 else s.pitch,
    )
    for s in samples
  ]
  result = evaluate_trace(samples, criteria())
  assert result.success is False
  assert result.failure_reason == FAIL_PITCH
  assert result.max_abs_pitch == pytest.approx(math.radians(55.0))


def test_brief_roll_spike_does_not_fail() -> None:
  """A 0.3 s spike above 45 deg must NOT fail the trial."""
  samples = trace_forward(0.0, PLATFORM_X + 1.2, v=0.5)
  samples = [
    TrialSample(
      t=s.t,
      x=s.x,
      z=s.z,
      roll=math.radians(60.0) if 1.0 <= s.t < 1.3 else s.roll,
      pitch=s.pitch,
    )
    for s in samples
  ]
  result = evaluate_trace(samples, criteria())
  assert result.success is True
  assert result.max_abs_roll == pytest.approx(math.radians(60.0))
  assert result.failure_reason == FAIL_OK


# ---------------------------------------------------------------------------
# 4. Stuck / no progress
# ---------------------------------------------------------------------------


def test_stuck_before_the_stairs_fails() -> None:
  samples = trace_forward(0.0, 0.5, v=0.5)
  samples = trace_hold_last(samples, extra_s=10.0)
  result = evaluate_trace(samples, criteria())
  assert result.success is False
  assert result.failure_reason == FAIL_STUCK
  # Halted at t ~= 1.0 s, stuck timeout is 6 s by default.
  assert result.elapsed_time == pytest.approx(1.0 + 6.0, abs=0.2)


def test_stuck_timeout_is_configurable() -> None:
  samples = trace_forward(0.0, 0.5, v=0.5)
  samples = trace_hold_last(samples, extra_s=4.0)
  result = evaluate_trace(samples, criteria(stuck_timeout_s=2.0))
  assert result.failure_reason == FAIL_STUCK
  assert result.elapsed_time == pytest.approx(1.0 + 2.0, abs=0.2)


def test_slow_but_steady_progress_is_not_stuck_until_timeout() -> None:
  """Creeping forward below the stuck distance still refreshes the timer."""
  samples = []
  t = 0.0
  x = 0.0
  for _ in range(int(35.0 / DT)):
    samples.append(sample(t, x, stairs_z(x)))
    x += 0.06 * DT  # 0.06 m/s: slow, but > progress epsilon per second
    t += DT
  result = evaluate_trace(samples, criteria())
  assert result.success is False
  assert result.failure_reason == FAIL_TIMEOUT
  assert result.elapsed_time == pytest.approx(30.0, abs=0.05)


# ---------------------------------------------------------------------------
# 5. Reaching the platform is not enough: 1 m of post-platform travel required
# ---------------------------------------------------------------------------


def test_reaches_platform_but_stops_must_fail() -> None:
  # Climb to 0.4 m past the platform edge, then freeze.
  samples = trace_forward(0.0, PLATFORM_X + 0.4, v=0.5)
  samples = trace_hold_last(samples, extra_s=10.0)
  result = evaluate_trace(samples, criteria())
  assert result.success is False
  assert result.failure_reason == FAIL_STUCK
  # It really did reach the platform, it just did not keep walking.
  assert result.platform_entry_x is not None
  assert result.post_platform_travel < 1.0


def test_platform_reached_but_less_than_one_metre_of_travel_fails() -> None:
  # 0.95 m of travel after the platform edge is not enough.
  samples = trace_forward(0.0, PLATFORM_X + 0.95, v=0.5)
  samples = trace_hold_last(samples, extra_s=10.0)
  result = evaluate_trace(samples, criteria())
  assert result.success is False
  assert result.post_platform_travel < 1.0


def test_exactly_one_metre_of_post_platform_travel_succeeds() -> None:
  samples = trace_forward(0.0, PLATFORM_X + 1.0, v=0.5)
  samples = trace_hold_last(samples, extra_s=0.1)
  result = evaluate_trace(samples, criteria())
  assert result.success is True
  assert result.post_platform_travel == pytest.approx(1.0, abs=1e-6)


def test_platform_x_without_platform_height_is_not_reached() -> None:
  """Crossing the platform's x while still low does not count as reaching it."""
  # x walks past the platform edge but the base never rises (e.g. it is
  # standing on the approach in a different patch / below the flight).
  samples = trace_forward(0.0, PLATFORM_X + 1.5, v=0.5)
  samples = [TrialSample(s.t, s.x, APPROACH_Z, s.roll, s.pitch) for s in samples]
  result = evaluate_trace(samples, criteria())
  assert result.success is False


# ---------------------------------------------------------------------------
# 6. Episode reset / timeout
# ---------------------------------------------------------------------------


def test_env_reset_before_completion_fails_with_reset() -> None:
  samples = trace_forward(0.0, 1.5, v=0.5)
  samples.append(
    sample(samples[-1].t + DT, 1.5, stairs_z(1.5), env_done=True, env_reason=None)
  )
  result = evaluate_trace(samples, criteria())
  assert result.success is False
  assert result.failure_reason == FAIL_RESET


def test_episode_timeout_fails() -> None:
  """A clean, steadily advancing walk still fails if it runs out of time."""
  samples = trace_forward(0.0, PLATFORM_X + 1.2, v=0.5)
  result = evaluate_trace(samples, criteria(episode_timeout_s=5.0))
  assert result.success is False
  assert result.failure_reason == FAIL_TIMEOUT
  assert result.elapsed_time == pytest.approx(5.0, abs=0.05)


def test_empty_trace_is_a_reset_failure() -> None:
  result = evaluate_trace([], criteria())
  assert result.success is False
  assert result.failure_reason == FAIL_RESET


# ---------------------------------------------------------------------------
# 7. Stateful evaluator API
# ---------------------------------------------------------------------------


def test_evaluator_returns_none_until_decided() -> None:
  ev = TrialEvaluator(criteria())
  samples = trace_forward(0.0, PLATFORM_X + 1.2, v=0.5)
  results = [ev.update(s) for s in samples]
  decided = [r for r in results if r is not None]
  assert len(decided) == 1
  assert decided[0].success is True
  # Once decided the evaluator is frozen: later updates return None and are the
  # same object the deciding call returned.
  assert results[-1] is None
  assert ev.finalize() is decided[0]


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(pytest.main([__file__, "-q"]))
