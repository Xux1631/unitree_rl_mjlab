"""Pure success/failure logic for the four-step-staircase evaluation.

This module deliberately has **no mjlab / mujoco / torch imports** so the
assessment's success criteria (assessment section 4.3) can be unit-tested on a
CPU-only machine.  ``scripts/eval_stairs.py`` is the thin runtime adapter that
reads the robot state out of a live environment and feeds :class:`TrialSample`
values into :class:`TrialEvaluator`.

Success (all must hold)
-----------------------
1. the main body crosses the stairs and reaches the top platform,
2. after reaching the platform it keeps moving forward at least ``1.0 m``,
3. the body / knees / other non-permitted parts do not *persistently* contact
   the ground or the steps,
4. no fall condition is triggered.

Failure (any one)
-----------------
* body touches the ground or the steps (sustained beyond ``body_contact_grace_s``,
  or a hard ``illegal_contact`` termination from the environment),
* ``|roll|`` or ``|pitch|`` exceeds 45 deg for longer than ``0.5 s`` continuously,
* the robot is stuck (no forward progress for ``stuck_timeout_s``),
* the episode ends / resets before the course is completed,
* the trial exceeds ``episode_timeout_s``.

Frames and units
----------------
* ``TrialSample.x`` / ``.z`` are metres.  ``x`` is measured in the *course
  frame*: the origin is the nominal spawn point on the flat approach (the
  environment's ``env_origins`` for that patch), so the first riser sits at
  ``+spawn_offset`` and the top platform starts at
  ``spawn_offset + num_steps * step_tread``.  ``z`` is the world height of the
  base link, i.e. ground level on the approach is ``z = 0``.
* ``roll`` / ``pitch`` are radians (feed them through
  :func:`quat_wxyz_to_roll_pitch` if you only have a quaternion).
* Times are seconds since episode start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

__all__ = [
  "MANDATED_NUM_STEPS",
  "MANDATED_STEP_HEIGHT",
  "MANDATED_STEP_TREAD",
  "MANDATED_TOTAL_RISE",
  "FAIL_OK",
  "FAIL_BODY_CONTACT",
  "FAIL_ROLL",
  "FAIL_PITCH",
  "FAIL_STUCK",
  "FAIL_TIMEOUT",
  "FAIL_RESET",
  "CourseGeometry",
  "assert_mandated_geometry",
  "SuccessCriteria",
  "TrialSample",
  "TrialResult",
  "TrialEvaluator",
  "evaluate_trace",
  "replace_sample",
  "quat_wxyz_to_roll_pitch",
]

# ---------------------------------------------------------------------------
# Mandated assessment geometry (assessment section 4.3 / terrain spec).
# ---------------------------------------------------------------------------

MANDATED_NUM_STEPS = 4
MANDATED_STEP_HEIGHT = 0.13
MANDATED_STEP_TREAD = 0.30
MANDATED_TOTAL_RISE = MANDATED_NUM_STEPS * MANDATED_STEP_HEIGHT  # 0.52 m

# Machine-readable ``failure_reason`` tokens required by ``results.csv``.
FAIL_OK = "ok"
FAIL_BODY_CONTACT = "body_contact"
FAIL_ROLL = "roll_exceeded"
FAIL_PITCH = "pitch_exceeded"
FAIL_STUCK = "stuck"
FAIL_TIMEOUT = "timeout"
FAIL_RESET = "reset"

# Environment termination term names (see
# ``src/tasks/velocity/config/quadruped_3320/env_cfgs.py``).
_ENV_REASON_BODY_CONTACT = "body_contact"
_ENV_REASON_ORIENTATION = "orientation"


def quat_wxyz_to_roll_pitch(quat: tuple[float, float, float, float] | list[float]):
  """Convert a ``(w, x, y, z)`` MuJoCo quaternion to ``(roll, pitch)`` radians.

  Roll is the rotation about ``+x`` (forward), pitch about ``+y`` (left).
  """
  w, x, y, z = (float(v) for v in quat)
  norm = math.sqrt(w * w + x * x + y * y + z * z)
  if norm == 0.0:
    raise ValueError("Cannot convert a zero quaternion to roll/pitch.")
  w, x, y, z = w / norm, x / norm, y / norm, z / norm
  roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
  pitch = math.asin(sin_pitch)
  return roll, pitch


@dataclass(frozen=True)
class CourseGeometry:
  """Geometry of the straight staircase in the course frame (metres)."""

  num_steps: int = MANDATED_NUM_STEPS
  step_height: float = MANDATED_STEP_HEIGHT
  step_tread: float = MANDATED_STEP_TREAD
  spawn_offset: float = 0.9
  """Distance from the course origin (spawn point) to the first riser."""
  platform_depth: float = 1.7
  """Depth of the flat top platform after the last step."""

  @property
  def total_rise(self) -> float:
    """Height of the top platform above the approach, in metres."""
    return self.num_steps * self.step_height

  @property
  def first_riser_x(self) -> float:
    return self.spawn_offset

  @property
  def platform_start_x(self) -> float:
    """``x`` at which the flat top platform begins (far edge of the last step)."""
    return self.spawn_offset + self.num_steps * self.step_tread

  @property
  def platform_end_x(self) -> float:
    return self.platform_start_x + self.platform_depth


def assert_mandated_geometry(geometry: CourseGeometry, tol: float = 1e-9) -> None:
  """Fail loudly if ``geometry`` is not the mandated 4 x 0.13 m / 0.30 m flight.

  The top-platform height (0.52 m) is asserted explicitly because the
  assessment mandates it and the success criterion "reached the top platform"
  is derived from it.
  """
  if geometry.num_steps != MANDATED_NUM_STEPS:
    raise ValueError(
      f"Mandated num_steps is {MANDATED_NUM_STEPS}, got {geometry.num_steps}."
    )
  if abs(geometry.step_height - MANDATED_STEP_HEIGHT) > tol:
    raise ValueError(
      f"Mandated step_height is {MANDATED_STEP_HEIGHT}, got {geometry.step_height}."
    )
  if abs(geometry.step_tread - MANDATED_STEP_TREAD) > tol:
    raise ValueError(
      f"Mandated step_tread is {MANDATED_STEP_TREAD}, got {geometry.step_tread}."
    )
  if abs(geometry.total_rise - MANDATED_TOTAL_RISE) > tol:
    raise ValueError(
      f"Mandated total rise is {MANDATED_TOTAL_RISE}, got {geometry.total_rise}."
    )


@dataclass(frozen=True)
class SuccessCriteria:
  """All tunable thresholds of the assessment's success determination."""

  geometry: CourseGeometry = field(default_factory=CourseGeometry)

  post_platform_travel_m: float = 1.0
  """Forward travel required *after* the platform is first reached (criterion 2)."""

  platform_reach_z_tol_m: float = 0.10
  """The base must be at least ``total_rise - tol`` high to count as "on the
  platform".  On the approach the base link sits well below this value, so the
  check rejects "reached x but still at the bottom of the flight"."""

  platform_reach_x_tol_m: float = 0.0
  """Tolerance on the platform-start ``x`` (0 = the base must cross the far edge
  of the last step exactly)."""

  orientation_limit_rad: float = math.radians(45.0)
  """``|roll|`` / ``|pitch|`` limit.  Mandated: 45 degrees."""

  orientation_limit_duration_s: float = 0.5
  """How long the orientation limit may be exceeded continuously.  A brief
  spike shorter than this must NOT fail the trial."""

  body_contact_grace_s: float = 0.2
  """How long a non-foot contact may persist before it counts as a failure.
  The environment's hard ``illegal_contact`` termination (force > 10 N) fails a
  trial immediately regardless of this grace period."""

  stuck_timeout_s: float = 6.0
  """No forward progress for longer than this fails the trial ("stuck at the
  stairs for a long time and cannot keep advancing")."""
  stuck_progress_eps_m: float = 0.05
  """Forward progress smaller than this does not refresh the stuck timer."""

  episode_timeout_s: float = 30.0
  """Hard per-trial wall-clock limit (seconds of simulated time)."""

  def __post_init__(self) -> None:
    if self.post_platform_travel_m <= 0.0:
      raise ValueError("post_platform_travel_m must be > 0.")
    if self.orientation_limit_duration_s <= 0.0:
      raise ValueError("orientation_limit_duration_s must be > 0.")
    if self.stuck_timeout_s <= 0.0:
      raise ValueError("stuck_timeout_s must be > 0.")
    if self.episode_timeout_s <= 0.0:
      raise ValueError("episode_timeout_s must be > 0.")


@dataclass(frozen=True)
class TrialSample:
  """One control-step observation of the robot, in the course frame.

  The runtime adapter produces these; unit tests construct them by hand.
  """

  t: float
  """Seconds since episode start."""
  x: float
  """Base-link forward position in the course frame (metres)."""
  z: float
  """Base-link world height (metres); approach ground is ``z = 0``."""
  roll: float
  pitch: float
  body_contact: bool = False
  """True if a non-foot collision geom (base/knee/thigh/calf/...) touches the
  terrain during this control step."""
  env_done: bool = False
  """True if the environment terminated/reset during this step."""
  env_reason: str | None = None
  """``"body_contact"``, ``"orientation"`` or ``None`` (mapped to ``reset``)."""
  contact_force_n: float = 0.0
  """Peak non-foot contact force during the step (informational only)."""


@dataclass
class TrialResult:
  """Outcome of one trial; the fields mirror ``results.csv``."""

  success: bool
  failure_reason: str
  elapsed_time: float
  max_abs_roll: float
  max_abs_pitch: float
  platform_entry_time: float | None = None
  platform_entry_x: float | None = None
  max_x: float = 0.0
  post_platform_travel: float = 0.0
  steps: int = 0
  notes: str = ""

  @property
  def max_abs_roll_deg(self) -> float:
    return math.degrees(self.max_abs_roll)

  @property
  def max_abs_pitch_deg(self) -> float:
    return math.degrees(self.max_abs_pitch)


class TrialEvaluator:
  """Stateful, step-by-step implementation of the success criteria.

  Feed samples in chronological order through :meth:`update`.  ``update``
  returns ``None`` while the trial is still undecided and a :class:`TrialResult`
  exactly once, at the step where the trial becomes a success or a failure.
  """

  def __init__(self, criteria: SuccessCriteria | None = None) -> None:
    self.criteria = criteria or SuccessCriteria()
    self._last_t: float | None = None
    self.max_abs_roll = 0.0
    self.max_abs_pitch = 0.0
    self.steps = 0

    self._max_x = -math.inf
    self._progress_x = -math.inf
    self._last_progress_t = 0.0

    self._contact_duration = 0.0
    self._orient_duration = 0.0
    self._orient_axis: str | None = None

    self._platform_reached = False
    self._platform_entry_x: float | None = None
    self._platform_entry_t: float | None = None
    self._last_sample: TrialSample | None = None
    self._decided: TrialResult | None = None

  # -- internals ------------------------------------------------------------

  def _result(self, success: bool, reason: str, t: float) -> TrialResult:
    entry_x = self._platform_entry_x
    max_x = self._max_x if self._max_x > -math.inf else 0.0
    travel = 0.0 if entry_x is None else max(0.0, max_x - entry_x)
    notes = (
      f"steps={self.steps}; max_x={max_x:.3f}; "
      f"platform_entry_x="
      + ("n/a" if entry_x is None else f"{entry_x:.3f}")
      + f"; post_platform_travel={travel:.3f}"
    )
    return TrialResult(
      success=success,
      failure_reason=reason,
      elapsed_time=t,
      max_abs_roll=self.max_abs_roll,
      max_abs_pitch=self.max_abs_pitch,
      platform_entry_time=self._platform_entry_t,
      platform_entry_x=entry_x,
      max_x=max_x,
      post_platform_travel=travel,
      steps=self.steps,
      notes=notes,
    )

  def _decide(self, success: bool, reason: str, t: float) -> TrialResult:
    """Commit and return the single result of this trial."""
    self._decided = self._result(success, reason, t)
    return self._decided

  def _env_failure_reason(self, sample: TrialSample) -> str:
    if sample.env_reason == _ENV_REASON_BODY_CONTACT:
      return FAIL_BODY_CONTACT
    if sample.env_reason == _ENV_REASON_ORIENTATION:
      if abs(sample.roll) >= abs(sample.pitch):
        return FAIL_ROLL
      return FAIL_PITCH
    if sample.env_reason == FAIL_TIMEOUT:
      return FAIL_TIMEOUT
    return FAIL_RESET

  # -- public API -----------------------------------------------------------

  def update(self, sample: TrialSample) -> TrialResult | None:
    """Process one control-step sample; return a result once decided.

    Returns ``None`` while the trial is undecided.  After the trial has been
    decided the evaluator is frozen and further calls return ``None`` (the
    result is available from the call that decided the trial, or from
    :meth:`finalize`).
    """
    if self._decided is not None:
      return None

    self._last_sample = sample
    self.steps += 1

    if self._last_t is None:
      dt = 0.0
    else:
      dt = max(0.0, sample.t - self._last_t)
    self._last_t = sample.t

    # Exact running maximum of the base's forward position (success criterion).
    self._max_x = max(self._max_x, sample.x)

    abs_roll = abs(sample.roll)
    abs_pitch = abs(sample.pitch)
    self.max_abs_roll = max(self.max_abs_roll, abs_roll)
    self.max_abs_pitch = max(self.max_abs_pitch, abs_pitch)

    # 1. Orientation limit, continuous-duration based (a short spike is fine).
    if max(abs_roll, abs_pitch) > self.criteria.orientation_limit_rad:
      self._orient_duration += dt
      self._orient_axis = "roll" if abs_roll >= abs_pitch else "pitch"
    else:
      self._orient_duration = 0.0

    # 2. Non-foot ground/step contact, continuous-duration based.
    if sample.body_contact:
      self._contact_duration += dt
    else:
      self._contact_duration = 0.0

    # 3. Forward progress monitor (drives the "stuck" failure only; the 1 m
    # criterion uses the exact running maximum above).
    if sample.x > self._progress_x + self.criteria.stuck_progress_eps_m:
      self._progress_x = sample.x
      self._last_progress_t = sample.t

    # --- terminal decisions ------------------------------------------------
    # A fall / hard body contact from the environment takes precedence, so a
    # robot that crashes *into* the platform is never scored as a success.
    if sample.env_done:
      return self._decide(False, self._env_failure_reason(sample), sample.t)

    if self._contact_duration > self.criteria.body_contact_grace_s:
      return self._decide(False, FAIL_BODY_CONTACT, sample.t)

    if self._orient_duration > self.criteria.orientation_limit_duration_s:
      reason = FAIL_ROLL if self._orient_axis == "roll" else FAIL_PITCH
      return self._decide(False, reason, sample.t)

    # "Reached the top platform": base crossed the last step's far edge AND is
    # at platform height.
    if not self._platform_reached:
      geom = self.criteria.geometry
      on_platform_x = sample.x >= geom.platform_start_x - self.criteria.platform_reach_x_tol_m
      on_platform_z = (
        sample.z >= geom.total_rise - self.criteria.platform_reach_z_tol_m
      )
      if on_platform_x and on_platform_z:
        self._platform_reached = True
        self._platform_entry_x = sample.x
        self._platform_entry_t = sample.t

    if (
      self._platform_reached
      and self._platform_entry_x is not None
      and self._max_x - self._platform_entry_x >= self.criteria.post_platform_travel_m
    ):
      return self._decide(True, FAIL_OK, sample.t)

    if sample.t - self._last_progress_t > self.criteria.stuck_timeout_s:
      return self._decide(False, FAIL_STUCK, sample.t)

    if sample.t >= self.criteria.episode_timeout_s:
      return self._decide(False, FAIL_TIMEOUT, sample.t)

    return None

  @property
  def platform_reached(self) -> bool:
    return self._platform_reached

  def finalize(
    self, reason: str = FAIL_RESET, t: float | None = None
  ) -> TrialResult:
    """Close out a trial that ended without a decision (e.g. episode reset)."""
    if self._decided is not None:
      return self._decided
    if t is None:
      t = self._last_t if self._last_t is not None else 0.0
    return self._decide(False, reason, t)


def replace_sample(sample: TrialSample, **changes) -> TrialSample:
  """Convenience wrapper around :func:`dataclasses.replace` for samples."""
  return replace(sample, **changes)


def evaluate_trace(
  samples: list[TrialSample] | tuple[TrialSample, ...],
  criteria: SuccessCriteria | None = None,
) -> TrialResult:
  """Run a complete synthetic/recorded trace through a fresh evaluator.

  If the trace ends before a decision is reached, the trial is treated as an
  incomplete course and reported as a ``reset`` failure (the episode ended
  without achieving the goal).
  """
  evaluator = TrialEvaluator(criteria)
  for sample in samples:
    result = evaluator.update(sample)
    if result is not None:
      return result
  last = samples[-1] if samples else TrialSample(0.0, 0.0, 0.0, 0.0, 0.0)
  result = evaluator.finalize(FAIL_RESET, last.t)
  result.notes += "; trace ended without a decision"
  return result
