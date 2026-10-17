"""every L3 tunable, with where it comes from. Provisional values are marked; they are simulator design values, not road-validated"""
import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SignalParams:
  conflict_debounce_s: float = 0.15   # both phases non-red this long: conflicting greens (target detection <= 0.3 s)
  dark_s: float = 0.3                 # own head dark / unknown this long
  stale_s: float = 1.0                # SPaT older than this
  behavioural_v: float = 3.0          # m/s: a cross-route car this fast that can no longer stop for its line while we're green
  behavioural_brake: float = 3.0      # m/s^2 that "can still stop" assumes
  # trust is latched until the trial resets (M2: two clean cycles)


@dataclass(frozen=True)
class JunctionParams:
  approach_d: float = 80.0      # m before the stop line the approach envelope starts
  a_b: float = 2.5              # m/s^2: until GO the car can always stop at the line with this (provisional, M0 calibration)
  rho: float = 0.5              # s ego response time (modeld + controls + engine lag, provisional)
  line_margin: float = 1.0      # m short of the stop line it stops
  k_v: float = 1.0              # 1/s envelope speed tracking gain
  hold_v: float = 0.3           # m/s below this near the line: brake hold
  hold_d: float = 3.0
  stop_v: float = 0.2           # unmarked rule: a "stop" is below this for stop_t
  stop_t: float = 1.0
  tie_s: float = 0.3            # stop times this close are a tie, broken by the lower car id
  deadlock_s: float = 8.0       # nobody moved this long: lowest (t_stop, id) whose gap test passes may go
  pet_min: float = 2.0          # s post-encroachment time margin the gap test demands
  a_go: float = 1.0             # m/s^2 ego planned launch accel through the junction
  v_cross_max: float = 8.0      # m/s ego planned crossing speed cap
  rho_other: float = 1.0        # s other car reaction time before it can accelerate
  rho_other_stopped: float = 0.5
  a_other_max: float = 2.0      # m/s^2 other car accel envelope
  v_other_factor: float = 1.1   # other car speed envelope: this x the limit
  a_other_brake: float = 4.0    # m/s^2 a decelerating car is assumed to stop if it needs at most this
  a_em: float = 6.0             # m/s^2 emergency braking (provisional: 0.85 x ABS full brake, M0 calibration)
  v_clear: float = 3.0          # m/s minimum speed while committed in the box
  exit_v: float = 2.0           # m/s a car slower than this just past the box blocks the exit
  exit_gap: float = 3.0         # m beyond the ego length that must be free past the box
  swept_half_w_margin: float = 0.5
  amber_stop_a: float = 3.0     # Highway Code rule 175: go on amber only if stopping would need more than this


@dataclass(frozen=True)
class ArbiterParams:
  a_floor: float = -3.5         # nominal most negative cap
  a_max: float = 2.0
  jerk_up: float = 2.5          # m/s^3: releasing a cap ramps up at most this (absorbs openpilot's PID windup)
  watchdog_s: float = 0.2       # no fresh L3 command for this long: FAILSAFE
  failsafe_a: float = -3.0
  failsafe_jerk: float = 2.5


@dataclass(frozen=True)
class L3Params:
  signal: SignalParams = field(default_factory=SignalParams)
  junction: JunctionParams = field(default_factory=JunctionParams)
  arbiter: ArbiterParams = field(default_factory=ArbiterParams)
  feed_stale_s: float = 0.5     # actor feed older than this: actors are "unknown" (worst-case arrival)
  tick_hz: float = 25.0

  def hash(self) -> str:
    return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


PARAMS = L3Params()
