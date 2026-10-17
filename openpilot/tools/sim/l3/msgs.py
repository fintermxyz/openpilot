"""L3 messages. Frozen at the start of M1: the modules are built against these"""
import math
from dataclasses import dataclass, field

SCHEMA = "l3/1"

# modes of the L3 layer (M1 uses OFF, ACTIVE, CONSTRAINED, FAILSAFE)
OFF, AVAILABLE, ACTIVE, CONSTRAINED, TOR, TOR_ESC, MRM, MRC, EM, HANDED_OVER, FAILSAFE = (
  "OFF", "AVAILABLE", "ACTIVE", "CONSTRAINED", "TOR", "TOR_ESC", "MRM", "MRC", "EM", "HANDED_OVER", "FAILSAFE")

# signal health
TRUSTED_GREEN, TRUSTED_AMBER, TRUSTED_RED, TRUSTED_RED_AMBER, UNTRUSTED = (
  "TRUSTED_GREEN", "TRUSTED_AMBER", "TRUSTED_RED", "TRUSTED_RED_AMBER", "UNTRUSTED")

# junction manager states
FAR, APPROACH, GO, STOP_AT_LINE, STOPPED, COMMIT, CLEARED = "FAR", "APPROACH", "GO", "STOP_AT_LINE", "STOPPED", "COMMIT", "CLEARED"
# what a car broadcasts while waiting at an unmarked (failed-signal) junction
STOPPED_YIELD = "STOPPED_YIELD"


@dataclass
class Actor:
  """another road user, as the cooperative feed reports it (source 'oracle' in M1: the director's ground truth)"""
  id: str
  route: str
  x: float
  z: float
  psi: float
  v: float
  s: float              # arc length along its own route
  t_obs: float          # time.monotonic() of the observation
  a: float = 0.0
  half_w: float = 0.92
  half_l: float = 2.3
  source: str = "oracle"
  l3: dict | None = None  # its L3 state if it has one: {mode, jx_state, t_stop, token}

  def extrapolated(self, now: float) -> "Actor":
    """constant velocity to now along its heading (s advances by the same distance)"""
    dt = max(0.0, now - self.t_obs)
    ds = self.v * dt
    return Actor(self.id, self.route, self.x + math.sin(self.psi) * ds, self.z + math.cos(self.psi) * ds, self.psi, self.v,
                 self.s + ds, now, self.a, self.half_w, self.half_l, self.source, self.l3)


@dataclass
class WorldMsg:
  """director -> car, 20 Hz: signal phase and timing (SPaT) + every car. There is deliberately no fault flag"""
  trial: int
  seq: int
  t: float                      # director time.monotonic() when sent (one clock on this machine)
  spat: dict[str, str]          # phase -> aspect: 'red' | 'red_amber' | 'green' | 'amber' | 'dark'
  cars: list[Actor] = field(default_factory=list)

  @staticmethod
  def from_json(m: dict) -> "WorldMsg":
    cars = [Actor(id=f"op{c['car']}", route=c["route"], x=c["x"], z=c["z"], psi=c["psi"], v=c["v"], s=c["s"], t_obs=c.get("t", m["t"]),
                  a=c.get("a", 0.0), half_w=c.get("half_w", 0.92), half_l=c.get("half_l", 2.3), l3=c.get("l3")) for c in m.get("cars", [])]
    return WorldMsg(m["trial"], m["seq"], m["t"], dict(m["spat"]), cars)


@dataclass
class EgoState:
  t: float
  id: str
  route: str
  x: float
  z: float
  psi: float
  v: float
  a: float
  s: float
  d: float
  op_active: bool
  half_w: float = 0.92
  half_l: float = 2.3


@dataclass
class SignalHealth:
  state: str            # TRUSTED_* or UNTRUSTED
  cause: str = ""       # conflict | dark | behavioural | stale | ...
  since: float = 0.0


@dataclass
class JunctionOut:
  """what the junction manager wants this tick"""
  state: str
  a_cap: float = math.inf       # m/s^2 upper bound on ego accel (inf = no constraint)
  a_floor: float | None = None  # lower bound when it must keep moving (COMMIT) or brake harder (EM)
  brake_hold: bool = False
  em: bool = False
  token: bool = False
  t_stop: float | None = None
  d_line: float = math.inf
  reasons: list[str] = field(default_factory=list)


@dataclass
class L3Command:
  """supervisor -> arbiter, written under the world lock at 25 Hz, applied at 100 Hz"""
  t_mono: float
  seq: int
  mode: str = OFF
  a_cap: float = math.inf       # accel cap (inf = none)
  a_floor: float = -3.5         # most negative accel the arbiter will command (more negative only in EM / FAILSAFE)
  a_min_keep: float | None = None  # COMMIT: keep at least this accel (never stop in the box), overrides openpilot braking
  commit_clear_d: float | None = None  # (added after the freeze, optional) COMMIT: m still to travel to clear the box (FAILSAFE keeps moving)
  jerk_down: float = 2.5        # informational: braking applies immediately
  brake_hold: bool = False
  steer_source: str = "op"      # 'op' | 'map_pp'
  hazards: bool = False
  hud: list[str] = field(default_factory=list)
  reasons: list[str] = field(default_factory=list)
