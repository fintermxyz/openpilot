"""Simplex arbiter at the simulated actuators (PLAN §4.1). It can only make the car more conservative than openpilot:
accel = min(openpilot, L3 cap), except the COMMIT keep-moving floor (a_min_keep) and the latched FAILSAFE.

Watchdog: a command older than watchdog_s puts the arbiter in a non-latching STALE hold (the last command, with any finite cap
lowered to <= 0: no accelerating while L3 is blind). Only a command older than STALE_LATCH_S, a fail() from the host, a NaN
command or a non-finite openpilot accel latch FAILSAFE. A short stall of the thread that ticks L3 must not throw the trial away.

FAILSAFE never weakens braking already in progress (an emergency stop at -6 stays at -6 until stopped, openpilot braking harder
always wins) and only ramps towards failsafe_a from above. If the last command was COMMIT (a_min_keep with commit_clear_d), it
first keeps moving through the box (>= keep_v, unless openpilot brakes harder than FS_YIELD_A) for commit_clear_d metres."""
import math
import threading
from dataclasses import dataclass

from openpilot.tools.sim.l3.msgs import L3Command
from openpilot.tools.sim.l3.params import ArbiterParams

STALE_LATCH_S = 1.0     # s: a command this old latches FAILSAFE (watchdog_s .. this: non-latching STALE hold)
FS_YIELD_A = -2.5       # m/s^2: in the FAILSAFE commit hold, openpilot braking harder than this wins (a lead it sees, L3 is blind)
KEEP_BRAKE_MAX = 2.0    # m/s^2 / gain of the FAILSAFE commit hold, as the junction's COMMIT floor
KEEP_GAIN = 1.0
KEEP_ACCEL_MAX = 1.0    # m/s^2 (the junction's a_go)


@dataclass
class ArbOut:
  accel: float
  brake_hold: bool
  steer_source: str   # 'op' | 'map_pp'
  source: str         # 'op' (output == openpilot) | 'l3' | 'failsafe'
  failsafe: bool
  hazards: bool


class Arbiter:
  def __init__(self, params: ArbiterParams, keep_v: float = 3.0):
    self.p = params
    self.keep_v = keep_v  # the junction's v_clear: the FAILSAFE commit hold keeps at least this
    self._lock = threading.Lock()
    self.reset()

  def reset(self) -> None:
    # apply() runs entirely under the lock, so a reset from another thread can never interleave with it
    with self._lock:
      self._cmd: L3Command | None = None
      self._fail_reason: str | None = None
      self.failsafe = False
      self.fail_reason = ""
      self.stale = False
      self._last_out: float | None = None
      self._last_bound = False  # last output != a_op (constrained or still ramping back up)
      self._fs_commit_d = 0.0   # m still to cover in the FAILSAFE commit hold
      self.stats = {"ticks": 0, "l3_ticks": 0, "failsafe_ticks": 0, "keep_ticks": 0, "violations": 0, "max_excess": 0.0,
                    "stale_events": 0, "max_stale_s": 0.0, "failsafe_reason": ""}

  def set_command(self, cmd: L3Command) -> None:
    """may be called from the L3 host thread while apply() runs on the actuator thread"""
    with self._lock:
      self._cmd = cmd

  def fail(self, reason: str) -> None:
    """the host's tick raised: latch FAILSAFE on the next apply()"""
    with self._lock:
      if self._fail_reason is None:
        self._fail_reason = reason

  def apply(self, a_op: float, now: float, dt: float, v: float | None = None) -> ArbOut:
    """v (optional): the car's speed, used by the FAILSAFE commit hold"""
    with self._lock:
      out = self._apply(a_op, now, max(dt, 0.0), v)
      st = self.stats
      st["ticks"] += 1
      if out.failsafe:
        st["failsafe_ticks"] += 1
        st["failsafe_reason"] = self.fail_reason
      elif out.source != "op":
        st["l3_ticks"] += 1
      keep = self._cmd is not None and self._cmd.a_min_keep is not None
      if keep and not out.failsafe:
        st["keep_ticks"] += 1
      elif not out.failsafe and math.isfinite(a_op) and out.accel > a_op:
        # the invariant (a_out <= a_op outside FAILSAFE and a_min_keep): counted, never expected
        st["violations"] += 1
        st["max_excess"] = max(st["max_excess"], out.accel - a_op)
      return out

  def _apply(self, a_op: float, now: float, dt: float, v: float | None) -> ArbOut:
    cmd = self._cmd
    age = now - cmd.t_mono if cmd is not None else 0.0

    if not self.failsafe:
      if self._fail_reason is not None:
        self._latch(self._fail_reason, cmd)
      elif cmd is not None and age > STALE_LATCH_S:
        self._latch(f"watchdog: no L3 command for {age:.3f} s", cmd)
      elif cmd is not None and (math.isnan(cmd.a_cap) or math.isnan(cmd.a_floor)):
        self._latch("nan in L3 command", cmd)
      elif not math.isfinite(a_op):
        self._latch("non-finite openpilot accel", cmd)

    if self.failsafe:
      return self._failsafe(a_op, dt, v)

    if cmd is None:
      # no L3 yet: openpilot alone
      self._last_out, self._last_bound = a_op, False
      return ArbOut(a_op, False, "op", "op", False, False)

    stale = age > self.p.watchdog_s
    if stale and not self.stale:
      self.stats["stale_events"] += 1
    if stale:
      self.stats["max_stale_s"] = max(self.stats["max_stale_s"], age)
    self.stale = stale

    # the floor bounds how hard L3 itself may brake; it never weakens openpilot's own braking (keeps a_out <= a_op)
    a_cap = max(cmd.a_cap, cmd.a_floor)
    if stale and math.isfinite(a_cap):
      a_cap = min(a_cap, 0.0)  # blind: hold the constraint, never accelerate under it
    target = min(a_op, a_cap, self.p.a_max)

    prev = self._last_out if self._last_out is not None else a_op
    if target < prev:
      out = target                                      # braking: immediate
    elif target == a_op and not self._last_bound:
      out = a_op                                        # not binding now nor last tick: exact passthrough
    else:
      out = min(target, prev + self.p.jerk_up * dt)     # cap released / rising: ramp (absorbs openpilot PID windup)

    if cmd.a_min_keep is not None:
      out = min(max(out, cmd.a_min_keep), self.p.a_max)  # COMMIT: never stop in the box, overrides openpilot braking

    self._last_out, self._last_bound = out, out != a_op
    return ArbOut(out, cmd.brake_hold, cmd.steer_source, "op" if out == a_op else "l3", False, cmd.hazards)

  def _failsafe(self, a_op: float, dt: float, v: float | None) -> ArbOut:
    # the arbiter owns longitudinal. Ramp down to failsafe_a at failsafe_jerk; braking already harder is held (never weakened),
    # openpilot braking harder still wins. Brake hold once stopped is the caller's job
    prev = self._last_out if self._last_out is not None and math.isfinite(self._last_out) else 0.0
    step = self.p.failsafe_jerk * dt
    out = max(self.p.failsafe_a, prev - step) if prev > self.p.failsafe_a else prev
    if math.isfinite(a_op):
      out = min(out, a_op)
    if self._fs_commit_d > 0.0:
      # committed in the box when L3 died: keep moving through it (cross traffic was granted its gap on that), unless openpilot
      # brakes hard (L3 can no longer see a lead)
      vv = self.keep_v if v is None else max(v, 0.0)
      self._fs_commit_d -= vv * dt
      if not (math.isfinite(a_op) and a_op < FS_YIELD_A):
        keep = 0.0 if v is None else min(KEEP_ACCEL_MAX, max(-KEEP_BRAKE_MAX, KEEP_GAIN * (self.keep_v - vv)))
        out = max(out, keep)
      else:
        self._fs_commit_d = 0.0  # openpilot took the brakes: the hold is over
    self._last_out, self._last_bound = out, True
    return ArbOut(out, False, "map_pp", "failsafe", True, True)

  def _latch(self, reason: str, cmd: L3Command | None) -> None:
    self.failsafe = True
    self.fail_reason = reason
    clear_d = getattr(cmd, "commit_clear_d", None) if cmd is not None else None
    if cmd is not None and cmd.a_min_keep is not None and clear_d is not None and clear_d > 0.0:
      self._fs_commit_d = float(clear_d)
