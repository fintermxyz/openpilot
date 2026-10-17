import math
import random
import threading

import pytest

from openpilot.tools.sim.l3.arbiter import FS_YIELD_A, STALE_LATCH_S, Arbiter
from openpilot.tools.sim.l3.msgs import ACTIVE, CONSTRAINED, L3Command
from openpilot.tools.sim.l3.params import ArbiterParams

P = ArbiterParams()
DT = 0.01


class Clock:
  def __init__(self, t: float = 100.0):
    self.t = t

  def tick(self, dt: float = DT) -> float:
    self.t += dt
    return self.t


def cmd(t: float, seq: int = 0, **kw) -> L3Command:
  return L3Command(t_mono=t, seq=seq, mode=kw.pop("mode", ACTIVE), **kw)


def test_passthrough_before_first_command():
  arb = Arbiter(P)
  for a in (-5.0, 0.0, 1.3, 3.0):
    out = arb.apply(a, 0.0, DT)
    assert out.accel == a and out.source == "op" and not out.failsafe and not out.brake_hold


def test_passthrough_exact_when_not_binding():
  arb, clk = Arbiter(P), Clock()
  rng = random.Random(0)
  for i in range(500):
    now = clk.tick()
    arb.set_command(cmd(now, i, a_cap=math.inf if i % 2 else 1.95))
    a_op = rng.uniform(-3.4, 1.9)
    out = arb.apply(a_op, now, DT)
    assert out.accel == a_op and out.source == "op"


def test_out_le_op_random():
  arb, clk = Arbiter(P), Clock()
  rng = random.Random(1)
  for i in range(5000):
    now = clk.tick()
    if i % 4 == 0:
      arb.set_command(cmd(now, i, mode=CONSTRAINED, a_cap=rng.choice([math.inf, rng.uniform(-6, 2.5)]), a_floor=rng.uniform(-6, -1)))
    a_op = rng.uniform(-6, 3)
    out = arb.apply(a_op, now, DT)
    assert not out.failsafe
    assert out.accel <= a_op
    assert out.accel <= P.a_max


def test_cap_binds_and_downward_immediate():
  arb, clk = Arbiter(P), Clock()
  now = clk.tick()
  arb.set_command(cmd(now, a_cap=math.inf))
  assert arb.apply(1.5, now, DT).accel == 1.5
  now = clk.tick()
  arb.set_command(cmd(now, mode=CONSTRAINED, a_cap=-2.0))
  out = arb.apply(1.5, now, DT)
  assert out.accel == -2.0 and out.source == "l3"
  # openpilot braking harder than the cap wins, immediately
  now = clk.tick()
  arb.set_command(cmd(now, mode=CONSTRAINED, a_cap=-2.0))
  out = arb.apply(-3.0, now, DT)
  assert out.accel == -3.0 and out.source == "op"


def test_floor_bounds_cap_not_openpilot():
  arb, clk = Arbiter(P), Clock()
  now = clk.tick()
  arb.set_command(cmd(now, mode=CONSTRAINED, a_cap=-9.0, a_floor=-3.5))
  assert arb.apply(0.5, now, DT).accel == -3.5
  now = clk.tick()
  arb.set_command(cmd(now, mode=CONSTRAINED, a_cap=-9.0, a_floor=-3.5))
  assert arb.apply(-5.0, now, DT).accel == -5.0  # never weakens openpilot's own braking


def test_a_max_clip():
  arb, clk = Arbiter(P), Clock()
  now = clk.tick()
  arb.set_command(cmd(now))
  out = arb.apply(3.0, now, DT)
  assert out.accel == P.a_max and out.source == "l3"


def test_upward_rate_limit_on_release():
  arb, clk = Arbiter(P), Clock()
  for i in range(50):
    now = clk.tick()
    arb.set_command(cmd(now, i, mode=CONSTRAINED, a_cap=-2.0))
    assert arb.apply(1.0, now, DT).accel == -2.0
  # cap released: ramp at jerk_up from -2.0 to openpilot's 1.0 (3 m/s^2 at 2.5 m/s^3 = 1.2 s)
  outs = []
  for i in range(200):
    now = clk.tick()
    arb.set_command(cmd(now, 100 + i, a_cap=math.inf))
    outs.append(arb.apply(1.0, now, DT))
  for a, b in zip(outs, outs[1:], strict=False):
    assert b.accel - a.accel <= P.jerk_up * DT + 1e-9
  assert outs[0].accel == pytest.approx(-2.0 + P.jerk_up * DT)
  assert outs[0].source == "l3"
  n_ramp = next(i for i, o in enumerate(outs) if o.accel == 1.0)
  assert n_ramp * DT == pytest.approx(3.0 / P.jerk_up, abs=2 * DT)
  # after the ramp: exact passthrough again, including openpilot's own upward steps
  assert all(o.source == "op" for o in outs[n_ramp:])
  now = clk.tick()
  arb.set_command(cmd(now, a_cap=math.inf))
  assert arb.apply(1.9, now, DT).accel == 1.9


def test_rising_cap_is_rate_limited():
  arb, clk = Arbiter(P), Clock()
  now = clk.tick()
  arb.set_command(cmd(now, mode=CONSTRAINED, a_cap=-3.0))
  assert arb.apply(1.0, now, DT).accel == -3.0
  now = clk.tick()
  arb.set_command(cmd(now, mode=CONSTRAINED, a_cap=0.0))
  assert arb.apply(1.0, now, DT).accel == pytest.approx(-3.0 + P.jerk_up * DT)


def test_ramp_interrupted_by_openpilot_braking():
  arb, clk = Arbiter(P), Clock()
  now = clk.tick()
  arb.set_command(cmd(now, mode=CONSTRAINED, a_cap=-2.0))
  arb.apply(1.0, now, DT)
  now = clk.tick()
  arb.set_command(cmd(now, a_cap=math.inf))
  arb.apply(1.0, now, DT)
  now = clk.tick()
  arb.set_command(cmd(now, a_cap=math.inf))
  out = arb.apply(-2.5, now, DT)
  assert out.accel == -2.5 and out.source == "op"
  now = clk.tick()
  arb.set_command(cmd(now, a_cap=math.inf))
  assert arb.apply(0.5, now, DT).accel == 0.5


def test_a_min_keep_overrides_openpilot_braking():
  arb, clk = Arbiter(P), Clock()
  now = clk.tick()
  arb.set_command(cmd(now, mode=CONSTRAINED, a_cap=math.inf, a_floor=0.0, a_min_keep=0.3))
  out = arb.apply(-3.0, now, DT)
  assert out.accel == 0.3 and out.source == "l3" and not out.failsafe


def test_brake_hold_and_flags_from_command():
  arb, clk = Arbiter(P), Clock()
  now = clk.tick()
  arb.set_command(cmd(now, mode=CONSTRAINED, a_cap=-1.0, brake_hold=True, hazards=True, steer_source="map_pp"))
  out = arb.apply(0.5, now, DT)
  assert out.brake_hold and out.hazards and out.steer_source == "map_pp"
  now = clk.tick()
  arb.set_command(cmd(now))
  out = arb.apply(0.5, now, DT)
  assert not out.brake_hold and not out.hazards and out.steer_source == "op"


def _check_failsafe_ramp(arb: Arbiter, clk: Clock, a_op: float = 1.0, start: float | None = None) -> None:
  prev = start
  for _ in range(300):
    now = clk.tick()
    out = arb.apply(a_op, now, DT)
    assert out.failsafe and out.hazards and out.steer_source == "map_pp" and out.source == "failsafe"
    if prev is not None:
      assert abs(out.accel - prev) <= P.failsafe_jerk * DT + 1e-9
    prev = out.accel
  assert prev == pytest.approx(P.failsafe_a)


def test_watchdog_stale_hold_then_latch():
  """watchdog_s .. STALE_LATCH_S without a command: non-latching STALE hold (constraint kept, finite cap lowered to <= 0); only past
  STALE_LATCH_S does FAILSAFE latch. A short stall of the ticking thread must not throw the trial away"""
  arb, clk = Arbiter(P), Clock()
  t_last = None
  for i in range(10):
    t_last = clk.tick()
    arb.set_command(cmd(t_last, i, mode=CONSTRAINED, a_cap=0.8))
    assert not arb.apply(0.5, t_last, DT).failsafe
  latched_at = None
  for _ in range(200):
    now = clk.tick()
    out = arb.apply(1.5, now, DT)
    if out.failsafe:
      latched_at = now
      break
    if now - t_last > P.watchdog_s:
      assert arb.stale and out.accel <= 0.0, (now - t_last, out)  # blind: no accelerating under the held cap
    else:
      assert not arb.stale and out.accel <= 0.8
  assert latched_at is not None
  assert STALE_LATCH_S < latched_at - t_last <= STALE_LATCH_S + DT + 1e-9
  assert arb.stats["stale_events"] == 1
  # latched: fresh commands don't clear it
  arb.set_command(cmd(clk.t + DT, 99))
  _check_failsafe_ramp(arb, clk, start=out.accel)
  assert "watchdog" in arb.fail_reason


def test_stale_hold_recovers_without_latching():
  """a 0.5 s gap between commands (engine round trip stall) on open road: STALE, then normal again, never FAILSAFE"""
  arb, clk = Arbiter(P), Clock()
  now = clk.tick()
  arb.set_command(cmd(now, 0))
  for _ in range(50):
    out = arb.apply(0.7, clk.tick(), DT)
    assert not out.failsafe and out.accel == 0.7  # unconstrained: passthrough even while stale
  assert arb.stale
  for i in range(1, 300):
    now = clk.tick()
    arb.set_command(cmd(now, i))
    out = arb.apply(0.7, now, DT)
    assert not out.failsafe and out.accel == 0.7
  assert not arb.stale and not arb.failsafe and arb.stats["stale_events"] == 1


def test_fail_latches_and_reset_clears():
  arb, clk = Arbiter(P), Clock()
  now = clk.tick()
  arb.set_command(cmd(now))
  arb.apply(0.0, now, DT)
  arb.fail("tick raised ValueError")
  arb.set_command(cmd(clk.t + DT, 1))
  _check_failsafe_ramp(arb, clk, start=0.0)
  assert arb.fail_reason == "tick raised ValueError"
  # failsafe from hard braking ramps gently back to failsafe_a, not a step
  arb.reset()
  assert not arb.failsafe
  out = arb.apply(0.7, clk.tick(), DT)
  assert out.accel == 0.7 and not out.failsafe and out.source == "op"
  now = clk.tick()
  arb.set_command(cmd(now, mode=CONSTRAINED, a_cap=-6.0, a_floor=-6.0))
  assert arb.apply(0.0, now, DT).accel == -6.0
  arb.fail("x")
  # an emergency stop in progress is never weakened by FAILSAFE: it stays at -6 (the caller holds the brake once stopped)
  for _ in range(100):
    out = arb.apply(0.0, clk.tick(), DT)
    assert out.failsafe and out.accel == -6.0


def test_failsafe_never_weaker_than_openpilot_or_em():
  """review finding: FAILSAFE ramped an EM stop from -6 up to -3 and started from +0.5 while openpilot braked at -3.5"""
  # (1) EM at -6, openpilot -6 too, tick raises 0.2 s in: the stop is not lengthened
  def em_stop(fail_at):
    arb, t, v, s = Arbiter(P), 0.0, 8.0, 0.0
    while v > 0.0:
      if fail_at is None or t < fail_at:
        arb.set_command(cmd(t, int(t * 25), mode="EM", a_cap=-6.0, a_floor=-6.0))
      if fail_at is not None and abs(t - fail_at) < DT / 2:
        arb.fail("tick raised")
      a = arb.apply(-6.0, t, DT).accel
      v = max(0.0, v + a * DT)
      s += v * DT
      t += DT
    return s
  assert em_stop(0.2) <= em_stop(None) + 1e-6
  # (2) last output +0.5, openpilot brakes -3.5: FAILSAFE follows openpilot at once and never exceeds it
  arb, clk = Arbiter(P), Clock()
  now = clk.tick()
  arb.set_command(cmd(now, a_cap=1.0))
  arb.apply(0.5, now, DT)
  arb.fail("x")
  for _ in range(300):
    out = arb.apply(-3.5, clk.tick(), DT)
    assert out.failsafe and out.accel <= -3.5


def test_failsafe_in_commit_keeps_moving_through_the_box():
  """review finding: a FAILSAFE latched in COMMIT braked to a standstill inside the box. With commit_clear_d it keeps >= v_clear
  for that distance (unless openpilot brakes hard), then ramps to failsafe_a"""
  def drive(a_op_fn, v0=4.0, clear_d=12.0):
    arb, clk = Arbiter(P, keep_v=3.0), Clock()
    v, s = v0, 0.0
    now = clk.tick()
    arb.set_command(cmd(now, mode=CONSTRAINED, a_cap=1.0, a_min_keep=-1.0, commit_clear_d=clear_d))
    arb.apply(-1.0, now, DT, v=v)
    arb.fail("tick raised")
    stop_s = None
    for _ in range(3000):
      out = arb.apply(a_op_fn(v), clk.tick(), DT, v=v)
      assert out.failsafe
      v = max(0.0, v + out.accel * DT)
      s += v * DT
      if s < clear_d:
        assert v >= 2.5 or a_op_fn(v) < FS_YIELD_A, (s, v)
      if v == 0.0:
        stop_s = s
        break
    return stop_s
  stop_s = drive(lambda v: -1.0)  # openpilot easing down
  assert stop_s is not None and stop_s > 12.0
  # openpilot braking hard (a lead L3 can no longer see): it wins
  assert drive(lambda v: -4.0) < 3.0


def test_thread_safety_set_command_vs_apply():
  arb = Arbiter(P)
  stop = threading.Event()
  errors: list[BaseException] = []
  t0 = 1000.0
  caps = (-2.0, -1.0, 0.5, math.inf)

  def writer():
    i = 0
    try:
      while not stop.is_set():
        # always fresh (t_mono far in the future relative to the reader's clock): the watchdog must never trip
        arb.set_command(cmd(t0 + 1e6, i, mode=CONSTRAINED, a_cap=caps[i % len(caps)], brake_hold=bool(i % 3 == 0)))
        i += 1
    except BaseException as e:
      errors.append(e)

  threads = [threading.Thread(target=writer) for _ in range(3)]
  for t in threads:
    t.start()
  rng = random.Random(2)
  try:
    for i in range(20000):
      a_op = rng.uniform(-4, 2)
      out = arb.apply(a_op, t0 + i * DT, DT)
      assert not out.failsafe
      assert out.accel <= a_op
      assert isinstance(out.brake_hold, bool)
  finally:
    stop.set()
    for t in threads:
      t.join()
  assert not errors
