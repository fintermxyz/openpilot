import math
import os
from types import SimpleNamespace

import pytest

from openpilot.tools.sim.l3.arbiter import STALE_LATCH_S, Arbiter
from openpilot.tools.sim.l3.host import NO_FEED, L3Host
from openpilot.tools.sim.l3.msgs import (ACTIVE, COMMIT, CONSTRAINED, EM, FAILSAFE, FAR, OFF, STOPPED, UNTRUSTED, EgoState, JunctionOut,
                                         WorldMsg)
from openpilot.tools.sim.l3.params import PARAMS
from openpilot.tools.sim.l3.signal_monitor import SignalMonitor

MAP = os.path.join(os.environ.get("ZOOMENGINE_DIR", os.path.expanduser("~/Documents/zoom/ZoomEngine-main")), "assets/maps/crossroads.built.json")


def fake_map():
  # same numbers as crossroads.built.json: stop line s 229, box s 231.35..250.65
  def mk(ph):
    return SimpleNamespace(phase=ph, stop_line_s=229.0, junction_s=(231.35, 250.65))
  return SimpleNamespace(routes={"N_S": mk("NS"), "S_N": mk("NS"), "E_W": mk("EW"), "W_E": mk("EW")})


class Clock:
  def __init__(self, t: float = 100.0):
    self.t = t

  def __call__(self) -> float:
    return self.t


class FakeJunction:
  def __init__(self, out: JunctionOut | None = None, exc: Exception | None = None):
    self.out = out if out is not None else JunctionOut(FAR)
    self.exc = exc
    self.calls: list[tuple] = []
    self.resets = 0

  def reset(self):
    self.resets += 1

  def step(self, ego, actors, signal, now):
    self.calls.append((ego, actors, signal, now))
    if self.exc is not None:
      raise self.exc
    return self.out


def ego(t: float, op_active: bool = True, s: float = 150.0) -> EgoState:
  return EgoState(t, "op0", "N_S", -1.825, 241.0 - s, math.pi, 8.0, 0.0, s, 0.0, op_active)


def car(cid: int, route: str, s: float, v: float, t: float, l3: dict | None = None) -> dict:
  return {"car": cid, "route": route, "x": 0.0, "z": 0.0, "psi": 0.0, "v": v, "s": s, "t": t, "l3": l3}


def world(t: float, cars: list[dict], spat: dict | None = None, seq: int = 0) -> WorldMsg:
  return WorldMsg.from_json({"type": "world", "trial": 0, "seq": seq, "t": t, "spat": spat or {"NS": "green", "EW": "red"}, "cars": cars})


def make_host(clock: Clock, junction: FakeJunction) -> L3Host:
  cmap = fake_map()
  return L3Host("op0", "N_S", cmap, clock=clock, signal=SignalMonitor(PARAMS.signal, "N_S", cmap), junction=junction)


# ---- actors ----
def test_actors_drop_ego_extrapolate_and_mark_stale_unknown():
  clock, jx = Clock(100.0), FakeJunction()
  host = make_host(clock, jx)
  host.feed(world(100.0, [car(0, "N_S", 150.0, 8.0, 100.0), car(1, "S_N", 100.0, 5.0, 99.9), car(2, "E_W", 200.0, 4.0, 99.0)]))
  clock.t = 100.2
  host.tick(ego(clock.t))
  _, actors, _, now = jx.calls[-1]
  assert now == pytest.approx(100.2)
  by_id = {a.id: a for a in actors}
  assert set(by_id) == {"op1", "op2"}  # ego (op0) dropped
  assert by_id["op1"].source == "oracle"  # 0.3 s old <= feed_stale_s
  assert by_id["op1"].s == pytest.approx(100.0 + 5.0 * 0.3)
  assert by_id["op1"].t_obs == pytest.approx(100.2)
  assert by_id["op2"].source == "unknown"  # 1.2 s old > feed_stale_s: worst case, never assumed gone
  assert by_id["op2"].s == pytest.approx(200.0 + 4.0 * 1.2)


def test_whole_feed_goes_unknown_when_director_stops():
  clock, jx = Clock(100.0), FakeJunction()
  host = make_host(clock, jx)
  host.feed(world(100.0, [car(1, "S_N", 100.0, 5.0, 100.0), car(2, "E_W", 200.0, 4.0, 100.0)]))
  clock.t = 100.0 + PARAMS.feed_stale_s + 0.05
  host.tick(ego(clock.t))
  _, actors, sig, _ = jx.calls[-1]
  assert len(actors) == 2 and all(a.source == "unknown" for a in actors)
  # the SPaT is that old too, but under stale_s: still trusted
  assert sig.state != UNTRUSTED
  clock.t = 100.0 + PARAMS.signal.stale_s + 0.05
  host.tick(ego(clock.t))
  assert jx.calls[-1][2].state == UNTRUSTED and jx.calls[-1][2].cause == "stale"


def test_no_world_message_yet_is_untrusted_but_not_latched():
  clock, jx = Clock(), FakeJunction()
  host = make_host(clock, jx)
  assert host.tick(ego(clock.t)) is not None
  _, actors, sig, _ = jx.calls[-1]
  assert actors == [] and sig.state == UNTRUSTED and sig.cause == NO_FEED
  # the first message arrives a tick later (director reset -> world): trusted, the monitor never latched 'stale'
  clock.t += 0.04
  host.feed(world(clock.t, []))
  host.tick(ego(clock.t))
  assert jx.calls[-1][2].state == "TRUSTED_GREEN"
  # after a feed existed, losing it is 'stale' (latched)
  clock.t += PARAMS.signal.stale_s + 0.1
  host.tick(ego(clock.t))
  clock.t += 0.04
  host.feed(world(clock.t, []))
  host.tick(ego(clock.t))
  assert jx.calls[-1][2].state == UNTRUSTED and jx.calls[-1][2].cause == "stale"


def test_l3_dict_relayed_to_actor():
  clock, jx = Clock(), FakeJunction()
  host = make_host(clock, jx)
  l3 = {"mode": ACTIVE, "jx_state": STOPPED, "t_stop": 99.0, "token": False}
  host.feed(world(clock.t, [car(2, "E_W", 225.0, 0.0, clock.t, l3)]))
  host.tick(ego(clock.t))
  assert jx.calls[-1][1][0].l3 == l3


# ---- failures ----
def test_exception_in_junction_latches_failsafe_and_never_raises(capsys):
  clock, jx = Clock(), FakeJunction(exc=ZeroDivisionError("boom"))
  arb = Arbiter(PARAMS.arbiter)
  cmap = fake_map()
  host = L3Host("op0", "N_S", cmap, clock=clock, signal=SignalMonitor(PARAMS.signal, "N_S", cmap), junction=jx, arbiter=arb)
  host.feed(world(clock.t, []))
  for _ in range(3):
    assert host.tick(ego(clock.t)) is None
    clock.t += 0.04
  assert capsys.readouterr().out.count("FAILSAFE") == 1  # logged once
  assert host.mode == FAILSAFE and host.status()["mode"] == FAILSAFE
  out = arb.apply(1.0, clock.t, 0.01)
  assert out.failsafe and out.steer_source == "map_pp" and out.hazards
  assert out.accel < 1.0
  # latched until reset
  host.reset()
  assert host.mode == OFF
  assert not arb.apply(1.0, clock.t, 0.01).failsafe


def test_exception_in_signal_monitor_is_caught():
  clock, jx = Clock(), FakeJunction()
  host = make_host(clock, jx)
  host.signal.update = lambda *a: (_ for _ in ()).throw(RuntimeError("bad spat"))
  host.feed(world(clock.t, []))
  assert host.tick(ego(clock.t)) is None
  assert host.mode == FAILSAFE and "bad spat" in host.reason


def test_watchdog_when_tick_stops():
  clock, jx = Clock(), FakeJunction()
  host = make_host(clock, jx)
  host.feed(world(clock.t, []))
  host.tick(ego(clock.t))
  assert not host.arbiter.apply(0.5, clock.t + 0.1, 0.01).failsafe
  assert not host.arbiter.apply(0.5, clock.t + PARAMS.arbiter.watchdog_s + 0.05, 0.01).failsafe  # stale hold, not latched
  assert host.arbiter.stale
  assert host.arbiter.apply(0.5, clock.t + STALE_LATCH_S + 0.05, 0.01).failsafe
  assert host.mode == FAILSAFE


# ---- commands and status ----
def test_unconstrained_command_is_active_passthrough():
  clock, jx = Clock(), FakeJunction(JunctionOut(FAR))
  host = make_host(clock, jx)
  host.feed(world(clock.t, []))
  cmd = host.tick(ego(clock.t))
  assert cmd.mode == ACTIVE and math.isinf(cmd.a_cap) and cmd.a_min_keep is None and cmd.a_floor == PARAMS.arbiter.a_floor
  assert cmd.t_mono == clock.t and cmd.seq == 1
  assert host.arbiter.apply(0.7, clock.t, 0.01).accel == 0.7
  st = host.status()
  assert st == {"mode": ACTIVE, "jx_state": FAR, "t_stop": None, "token": False, "a_cap": None, "signal": "TRUSTED_GREEN", "cause": "",
                "em": False}
  assert host.tick(ego(clock.t)).seq == 2


def test_status_fields_constrained_and_commit_floor():
  clock = Clock()
  jx = FakeJunction(JunctionOut(STOPPED, a_cap=-1.5, brake_hold=True, token=True, t_stop=97.5, d_line=1.2, reasons=["queue"]))
  host = make_host(clock, jx)
  host.feed(world(clock.t, [], spat={"NS": "green", "EW": "green"}))
  host.tick(ego(clock.t))
  clock.t += 0.2  # past the conflict debounce
  host.feed(world(clock.t, [], spat={"NS": "green", "EW": "green"}))
  cmd = host.tick(ego(clock.t))
  assert cmd.mode == CONSTRAINED and cmd.a_cap == -1.5 and cmd.brake_hold
  assert "queue" in cmd.reasons
  st = host.status()
  assert st["mode"] == CONSTRAINED and st["jx_state"] == STOPPED and st["t_stop"] == 97.5 and st["token"] is True and st["a_cap"] == -1.5
  assert st["signal"] == UNTRUSTED and st["cause"] == "conflict"
  assert any("CONSTRAINED" in line for line, _ in host.hud_lines())

  jx.out = JunctionOut(COMMIT, a_floor=0.5)
  cmd = host.tick(ego(clock.t))
  assert cmd.mode == CONSTRAINED and cmd.a_min_keep == 0.5 and cmd.a_floor == PARAMS.arbiter.a_floor
  assert host.arbiter.apply(-2.0, clock.t, 0.01).accel >= 0.5  # never stops in the box


def test_em_and_op_inactive_modes():
  clock = Clock()
  jx = FakeJunction(JunctionOut(STOPPED, a_cap=-PARAMS.junction.a_em, a_floor=-PARAMS.junction.a_em, em=True))
  host = make_host(clock, jx)
  host.feed(world(clock.t, []))
  cmd = host.tick(ego(clock.t))
  assert cmd.mode == EM and cmd.a_floor == -PARAMS.junction.a_em and cmd.hazards and cmd.a_min_keep is None
  assert host.arbiter.apply(0.0, clock.t, 0.01).accel == pytest.approx(-PARAMS.junction.a_em)
  assert host.status()["em"] is True
  cmd = host.tick(ego(clock.t, op_active=False))
  assert cmd.mode == OFF and host.status()["mode"] == OFF


def test_reset_clears_feed_and_resets_modules():
  clock, jx = Clock(), FakeJunction()
  host = make_host(clock, jx)
  host.feed(world(clock.t, [car(1, "S_N", 100.0, 5.0, clock.t)]))
  host.tick(ego(clock.t))
  n = jx.resets
  host.reset()
  assert jx.resets == n + 1 and host.status()["jx_state"] is None and host.actors(clock.t) == []


# ---- the real modules, end to end (no engine) ----
@pytest.mark.skipif(not os.path.exists(MAP), reason="crossroads map not found")
def test_real_modules_smoke():
  from openpilot.tools.sim.bridge.zoomengine.crossroads import CrossroadsMap
  cmap = CrossroadsMap(MAP)
  clock = Clock()
  host = L3Host("op0", "N_S", cmap, clock=clock)
  modes = set()
  s, v = 150.0, 8.0
  for i in range(250):
    other_s = 150.0 + 6.0 * i * 0.04
    x, z, psi = cmap.routes["E_W"].track.pose(other_s, 0.0)
    host.feed(WorldMsg.from_json({"type": "world", "trial": 0, "seq": i, "t": clock.t, "spat": {"NS": "green", "EW": "green"},
                                  "cars": [{"car": 2, "route": "E_W", "x": x, "z": z, "psi": psi, "v": 6.0, "s": other_s, "t": clock.t}]}))
    ex, ez, epsi = cmap.routes["N_S"].track.pose(s, 0.0)
    cmd = host.tick(EgoState(clock.t, "op0", "N_S", ex, ez, epsi, v, 0.0, s, 0.0, True))
    assert cmd is not None, host.reason
    modes.add(cmd.mode)
    out = host.arbiter.apply(0.5, clock.t, 0.04)
    assert out.accel <= 0.5 or cmd.a_min_keep is not None
    v = max(0.0, v + out.accel * 0.04)
    s += v * 0.04
    clock.t += 0.04
  assert host.status()["signal"] == UNTRUSTED
  assert CONSTRAINED in modes
