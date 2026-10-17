import inspect
import os
from types import SimpleNamespace

import pytest

from openpilot.tools.sim.bridge.zoomengine.crossroads import SignalController
from openpilot.tools.sim.l3 import signal_monitor
from openpilot.tools.sim.l3.msgs import TRUSTED_AMBER, TRUSTED_GREEN, TRUSTED_RED, TRUSTED_RED_AMBER, UNTRUSTED, Actor
from openpilot.tools.sim.l3.params import SignalParams
from openpilot.tools.sim.l3.signal_monitor import SignalMonitor

MAP = os.path.join(os.environ.get("ZOOMENGINE_DIR", os.path.expanduser("~/Documents/zoom/ZoomEngine-main")), "assets/maps/crossroads.built.json")
DT = 0.04  # 25 Hz host tick
P = SignalParams()


def fake_map():
  # same numbers as crossroads.built.json: stop line s 229, box s 231.35..250.65
  def mk(ph):
    return SimpleNamespace(phase=ph, stop_line_s=229.0, junction_s=(231.35, 250.65))
  return SimpleNamespace(routes={"N_S": mk("NS"), "S_N": mk("NS"), "E_W": mk("EW"), "W_E": mk("EW")})


def actor(route, s, v, cid="op2"):
  return Actor(cid, route, 0.0, 0.0, 0.0, v, s, 0.0)


def run(mon, spat_fn, t0, t1, actors=(), age=0.05):
  """tick the monitor at 25 Hz with fresh SPaT; return [(t, health)]"""
  out, t = [], t0
  while t < t1 - 1e-9:
    out.append((t, mon.update(spat_fn(t), t - age, list(actors), t)))
    t += DT
  return out


@pytest.fixture
def mon():
  return SignalMonitor(P, "N_S", fake_map())


def test_normal_uk_cycle_trusted(mon):
  sc = SignalController()
  seen = set()
  for t, h in run(mon, sc.aspects, 0.0, 3 * sc.cycle):
    assert h.state != UNTRUSTED, (t, h)
    want = {"green": TRUSTED_GREEN, "amber": TRUSTED_AMBER, "red": TRUSTED_RED, "red_amber": TRUSTED_RED_AMBER}[sc.aspects(t)["NS"]]
    assert h.state == want, (t, h, sc.aspects(t))
    seen.add(h.state)
  assert seen == {TRUSTED_GREEN, TRUSTED_AMBER, TRUSTED_RED, TRUSTED_RED_AMBER}


def test_normal_cycle_with_cross_traffic_stopping_on_red(mon):
  # EW cars waiting at / braking gently far from their line while we are green: no behavioural trigger
  sc = SignalController()
  cars = [actor("E_W", 228.0, 0.0, "op3"), actor("W_E", 150.0, 8.0, "op4"), actor("S_N", 240.0, 10.0, "op2")]  # S_N is our phase: ignored
  assert all(h.state != UNTRUSTED for _, h in run(mon, sc.aspects, 0.0, 12.0, cars))


def test_stuck_green_detected_and_latched(mon):
  sc = SignalController()
  fault_at = 6.0
  def fn(t):
    return {"NS": "green", "EW": "green"} if t >= fault_at else sc.aspects(t)
  hist = run(mon, fn, 0.0, 30.0)
  first = next(t for t, h in hist if h.state == UNTRUSTED)
  assert fault_at <= first <= fault_at + P.conflict_debounce_s + DT + 1e-9
  assert all(h.state == UNTRUSTED and h.cause == "conflict" for t, h in hist if t >= first)
  # latched even after the lights look normal again
  assert all(h.state == UNTRUSTED for _, h in run(mon, sc.aspects, 30.0, 60.0))


@pytest.mark.parametrize("other", ["amber", "red_amber"])
def test_conflict_any_non_red(mon, other):
  hist = run(mon, lambda t: {"NS": "green", "EW": other}, 0.0, 1.0)
  assert hist[-1][1].state == UNTRUSTED and hist[-1][1].cause == "conflict"


def test_conflict_glitch_shorter_than_debounce_ignored(mon):
  def fn(t):
    return {"NS": "green", "EW": "green"} if 1.0 <= t < 1.0 + P.conflict_debounce_s - DT else {"NS": "green", "EW": "red"}
  assert all(h.state == TRUSTED_GREEN for _, h in run(mon, fn, 0.0, 3.0))


@pytest.mark.parametrize("spat", [{"NS": "dark", "EW": "red"}, {"EW": "red"}])
def test_dark_or_missing(mon, spat):
  run(mon, lambda t: {"NS": "green", "EW": "red"}, 0.0, 1.0)
  hist = run(mon, lambda t: spat, 1.0, 2.0)
  first = next(t for t, h in hist if h.state == UNTRUSTED)
  assert hist[-1][1].cause == "dark"
  assert 1.0 + P.dark_s - 1e-9 <= first <= 1.0 + P.dark_s + DT + 1e-9
  # until debounced it never invents a green from a dark head; it holds the last trusted aspect
  assert all(h.state == TRUSTED_GREEN for t, h in hist if t < first)


def test_dark_flicker_ignored(mon):
  def fn(t):
    return {"NS": "dark", "EW": "red"} if 1.0 <= t < 1.0 + P.dark_s - DT else {"NS": "red", "EW": "green"}
  assert all(h.state == TRUSTED_RED for _, h in run(mon, fn, 0.0, 3.0))


def test_dark_cross_phase_is_not_a_conflict(mon):
  assert all(h.state == TRUSTED_GREEN for _, h in run(mon, lambda t: {"NS": "green", "EW": "dark"}, 0.0, 2.0))


def test_stale(mon):
  spat = {"NS": "green", "EW": "red"}
  assert mon.update(spat, 10.0, [], 10.0 + P.stale_s - 0.01).state == TRUSTED_GREEN
  h = mon.update(spat, 10.0, [], 10.0 + P.stale_s + 0.01)
  assert h.state == UNTRUSTED and h.cause == "stale"
  assert mon.update(spat, 11.0, [], 11.0).state == UNTRUSTED  # latched


def test_stale_none(mon):
  assert mon.update(None, None, [], 1.0).cause == "stale"
  mon.reset()
  assert mon.update({"NS": "green", "EW": "red"}, None, [], 1.0).cause == "stale"


def test_behavioural_cross_car_cannot_stop(mon):
  def green(t):
    return {"NS": "green", "EW": "red"}
  v = 10.0
  d_need = v ** 2 / (2 * P.behavioural_brake) + 2.0  # 18.7 m
  # far enough to stop: fine
  assert run(mon, green, 0.0, 0.5, [actor("E_W", 229.0 - d_need - 1.0, v)])[-1][1].state == TRUSTED_GREEN
  h = mon.update(green(0.5), 0.45, [actor("E_W", 229.0 - d_need + 1.0, v)], 0.5)
  assert h.state == UNTRUSTED and h.cause == "behavioural"


def test_behavioural_car_already_in_box(mon):
  h = mon.update({"NS": "green", "EW": "red"}, 0.0, [actor("W_E", 240.0, 5.0)], 0.0)
  assert h.cause == "behavioural"


def test_behavioural_ignored_cases(mon):
  green = {"NS": "green", "EW": "red"}
  ok = [actor("E_W", 225.0, P.behavioural_v - 0.1),   # slow (creeping to its line)
        actor("W_E", 255.0, 10.0),                    # past the box: gone
        actor("S_N", 228.0, 10.0),                    # our own phase (opposite through route)
        actor("N_S", 230.0, 10.0)]
  assert mon.update(green, 0.0, ok, 0.0).state == TRUSTED_GREEN
  # not our green: a crossing car going on its own green is normal
  mon.reset()
  assert mon.update({"NS": "red", "EW": "green"}, 0.0, [actor("E_W", 225.0, 10.0)], 0.0).state == TRUSTED_RED


def test_reset_clears_latch(mon):
  run(mon, lambda t: {"NS": "green", "EW": "green"}, 0.0, 1.0)
  assert mon.update({"NS": "green", "EW": "red"}, 1.0, [], 1.0).state == UNTRUSTED
  mon.reset()
  assert mon.update({"NS": "green", "EW": "red"}, 1.0, [], 1.0).state == TRUSTED_GREEN


def test_ew_ego_phase():
  m = SignalMonitor(P, "W_E", fake_map())
  assert m.update({"NS": "green", "EW": "red"}, 0.0, [actor("N_S", 225.0, 10.0)], 0.0).state == TRUSTED_RED
  assert m.update({"NS": "red", "EW": "red_amber"}, 0.1, [], 0.1).state == TRUSTED_RED_AMBER


def test_never_reads_fault_flag():
  src = inspect.getsource(signal_monitor)
  assert "fault" not in src.replace("fault flag is never an input", "")
  assert "fault" not in inspect.signature(SignalMonitor.update).parameters


@pytest.mark.skipif(not os.path.exists(MAP), reason="crossroads map not present")
def test_real_map():
  from openpilot.tools.sim.bridge.zoomengine.crossroads import CrossroadsMap
  m = SignalMonitor(P, "N_S", CrossroadsMap(MAP))
  assert m.update({"NS": "green", "EW": "red"}, 0.0, [], 0.0).state == TRUSTED_GREEN
  assert m.update({"NS": "green", "EW": "red"}, 0.0, [actor("E_W", 220.0, 10.0)], 0.0).cause == "behavioural"
