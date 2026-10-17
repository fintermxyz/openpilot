"""regression tests for the M1 review findings (one or more per fixed behaviour). SIMULATOR DESIGN VERIFICATION ONLY"""
import math
import os
import random
import threading
import tracemalloc
from types import SimpleNamespace

import pytest

from openpilot.tools.sim.bridge.zoomengine.crossroads import CrossroadsMap
from openpilot.tools.sim.l3 import junction as J
from openpilot.tools.sim.l3.arbiter import Arbiter
from openpilot.tools.sim.l3.host import L3Host
from openpilot.tools.sim.l3.msgs import (ACTIVE, COMMIT, CONSTRAINED, GO, STOPPED, TRUSTED_GREEN, UNTRUSTED, Actor, EgoState, L3Command,
                                         SignalHealth, WorldMsg)
from openpilot.tools.sim.l3.params import PARAMS
from openpilot.tools.sim.l3.signal_monitor import SignalMonitor

MAP = os.environ.get("CROSSROADS_MAP", os.path.join(os.environ.get("ZOOMENGINE_DIR", os.path.expanduser("~/Documents/zoom/ZoomEngine-main")),
                                                    "assets/maps/crossroads.built.json"))
JP = PARAMS.junction
pytestmark = pytest.mark.skipif(not os.path.exists(MAP), reason="crossroads map not found")


@pytest.fixture(scope="module")
def cmap():
  return CrossroadsMap(MAP)


class Clock:
  def __init__(self, t: float = 1000.0):
    self.t = t

  def __call__(self) -> float:
    return self.t


def actor(cmap, aid, route, s, v, t, a=0.0, l3=None, source="oracle") -> Actor:
  x, z, psi = cmap.routes[route].track.pose(s)
  return Actor(aid, route, x, z, psi, v, s, t, a, l3=l3, source=source)


def ego_state(cmap, s, v, t, route="N_S", a=0.0) -> EgoState:
  x, z, psi = cmap.routes[route].track.pose(s)
  return EgoState(t, "op0", route, x, z, psi, v, a, s, 0.0, True)


def car_json(cmap, i, route, s, v, t, l3=None) -> dict:
  x, z, psi = cmap.routes[route].track.pose(s)
  return {"car": i, "route": route, "x": x, "z": z, "psi": psi, "v": v, "s": s, "t": t, "l3": l3}


# ---- critical: the keep-moving floor never pushes the car into a slow lead on its route ----

def test_commit_floor_dropped_for_stopped_lead(cmap):
  jm = J.JunctionManager("op0", "N_S", cmap, JP)
  jm.state = COMMIT
  sig = SignalHealth(TRUSTED_GREEN)
  e = ego_state(cmap, 232.0, 8.0, 1000.0)
  # no lead: the floor holds (overrides openpilot easing down in the box)
  assert jm.step(e, [], sig, 1000.0).a_floor is not None
  # a lead stalled in the box 20 m ahead: no floor, openpilot's braking for it wins
  lead = actor(cmap, "op9", "N_S", 232.0 + 4.6 + 15.0, 0.0, 1000.0)
  out = jm.step(e, [lead], sig, 1000.04)
  assert out.state == COMMIT and out.a_floor is None and any("lead op9" in r for r in out.reasons)
  # a lead far ahead, or moving away as fast as us: floor kept
  far = actor(cmap, "op9", "N_S", 232.0 + 60.0, 0.0, 1000.0)
  assert jm.step(e, [far], sig, 1000.08).a_floor is not None
  # a stale (unknown) lead counts as stopped
  stale = actor(cmap, "op9", "N_S", 232.0 + 4.6 + 15.0, 8.0, 1000.0, source="unknown")
  assert jm.step(e, [stale], sig, 1000.12).a_floor is None


def test_launch_floor_dropped_for_stopped_lead(cmap):
  jm = J.JunctionManager("op0", "N_S", cmap, JP)
  jm.state = GO
  e = ego_state(cmap, JP_line(cmap) - 3.3, 0.5, 1000.0)
  out = jm.step(e, [], SignalHealth(TRUSTED_GREEN), 1000.0)
  assert out.state == GO and out.a_floor is not None and out.a_floor > 0.0
  lead = actor(cmap, "op9", "N_S", e.s + 4.6 + 2.0, 0.0, 1000.0)
  out = jm.step(e, [lead], SignalHealth(TRUSTED_GREEN), 1000.04)
  assert out.a_floor is None


def JP_line(cmap) -> float:
  return cmap.routes["N_S"].stop_line_s


@pytest.mark.slow
def test_commit_rear_end_closed_loop():
  """review scenario: a non-L3 lead stalls in the box (brakes at -6 from s_stall) while the L3 follower is committed 15-30 m behind
  at 8-10 m/s. With the floor overriding openpilot's IDM braking 14/40 ran into it; openpilot alone 0/40"""
  from openpilot.tools.sim.l3.sim_kinematic import OpLike, run_episode

  class Stall(OpLike):
    stall_s = 243.0

    def accel(self, s, v, stop_line_s, junction_end, lead_gap, lead_v):
      if s >= self.stall_s:
        return -6.0
      return super().accel(s, v, stop_line_s, junction_end, lead_gap, lead_v)

  hits = []
  for stall_s in (236.0, 239.0, 242.0, 245.0, 248.0):
    for gap0 in (15.0, 20.0, 25.0, 30.0):
      for v1 in (8.0, 10.0):
        lead = Stall(v_cruise=13.41, v_junction=5.0)
        lead.stall_s = stall_s
        r = run_episode(0, "none", -1.0, True, 5, l3_mask={4}, max_t=30.0,
                        spawn={0: (150.0, 8.0, 0.0), 1: (15.0, 0.0, 29.0), 2: (15.0, 0.0, 29.0), 3: (15.0, 0.0, 29.0), 4: (150.0 - gap0, v1, 0.0)},
                        v_junction={4: 5.0}, ctrl={0: lead})
        if any("op4" in c["cars"] for c in r["collisions"]):
          hits.append((stall_s, gap0, v1))
        assert r["arb_violations"] == 0
  assert not hits, hits


# ---- major: FAILSAFE in COMMIT keeps moving through the box (host side: commit_clear_d) ----

def test_host_commit_command_carries_clear_distance(cmap):
  clock = Clock()
  host = L3Host("op0", "N_S", cmap, clock=clock)
  host.junction.state = COMMIT
  host.feed(WorldMsg(0, 0, clock.t, {"NS": "green", "EW": "red"}, []))
  cmd = host.tick(ego_state(cmap, 236.0, 4.0, clock.t))
  r = cmap.routes["N_S"]
  assert cmd.a_min_keep is not None and cmd.commit_clear_d == pytest.approx(r.junction_s[1] + 2.3 + 1.0 - 236.0)
  # the tick dies: FAILSAFE, yet the car keeps >= ~v_clear until its rear is out of the box
  host.arbiter.fail("tick raised")
  v, s, t = 4.0, 236.0, clock.t
  while s - 2.3 < r.junction_s[1]:
    t += 0.01
    out = host.arbiter.apply(-1.0, t, 0.01, v=v)
    assert out.failsafe
    v = max(0.0, v + out.accel * 0.01)
    s += v * 0.01
    assert v > 2.5, (s, v)


# ---- major: conflicting greens that flicker faster than the debounce are latched ----

@pytest.mark.parametrize("period,on", [(2, 1), (4, 1), (4, 3), (6, 2)])
def test_flickering_conflict_latched(period, on):
  routes = {n: SimpleNamespace(phase=ph, stop_line_s=229.0, junction_s=(231.35, 250.65))
            for n, ph in (("N_S", "NS"), ("S_N", "NS"), ("E_W", "EW"), ("W_E", "EW"))}
  mon = SignalMonitor(PARAMS.signal, "N_S", SimpleNamespace(routes=routes))
  first = None
  for k in range(200):  # 20 Hz messages, ticks at 25 Hz see the latest
    t = k * 0.05
    spat = {"NS": "green", "EW": "green" if (k % period) < on else "red"}
    for j in range(2):
      h = mon.update(spat, t, [], t + 0.02 * j)
      if h.state == UNTRUSTED and first is None:
        first = t
        assert h.cause == "conflict"
  assert first is not None and first <= 1.0, first


def test_single_short_glitch_still_ignored():
  routes = {n: SimpleNamespace(phase=ph, stop_line_s=229.0, junction_s=(231.35, 250.65))
            for n, ph in (("N_S", "NS"), ("S_N", "NS"), ("E_W", "EW"), ("W_E", "EW"))}
  mon = SignalMonitor(PARAMS.signal, "N_S", SimpleNamespace(routes=routes))
  for k in range(200):
    t = k * 0.05
    spat = {"NS": "green", "EW": "green" if k in (40, 41) else "red"}  # one 0.1 s glitch
    assert mon.update(spat, t, [], t).state == TRUSTED_GREEN
  # another one more than the window later: still fine; two close together latch
  for k in range(200, 260):
    t = k * 0.05
    h = mon.update({"NS": "green", "EW": "green" if k in (230, 231, 236) else "red"}, t, [], t)
  assert h.state == UNTRUSTED and h.cause == "conflict"


# ---- major: a world message of another trial (restarted director) never empties the junction ----

def test_wrong_trial_world_message_ignored_and_cars_never_vanish(cmap):
  clock = Clock()
  host = L3Host("op0", "N_S", cmap, clock=clock)
  host.reset(trial=5)
  r_e, r_o = cmap.routes["N_S"], cmap.routes["E_W"]
  coop_first = {"mode": CONSTRAINED, "jx_state": STOPPED, "t_stop": 990.0, "token": False}
  s_e, s_o = r_e.stop_line_s - 3.3, r_o.stop_line_s - 3.3
  for k in range(100):  # both stopped at their lines, op2 first in the queue
    host.feed(WorldMsg(5, k, clock.t, {"NS": "green", "EW": "green"}, [WorldMsg.from_json({"trial": 5, "seq": 0, "t": clock.t, "spat": {},
              "cars": [car_json(cmap, 2, "E_W", s_o, 0.0, clock.t, coop_first)]}).cars[0]]))
    cmd = host.tick(ego_state(cmap, s_e, 0.0, clock.t))
    clock.t += 0.04
  assert host.status()["jx_state"] == STOPPED and not host.status()["token"]
  # the director restarts: trial -1, no cars, for seconds
  for k in range(150):
    host.feed(WorldMsg(-1, k, clock.t, {"NS": "green", "EW": "red"}, []))
    cmd = host.tick(ego_state(cmap, s_e, 0.0, clock.t))
    assert host.status()["jx_state"] != GO and not host.status()["token"] and cmd.a_min_keep is None
    clock.t += 0.04
  assert host.wrong_trial_msgs == 150
  # same trial, but op2 drops out of the list: it is kept as unknown (worst case), never gone
  host2 = L3Host("op0", "N_S", cmap, clock=clock)
  host2.feed(WorldMsg(0, 0, clock.t, {"NS": "green", "EW": "green"}, [WorldMsg.from_json({"trial": 0, "seq": 0, "t": clock.t, "spat": {},
             "cars": [car_json(cmap, 2, "E_W", s_o, 0.0, clock.t, coop_first)]}).cars[0]]))
  host2.tick(ego_state(cmap, s_e, 0.0, clock.t))
  clock.t += 1.0
  host2.feed(WorldMsg(0, 1, clock.t, {"NS": "green", "EW": "green"}, []))
  host2.tick(ego_state(cmap, s_e, 0.0, clock.t))
  acts = host2.last_actors
  assert [a.id for a in acts] == ["op2"] and acts[0].source == "unknown"
  # a car last seen through the box can never come back: dropped
  host3 = L3Host("op0", "N_S", cmap, clock=clock)
  host3.feed(WorldMsg(0, 0, clock.t, {"NS": "green", "EW": "red"}, [WorldMsg.from_json({"trial": 0, "seq": 0, "t": clock.t, "spat": {},
             "cars": [car_json(cmap, 2, "E_W", r_o.junction_s[1] + 10.0, 8.0, clock.t)]}).cars[0]]))
  host3.tick(ego_state(cmap, 100.0, 8.0, clock.t))
  clock.t += 1.0
  host3.feed(WorldMsg(0, 1, clock.t, {"NS": "green", "EW": "red"}, []))
  host3.tick(ego_state(cmap, 100.0, 8.0, clock.t))
  assert host3.last_actors == []


# ---- minor: reset() serialized with apply() ----

def test_reset_never_leaves_failsafe_latched_under_concurrency():
  arb = Arbiter(PARAMS.arbiter)
  stop = threading.Event()
  t = [0.0]

  def control():
    while not stop.is_set():
      t[0] += 0.001
      arb.set_command(L3Command(t_mono=t[0], seq=1, mode=ACTIVE))
      arb.apply(0.5, t[0], 0.001)

  th = threading.Thread(target=control)
  th.start()
  try:
    rng = random.Random(0)
    for _ in range(2000):
      arb.fail("tick raised")
      if rng.random() < 0.5:
        threading.Event().wait(0.0)
      arb.reset()
      with arb._lock:  # right after reset nothing latched remains (no stale snapshot can re-latch it)
        assert not arb.failsafe and arb._fail_reason is None
  finally:
    stop.set()
    th.join()


# ---- minor: geometry build is cheap ----

def test_geometry_build_cheap_and_identical(cmap):
  g = J.ConflictGeometry(cmap, JP)
  cg = J.ConflictGeometry.__new__(J.ConflictGeometry)
  cg.half_w, cg.half_l = g.half_w, g.half_l
  tracemalloc.start()
  built = cg._build(cmap)
  peak = tracemalloc.get_traced_memory()[1]
  tracemalloc.stop()
  assert built == g.conflicts
  assert peak < 30e6, peak


# ---- minor: arbiter counters for the invariant (engine runs) ----

def test_arbiter_counts_invariant_and_stale():
  arb = Arbiter(PARAMS.arbiter)
  arb.set_command(L3Command(t_mono=0.0, seq=1, mode=CONSTRAINED, a_cap=-1.0))
  for k in range(10):
    arb.apply(0.5, 0.01 * k, 0.01)
  assert arb.stats["ticks"] == 10 and arb.stats["l3_ticks"] == 10 and arb.stats["violations"] == 0
  arb.apply(0.5, 0.5, 0.01)  # stale
  assert arb.stats["stale_events"] == 1 and arb.stats["max_stale_s"] == pytest.approx(0.5)
  assert math.isfinite(arb.stats["max_excess"])


# ---- director: trial length, per-car crossing time, t_clear and timeout ----

def test_director_records_clear_times_and_timeouts(cmap, tmp_path):
  from openpilot.tools.sim.bridge.zoomengine import crossroads_director as cd
  d = cd.Director.__new__(cd.Director)
  d.cmap, d.cars, d.trial, d.trial_len, d.out, d.results = cmap, 2, 0, 90.0, str(tmp_path / "res.json"), []
  d.signals, d.shown_phases, d.still_since, d.poses = SimpleNamespace(fault=True), {"NS": "green", "EW": "green"}, None, {}
  d.outcome = {"trial": 0, "collisions": [], "cars": {}, "deadlock": False, "deadlock_at": None, "fault_t": 6.0}
  d.sc = cd.SCENARIOS["stuck_green"]
  import time as _time

  def pose(c, route, s, v):
    x, z, psi = cmap.routes[route].track.pose(s)
    return {"car": c, "route": route, "trial": 0, "x": x, "z": z, "psi": psi, "v": v, "s": s, "engaged": True, "parked": False,
            "t": _time.monotonic(), "l3_stats": {"violations": 0}}
  j1 = cmap.routes["N_S"].junction_s[1]
  for k, s in enumerate((200.0, 240.0, j1 + 1.0, j1 + 3.0)):
    d.poses = {0: pose(0, "N_S", s, 5.0), 1: pose(1, "E_W", 100.0, 0.0)}
    d.judge(10.0 + k)
  o = d.outcome["cars"]
  assert o[0]["crossed"] and o[0]["crossed_at"] == 13.0 and o[1]["crossed_at"] is None and o[0]["l3_stats"] == {"violations": 0}
  d.finish_trial()
  assert d.results[-1]["timeout"] and d.results[-1]["t_clear"] is None


def test_crossroads_car_l3_hud_per_car():
  from openpilot.tools.sim.bridge.zoomengine.crossroads_car import CrossroadsCarWorld
  w = CrossroadsCarWorld.__new__(CrossroadsCarWorld)
  w.car, w.OVERLAY_PREFIX = 2, "op2_"
  w.l3 = SimpleNamespace(hud_lines=lambda: [("L3 FAILSAFE   junction --", [1.0, 0.2, 0.2]), ("L3 FAILSAFE: watchdog", [1.0, 0.2, 0.2])])
  cmds = w._l3_hud_commands()
  assert len(cmds) == 1 and cmds[0]["id"] == "op2_l3" and cmds[0]["action"] == "hud_text"
  assert cmds[0]["text"].startswith("car 3") and "watchdog" in cmds[0]["text"]
