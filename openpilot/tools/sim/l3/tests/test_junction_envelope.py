"""approach envelope: until GO the commanded profile can always stop before the stop line with <= a_b (closed loop, first-order lag rho)"""
import math
import os

import numpy as np
import pytest

from openpilot.tools.sim.bridge.zoomengine.crossroads import CrossroadsMap
from openpilot.tools.sim.l3.junction import LIGHT_STOP_SLACK, ConflictGeometry, JunctionManager
from openpilot.tools.sim.l3.msgs import (APPROACH, CLEARED, COMMIT, FAR, GO, STOP_AT_LINE, STOPPED, TRUSTED_GREEN, TRUSTED_RED,
                                         UNTRUSTED, EgoState, SignalHealth)
from openpilot.tools.sim.l3.params import PARAMS

MAP_PATH = os.environ.get("CROSSROADS_MAP", os.path.join(os.environ.get("ZOOMENGINE_DIR", os.path.expanduser("~/Documents/zoom/ZoomEngine-main")),
                                                         "assets/maps/crossroads.built.json"))
JP = PARAMS.junction
DT = 0.04  # 25 Hz tick
SUB = 4    # physics substeps per tick
# openpilot: flooring it, and cruising towards 8 m/s
OPS = (lambda v: 2.0, lambda v: min(1.0, max(-1.0, 0.5 * (8.0 - v))))


@pytest.fixture(scope="module")
def cmap():
  return CrossroadsMap(MAP_PATH)


@pytest.fixture(scope="module")
def geom(cmap):
  return ConflictGeometry(cmap, JP)


def ego_at(cmap, route: str, s: float, v: float, a: float = 0.0, t: float = 0.0) -> EgoState:
  x, z, psi = cmap.routes[route].track.pose(s)
  return EgoState(t, "op0", route, x, z, psi, v, a, s, 0.0, True)


def run_stop(cmap, geom, v0: float, d0: float, a_op, signal: SignalHealth, a0: float = 0.0, t_max: float = 60.0):
  """closed loop: arbiter min(a_op(v), a_cap) (floor where asked), actual accel follows the command with a first-order lag rho.
  returns (min commanded accel while not GO, min d_line, states seen, final v)"""
  r = cmap.routes["N_S"]
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  s, v, a = r.stop_line_s - 2.3 - d0, v0, a0
  t, a_min, d_min, states = 1000.0, math.inf, math.inf, set()
  while t < 1000.0 + t_max:
    jo = jm.step(ego_at(cmap, "N_S", s, v, a, t), [], signal, t)
    states.add(jo.state)
    cmd = min(a_op(v), jo.a_cap)
    if jo.a_floor is not None:
      cmd = max(cmd, jo.a_floor)
    if jo.state != GO and jo.state != COMMIT:
      a_min = min(a_min, cmd)
    for _ in range(SUB):
      h = DT / SUB
      if jo.brake_hold:
        a = -3.0 if v > 0 else 0.0
      else:
        a += (cmd - a) * h / JP.rho
      v = max(v + a * h, 0.0)
      s += v * h
      d_min = min(d_min, r.stop_line_s - (s + 2.3))
    t += DT
    if jo.state == STOPPED and v == 0.0:
      break
  return a_min, d_min, states, v


def test_envelope_grid(cmap, geom):
  """v in [0, 13.4] x d_line in [0, 80]: from every state the envelope admits (it is entered at 80 m, where it admits any speed
  up to the limit), with openpilot flooring it or cruising, the car stops before the line, never commanded below -a_b"""
  red = SignalHealth(TRUSTED_RED)
  checked = 0
  for v0 in np.linspace(0.0, 13.4, 28):
    for d0 in np.linspace(0.0, 80.0, 81):
      inside = v0 * JP.rho + v0 * v0 / (2 * JP.a_b) <= d0 - JP.line_margin
      if not inside:
        continue
      for a_op in OPS:
        a_min, d_min, states, v = run_stop(cmap, geom, v0, d0, a_op, red)
        assert GO not in states and COMMIT not in states, (v0, d0, states)
        assert d_min >= 0.0, f"overran the line by {-d_min:.2f} m from v {v0:.2f} d {d0:.1f}"
        assert a_min >= -JP.a_b - 1e-9, (v0, d0, a_min)
        assert v == 0.0
        checked += 1
  assert checked > 1500


def test_envelope_from_approach_start(cmap, geom):
  """any speed up to the limit at the approach start: always inside the envelope"""
  for v0 in np.linspace(0.0, cmap.speed_limit, 15):
    assert v0 * JP.rho + v0 * v0 / (2 * JP.a_b) <= JP.approach_d - JP.line_margin
    a_min, d_min, states, v = run_stop(cmap, geom, v0, JP.approach_d - 1e-3, OPS[0], SignalHealth(TRUSTED_RED), a0=2.0)
    assert d_min >= 0.0 and a_min >= -JP.a_b - 1e-9 and v == 0.0


def test_untrusted_also_stops(cmap, geom):
  """failed signal, nobody about: mandatory stop at the line first, then (token, empty junction) through"""
  r = cmap.routes["N_S"]
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  s, v, t = r.stop_line_s - 2.3 - 60.0, 12.0, 1000.0
  states, d_min_before_go = [], math.inf
  sig = SignalHealth(UNTRUSTED, "conflict")
  while t < 1060.0:
    jo = jm.step(ego_at(cmap, "N_S", s, v, 0.0, t), [], sig, t)
    if not states or states[-1] != jo.state:
      states.append(jo.state)
    cmd = min(1.0, jo.a_cap) if not jo.brake_hold else -3.0
    if jo.a_floor is not None and not jo.brake_hold:
      cmd = max(cmd, jo.a_floor)
    v = max(v + cmd * DT, 0.0)
    s += v * DT
    if jo.state not in (GO, COMMIT, CLEARED):
      d_min_before_go = min(d_min_before_go, jo.d_line)
    t += DT
    if jo.state == CLEARED:
      break
  assert states[:2] == [STOP_AT_LINE, STOPPED], states
  assert CLEARED in states and COMMIT in states, states
  assert d_min_before_go >= 0.0
  assert jm.t_stop is not None


def test_far_and_cleared_unconstrained(cmap, geom):
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  r = cmap.routes["N_S"]
  jo = jm.step(ego_at(cmap, "N_S", r.stop_line_s - 2.3 - 100.0, 13.0), [], SignalHealth(TRUSTED_RED), 1000.0)
  assert jo.state == FAR and math.isinf(jo.a_cap) and jo.a_floor is None
  jo = jm.step(ego_at(cmap, "N_S", r.junction_s[1] + 5.0, 8.0), [], SignalHealth(TRUSTED_RED), 1001.0)
  assert jo.state == CLEARED and math.isinf(jo.a_cap) and jo.a_floor is None


def test_green_empty_junction_goes(cmap, geom):
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  r = cmap.routes["N_S"]
  jo = jm.step(ego_at(cmap, "N_S", r.stop_line_s - 2.3 - 50.0, 10.0), [], SignalHealth(TRUSTED_GREEN), 1000.0)
  assert jo.state == GO and jo.a_cap >= 0.0


def test_point_of_no_return(cmap, geom):
  """GO lost past the point where a_b can stop at the line: emergency stop if it fits before the conflict zone, else commit"""
  r = cmap.routes["N_S"]
  green, red = SignalHealth(TRUSTED_GREEN), SignalHealth(UNTRUSTED, "conflict")
  # 8 m/s, 6 m from the line: a_b can't stop at the line, a_em can before cin - 1
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  s = r.stop_line_s - 2.3 - 6.0
  assert jm.step(ego_at(cmap, "N_S", s, 8.0), [], green, 1000.0).state == GO
  jo = jm.step(ego_at(cmap, "N_S", s + 0.3, 8.0), [], red, 1000.04)
  assert jo.em and jo.a_floor == -JP.a_em and jo.a_cap == -JP.a_em
  # 12 m/s, 2 m from the line: nothing stops it before the conflict zone: commit, keep moving
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  s = r.stop_line_s - 2.3 - 2.0
  jm.step(ego_at(cmap, "N_S", s, 12.0), [], green, 1000.0)
  jo = jm.step(ego_at(cmap, "N_S", s + 0.5, 12.0), [], red, 1000.04)
  assert jo.state == COMMIT and not jo.em
  for v in (0.0, 1.0, 2.9):
    jo = jm.step(ego_at(cmap, "N_S", s + 5.0, v), [], red, 1000.1)
    assert jo.state == COMMIT and jo.a_floor is not None and jo.a_floor > 0.0
  # GO lost while the a_b stop is still feasible: back to stopping
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  s = r.stop_line_s - 2.3 - 40.0
  assert jm.step(ego_at(cmap, "N_S", s, 8.0), [], green, 1000.0).state == GO
  jo = jm.step(ego_at(cmap, "N_S", s + 0.3, 8.0), [], red, 1000.04)
  assert jo.state == STOP_AT_LINE and not jo.em and jo.a_cap >= -JP.a_b


def test_amber_rule(cmap, geom):
  """rule 175: on amber stop unless stopping would need more than amber_stop_a"""
  r = cmap.routes["N_S"]
  amber = SignalHealth("TRUSTED_AMBER")
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  jo = jm.step(ego_at(cmap, "N_S", r.stop_line_s - 2.3 - 40.0, 10.0), [], amber, 1000.0)
  assert jo.state in (APPROACH, STOP_AT_LINE) and jo.state != GO
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  jo = jm.step(ego_at(cmap, "N_S", r.stop_line_s - 2.3 - 8.0, 10.0), [], amber, 1000.0)
  assert jo.state in (GO, COMMIT)


@pytest.mark.parametrize("v,d,a", [(8.0, 13.43, -0.14), (8.55, 16.61, -0.97), (2.8, 1.6, 0.91), (12.0, 30.87, -0.24), (5.5, -3.4, -2.1)])
def test_light_change_while_going_is_never_an_emergency(cmap, geom, v, d, a):
  """GO on green, then a normal amber -> red change (trusted): stop at the line if amber_stop_a still can (rule 175), else keep
  going; the stop then brakes at most amber_stop_a + LIGHT_STOP_SLACK. Never EM
  (states taken from 8-car kinematic episodes where the lag-free amber rule disagreed with the STOP profile's feasibility)"""
  r = cmap.routes["N_S"]
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  s = r.stop_line_s - 2.3 - d
  assert jm.step(ego_at(cmap, "N_S", s, v, a), [], SignalHealth(TRUSTED_GREEN), 1000.0).state in (GO, COMMIT)
  stop_ok = d > 0.0 and jm.stop_feasible(v, a, d, JP.amber_stop_a)
  for i, sig in enumerate(("TRUSTED_AMBER", TRUSTED_RED)):
    jo = jm.step(ego_at(cmap, "N_S", s, v, a), [], SignalHealth(sig), 1000.04 * (i + 2))
    assert not jo.em
    assert jo.state == (STOP_AT_LINE if stop_ok else jo.state) and (stop_ok or jo.state in (GO, COMMIT))
    if stop_ok:
      assert jo.a_cap >= -(JP.amber_stop_a + LIGHT_STOP_SLACK)
      break
