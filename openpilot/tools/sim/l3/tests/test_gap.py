"""conflict geometry and the PET gap test: whenever the gap test admits, no trajectory inside the envelopes violates pet_min"""
import math
import os

import numpy as np
import pytest

from openpilot.tools.sim.bridge.zoomengine.crossroads import CrossroadsMap, overlap
from openpilot.tools.sim.l3.junction import (COMMIT_BRAKE_MAX, DECEL_A, OTHER_BRAKE_MAX, STOPPED_V, YIELD_SUSTAIN_S, ConflictGeometry,
                                             JunctionManager)
from openpilot.tools.sim.l3.msgs import ACTIVE, STOPPED, TRUSTED_GREEN, UNTRUSTED, Actor, SignalHealth
from openpilot.tools.sim.l3.params import PARAMS

MAP_PATH = os.environ.get("CROSSROADS_MAP", os.path.join(os.environ.get("ZOOMENGINE_DIR", os.path.expanduser("~/Documents/zoom/ZoomEngine-main")),
                                                         "assets/maps/crossroads.built.json"))
JP = PARAMS.junction
N_SAMPLES = 100_000
DT = 0.02
T_END = 40.0


@pytest.fixture(scope="module")
def cmap():
  return CrossroadsMap(MAP_PATH)


@pytest.fixture(scope="module")
def geom(cmap):
  return ConflictGeometry(cmap, JP)


def rect(cmap, route: str, s: float, half_w: float, half_l: float) -> np.ndarray:
  x, z, psi = cmap.routes[route].track.pose(s)
  f = np.array([math.sin(psi), math.cos(psi)])
  r = np.array([-math.cos(psi), math.sin(psi)])
  c = np.array([x, z])
  return np.array([c + f * half_l + r * half_w, c + f * half_l - r * half_w, c - f * half_l - r * half_w, c - f * half_l + r * half_w])


def actor(cmap, aid: str, route: str, s: float, v: float, a: float = 0.0, l3=None, source: str = "oracle") -> Actor:
  x, z, psi = cmap.routes[route].track.pose(s)
  return Actor(aid, route, x, z, psi, v, s, 1000.0, a, l3=l3, source=source)


# ---- geometry ----

def test_geometry_pairs(cmap, geom):
  c = geom.conflicts
  for a, b in (("N_S", "E_W"), ("N_S", "W_E"), ("S_N", "E_W"), ("S_N", "W_E")):
    assert b in c[a] and a in c[b]
    cin_a, cout_a, cin_b, cout_b = c[a][b]
    assert c[b][a] == (cin_b, cout_b, cin_a, cout_a)
    for (cin, cout), r in (((cin_a, cout_a), cmap.routes[a]), ((cin_b, cout_b), cmap.routes[b])):
      assert r.junction_s[0] <= cin < cout <= r.junction_s[1], (a, b, cin, cout, r.junction_s)
      assert r.stop_line_s < cin
  for a, b in (("N_S", "S_N"), ("E_W", "W_E")):
    assert b not in c[a] and a not in c[b]
    assert not geom.conflicting(a, b)


def test_geometry_brute_force(cmap, geom):
  """any overlap of the swept rectangles (independent SAT) lies inside both intervals"""
  rng = np.random.default_rng(1)
  hw, hl = geom.half_w, geom.half_l
  for a, b in (("N_S", "E_W"), ("N_S", "W_E"), ("E_W", "S_N")):
    cin_a, cout_a, cin_b, cout_b = geom.conflicts[a][b]
    for _ in range(4000):
      sa, sb = rng.uniform(225.0, 255.0, 2)
      if overlap(rect(cmap, a, sa, hw, hl), rect(cmap, b, sb, hw, hl)):
        assert cin_a <= sa <= cout_a and cin_b <= sb <= cout_b, (a, b, sa, sb)
    # and the intervals are tight: the middle of both overlaps
    assert overlap(rect(cmap, a, (cin_a + cout_a) / 2, hw, hl), rect(cmap, b, (cin_b + cout_b) / 2, hw, hl))
  for a, b in (("N_S", "S_N"), ("E_W", "W_E")):
    for sa in np.arange(200.0, 280.0, 0.5):
      assert not overlap(rect(cmap, a, sa, hw, hl), rect(cmap, b, 482.0 - sa, hw, hl))


# ---- gap test property ----

def simulate(jm: JunctionManager, s_e, v_e, ego_kind, cin_e, cout_e, s_o, v_o, a_o, other_kind, cin_o, cout_o, v_max_o, rng):
  """vectorised forward simulation; returns first / last occupancy times of each car (nan if never inside)"""
  n = len(s_e)
  p, prof = JP, jm.prof
  rho_o = np.where(v_o < STOPPED_V, p.rho_other_stopped, p.rho_other)
  a0_o = np.clip(a_o, 0.0, p.a_other_max)
  v_hi_e = np.maximum(prof.v_cross, v_e)
  first_e, last_e = np.full(n, np.nan), np.full(n, np.nan)
  first_o, last_o = np.full(n, np.nan), np.full(n, np.nan)
  a_rand_e = np.zeros(n)
  a_rand_o = np.zeros(n)
  # other: per sample a "style": 0 random, 1 always max accel, 2 brake hard then relaunch at max, 3 cruise
  steps = int(T_END / DT)
  for k in range(steps + 1):
    t = k * DT
    ins_e = (s_e >= cin_e) & (s_e <= cout_e)
    ins_o = (s_o >= cin_o) & (s_o <= cout_o)
    first_e = np.where(ins_e & np.isnan(first_e), t, first_e)
    last_e = np.where(ins_e, t, last_e)
    first_o = np.where(ins_o & np.isnan(first_o), t, first_o)
    last_o = np.where(ins_o, t, last_o)
    if k % 25 == 0:
      a_rand_e = rng.uniform(0.0, 1.0, n)
      a_rand_o = rng.uniform(0.0, 1.0, n)
    # ego: 0 planned (fast), 1 slow (COMMIT floor), 2 random between the floor and the planned accel
    hi_e = np.where(v_e < v_hi_e - 1e-9, p.a_go, 0.0)
    lo_e = np.minimum(p.a_go, np.maximum(-COMMIT_BRAKE_MAX, prof.k * (p.v_clear - v_e)))
    lo_e = np.minimum(lo_e, hi_e)
    a_e = np.where(ego_kind == 0, hi_e, np.where(ego_kind == 1, lo_e, lo_e + a_rand_e * (hi_e - lo_e)))
    hi_o = np.where(t < rho_o, a0_o, p.a_other_max)
    a_o_t = np.select([other_kind == 1, other_kind == 2, other_kind == 3],
                      [hi_o, np.where(t < 1.5, -OTHER_BRAKE_MAX, hi_o), np.minimum(0.0, hi_o)],
                      -OTHER_BRAKE_MAX + a_rand_o * (hi_o + OTHER_BRAKE_MAX))
    # exact step with speed limits
    for s_, v_, a_, vmax in ((s_e, v_e, a_e, v_hi_e), (s_o, v_o, a_o_t, v_max_o)):
      v1 = np.clip(v_ + a_ * DT, 0.0, vmax)
      s_ += 0.5 * (v_ + v1) * DT
      v_[:] = v1
  return first_e, last_e, first_o, last_o


def test_gap_property(cmap, geom):
  rng = np.random.default_rng(20261017)
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  sig = SignalHealth(UNTRUSTED, "conflict")
  r_e = cmap.routes["N_S"]
  v_max_lim = JP.v_other_factor * cmap.speed_limit
  rows = []
  for _ in range(N_SAMPLES):
    route_o = "E_W" if rng.random() < 0.5 else "W_E"
    cin_e, cout_e, cin_o, cout_o = geom.conflicts["N_S"][route_o]
    s_e = rng.uniform(r_e.stop_line_s - 2.3 - 25.0, cout_e)
    v_e = rng.choice([0.0, rng.uniform(0.0, 13.4)])
    s_o = rng.uniform(cin_o - 70.0, cout_o + 1.0)
    v_o = rng.choice([0.0, rng.uniform(0.0, v_max_lim)])
    a_o = rng.uniform(-DECEL_A, JP.a_other_max)   # not "already decelerating": that is a trust rule, tested separately
    ok, _ = jm.gap_test("N_S", s_e, v_e, [actor(cmap, "op2", route_o, s_o, v_o, a_o)], sig, JP.pet_min)
    if ok:
      rows.append((s_e, v_e, cin_e, cout_e, s_o, v_o, a_o, cin_o, cout_o))
  rows = np.array(rows)
  assert len(rows) > 0.05 * N_SAMPLES, len(rows)  # the property is not vacuous
  assert len(rows) < 0.95 * N_SAMPLES
  s_e, v_e, cin_e, cout_e, s_o, v_o, a_o, cin_o, cout_o = rows.T.copy()
  n = len(rows)
  ego_kind = rng.integers(0, 3, n)
  other_kind = rng.integers(0, 4, n)
  v_max_o = np.maximum(v_o, v_max_lim)
  fe, le, fo, lo = simulate(jm, s_e.copy(), v_e.copy(), ego_kind, cin_e, cout_e, s_o.copy(), v_o.copy(), a_o, other_kind, cin_o, cout_o,
                            v_max_o, rng)
  both = ~np.isnan(fe) & ~np.isnan(fo)
  tol = 3 * DT
  sep = (le + JP.pet_min - tol <= fo) | (lo + JP.pet_min - tol <= fe)
  bad = np.nonzero(both & ~sep)[0]
  assert len(bad) == 0, [dict(zip(("s_e", "v_e", "s_o", "v_o", "a_o", "ego", "other", "fe", "le", "fo", "lo"),
                                  (s_e[i], v_e[i], s_o[i], v_o[i], a_o[i], ego_kind[i], other_kind[i], fe[i], le[i], fo[i], lo[i]), strict=False))
                             for i in bad[:5]]
  # a meaningful share of the admitted samples had both cars through the conflict, closely
  assert both.sum() > 1000



# ---- trust rules and stale actors ----

def test_cooperative_yield(cmap, geom):
  """a cooperative car stopped at its line without the token is stationary; the same car non-cooperative may launch"""
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  r_e, r_o = cmap.routes["N_S"], cmap.routes["E_W"]
  sig = SignalHealth(UNTRUSTED, "conflict")
  s_e, s_o = r_e.stop_line_s - 3.3, r_o.stop_line_s - 3.3
  coop = {"mode": ACTIVE, "jx_state": STOPPED, "t_stop": 1001.0, "token": False}
  assert jm.gap_test("N_S", s_e, 0.0, [actor(cmap, "op2", "E_W", s_o, 0.0, l3=coop)], sig, JP.pet_min)[0]
  assert not jm.gap_test("N_S", s_e, 0.0, [actor(cmap, "op2", "E_W", s_o, 0.0)], sig, JP.pet_min)[0]
  assert not jm.gap_test("N_S", s_e, 0.0, [actor(cmap, "op2", "E_W", s_o, 0.0, l3=dict(coop, token=True))], sig, JP.pet_min)[0]
  # trusted green: the cross phase is red, a car stopped at its line waits for it
  assert jm.gap_test("N_S", s_e, 0.0, [actor(cmap, "op2", "E_W", s_o, 0.0)], SignalHealth(TRUSTED_GREEN), JP.pet_min)[0]
  # braking at least as hard as stopping at its line needs (<= a_other_brake), seen for YIELD_SUSTAIN_S: will not enter
  braking = actor(cmap, "op2", "E_W", s_o - 30.0, 8.0, -2.0)
  jm.observe([braking], 1000.0)
  assert not jm.gap_test("N_S", s_e, 0.0, [braking], sig, JP.pet_min)[0]  # not sustained yet
  jm.observe([braking], 1000.0 + YIELD_SUSTAIN_S)
  assert jm.gap_test("N_S", s_e, 0.0, [braking], sig, JP.pet_min)[0]
  assert not jm.gap_test("N_S", s_e, 0.0, [actor(cmap, "op2", "E_W", s_o - 30.0, 8.0, 0.0)], sig, JP.pet_min)[0]


def test_easing_down_is_not_yielding(cmap, geom):
  """review finding: openpilot easing from 13 m/s towards a 2.5 m/s crossing speed (a ~ -1, v^2/2d <= 4) was taken as yielding and
  the ego got GO, saved only by a 6 m/s^2 emergency stop. Its projected stop point is past its line: not yielding, however long"""
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  r_e = cmap.routes["N_S"]
  sig = SignalHealth(UNTRUSTED, "conflict")
  s_e = r_e.stop_line_s - 3.3
  x = actor(cmap, "op2", "E_W", 185.3, 13.0, -1.05)  # the traced case: ~41 m from its line, needs ~2.0 m/s^2 to stop there
  for k in range(30):
    jm.observe([x], 1000.0 + 0.04 * k)
    assert not jm.yielding(x, sig)
  assert not jm.gap_test("N_S", s_e, 0.0, [x], sig, JP.pet_min)[0]
  # the same car braking for its line (a <= -v^2/2d) for YIELD_SUSTAIN_S does yield
  y = actor(cmap, "op2", "E_W", 185.3, 13.0, -2.2)
  for k in range(30):
    jm.observe([y], 1000.0 + 0.04 * k)
  assert jm.yielding(y, sig)


def test_unknown_actor_never_gone(cmap, geom):
  """a stale actor: worst-case arrival, never assumed gone, never trusted to yield"""
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  r_e, r_o = cmap.routes["N_S"], cmap.routes["E_W"]
  sig = SignalHealth(UNTRUSTED, "conflict")
  s_e = r_e.stop_line_s - 3.3
  coop = {"mode": ACTIVE, "jx_state": STOPPED, "t_stop": 1001.0, "token": False}
  for s_o, v_o in ((r_o.stop_line_s - 3.3, 0.0), (r_o.junction_s[1] + 30.0, 10.0), (r_o.stop_line_s - 60.0, 13.0)):
    assert not jm.gap_test("N_S", s_e, 0.0, [actor(cmap, "op2", "E_W", s_o, v_o, l3=coop, source="unknown")], sig, JP.pet_min)[0]
  # the same car known and past its zone is gone
  assert jm.gap_test("N_S", s_e, 0.0, [actor(cmap, "op2", "E_W", r_o.junction_s[1] + 30.0, 10.0)], sig, JP.pet_min)[0]


def test_far_car_admits_and_close_car_blocks(cmap, geom):
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  r_e, r_o = cmap.routes["N_S"], cmap.routes["E_W"]
  sig = SignalHealth(UNTRUSTED, "conflict")
  s_e = r_e.stop_line_s - 3.3
  assert jm.gap_test("N_S", s_e, 0.0, [actor(cmap, "op2", "E_W", r_o.stop_line_s - 200.0, 13.4)], sig, JP.pet_min)[0]
  assert not jm.gap_test("N_S", s_e, 0.0, [actor(cmap, "op2", "E_W", r_o.stop_line_s - 30.0, 13.4)], sig, JP.pet_min)[0]
  # opposite route never conflicts
  assert jm.gap_test("N_S", s_e, 0.0, [actor(cmap, "op1", "S_N", r_e.stop_line_s - 3.3, 13.4)], sig, JP.pet_min)[0]
