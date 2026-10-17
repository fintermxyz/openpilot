"""unmarked (failed-signal) crossroads: 4 cooperative cars, every arrival order and exact ties. Small closed loop of 4
JunctionManagers exchanging their l3 dicts with one tick of latency"""
import itertools
import os

import pytest

from openpilot.tools.sim.bridge.zoomengine.crossroads import CrossroadsMap
from openpilot.tools.sim.l3.junction import TOKEN_SETTLE_S, ConflictGeometry, JunctionManager
from openpilot.tools.sim.l3.msgs import ACTIVE, CLEARED, COMMIT, GO, STOPPED, UNTRUSTED, Actor, EgoState, SignalHealth
from openpilot.tools.sim.l3.params import PARAMS

MAP_PATH = os.environ.get("CROSSROADS_MAP", os.path.join(os.environ.get("ZOOMENGINE_DIR", os.path.expanduser("~/Documents/zoom/ZoomEngine-main")),
                                                         "assets/maps/crossroads.built.json"))
JP = PARAMS.junction
ROUTES = ("N_S", "S_N", "E_W", "W_E")
DT = 0.04
SUB = 4
TAU = 0.3
V0 = 8.0
D0 = 40.0
T_MAX = 90.0


@pytest.fixture(scope="module")
def cmap():
  return CrossroadsMap(MAP_PATH)


@pytest.fixture(scope="module")
def geom(cmap):
  return ConflictGeometry(cmap, JP)


class Car:
  def __init__(self, cid: str, route: str, cmap, geom, delay: float):
    self.id, self.route = cid, route
    self.r = cmap.routes[route]
    self.jm = JunctionManager(cid, route, cmap, JP, geom)
    # identical approaches shifted in time by `delay`: same speed, further back
    self.s = self.r.stop_line_s - 2.3 - D0 - V0 * delay
    self.v, self.a = V0, 0.0
    self.out = None
    self.l3 = {"mode": ACTIVE, "jx_state": None, "t_stop": None, "token": False}

  def pose(self):
    return self.r.track.pose(self.s)

  def actor(self, now: float) -> Actor:
    x, z, psi = self.pose()
    return Actor(self.id, self.route, x, z, psi, self.v, self.s, now, self.a, l3=dict(self.l3))

  def ego(self, now: float) -> EgoState:
    x, z, psi = self.pose()
    return EgoState(now, self.id, self.route, x, z, psi, self.v, self.a, self.s, 0.0, True)

  def physics(self):
    jo = self.out
    a_op = min(1.5, max(-2.0, 0.8 * (V0 - self.v)))  # openpilot: blind to the junction, cruising at 8 m/s
    cmd = min(a_op, jo.a_cap)
    if jo.a_floor is not None:
      cmd = max(cmd, jo.a_floor)
    for _ in range(SUB):
      h = DT / SUB
      if jo.brake_hold:
        self.a = -3.0 if self.v > 0 else 0.0
      else:
        self.a += (cmd - self.a) * h / TAU
      self.v = max(self.v + self.a * h, 0.0)
      self.s += self.v * h


def queue_order(t_stops: dict[str, float]) -> list[str]:
  """reference: sort by t_stop, clusters of stop times within tie_s of the cluster's first, by id inside a cluster"""
  keys, start, idx = {}, None, -1
  for cid, t in sorted(t_stops.items(), key=lambda kv: (kv[1], kv[0])):
    if start is None or t - start > JP.tie_s:
      start, idx = t, idx + 1
    keys[cid] = (idx, cid)
  return sorted(t_stops, key=lambda c: keys[c])


def run(cmap, geom, assign: dict[str, str], delays: dict[str, float]):
  """assign: id -> route; delays: id -> arrival delay. Checks the invariants every tick.
  returns (trace, cars, queue order, first GO tick per car, opposite cars shared the token, end time)"""
  cars = {cid: Car(cid, route, cmap, geom, delays[cid]) for cid, route in assign.items()}
  sig = SignalHealth(UNTRUSTED, "conflict")
  conf = geom.conflicts
  t, trace = 1000.0, []
  first_in: dict[tuple[str, str], float] = {}
  last_in: dict[tuple[str, str], float] = {}
  shared = False
  checked_tokens = 0
  while t < 1000.0 + T_MAX:
    snap = {cid: c.actor(t) for cid, c in cars.items()}
    for cid, c in cars.items():
      c.out = c.jm.step(c.ego(t), [a for o, a in snap.items() if o != cid], sig, t)
    for c in cars.values():
      c.l3 = {"mode": ACTIVE, "jx_state": c.out.state, "t_stop": c.out.t_stop, "token": c.out.token}
    trace.append(tuple((cid, c.out.state, c.out.token, round(c.s, 6)) for cid, c in sorted(cars.items())))

    # no two conflicting cars hold the token at once; opposite ones may
    holders = [cid for cid, c in cars.items() if c.out.token and c.out.state != CLEARED]
    for a, b in itertools.combinations(holders, 2):
      ca, cb = cars[a], cars[b]
      if geom.conflicting(ca.route, cb.route):
        # a holder through its conflict with the other no longer matters
        _, cout_a, _, cout_b = conf[ca.route][cb.route]
        assert ca.s > cout_a or cb.s > cout_b, (t, a, b)
      else:
        shared = True

    # exactly the right cars hold the token: a stopped, settled car holds it iff every conflicting car ahead of it in the queue
    # is through its conflict with it
    stops = {cid: c.jm.t_stop for cid, c in cars.items() if c.jm.t_stop is not None}
    order = queue_order(stops)
    for cid, c in cars.items():
      if c.out.state != STOPPED or c.jm.t_stop is None or t - c.jm.t_stop_known < TOKEN_SETTLE_S:
        continue
      ahead = order[:order.index(cid)]
      expect = all(cars[o].out.state == CLEARED or cars[o].s > conf[cars[o].route][c.route][1]
                   for o in ahead if geom.conflicting(c.route, cars[o].route))
      assert c.out.token == expect, (t, cid, order, {o: (cars[o].out.state, cars[o].out.token) for o in cars})
      checked_tokens += 1

    # physical separation of conflicting pairs
    for a, b in itertools.combinations(sorted(cars), 2):
      ca, cb = cars[a], cars[b]
      if not geom.conflicting(ca.route, cb.route):
        continue
      cin_a, cout_a, cin_b, cout_b = conf[ca.route][cb.route]
      ins_a, ins_b = cin_a <= ca.s <= cout_a, cin_b <= cb.s <= cout_b
      assert not (ins_a and ins_b), (t, a, b)
      for cid, ins in ((a, ins_a), (b, ins_b)):
        if ins:
          first_in.setdefault((cid, a + b), t)
          last_in[(cid, a + b)] = t
    for c in cars.values():
      c.physics()
      # never stopped in the box
      if c.r.junction_s[0] <= c.s <= c.r.junction_s[1]:
        assert c.v > 0.5 or c.out.state not in (COMMIT,), (t, c.id, c.v)
    t += DT
    if all(c.out.state == CLEARED for c in cars.values()):
      break
  assert all(c.out.state == CLEARED for c in cars.values()), {cid: (c.out.state, c.out.reasons) for cid, c in cars.items()}
  # PET between every conflicting pair
  pets = []
  for a, b in itertools.combinations(sorted(cars), 2):
    if geom.conflicting(cars[a].route, cars[b].route):
      k = a + b
      pet = max(first_in[(b, k)] - last_in[(a, k)], first_in[(a, k)] - last_in[(b, k)])
      pets.append(pet)
  assert min(pets) >= JP.pet_min - 0.2, pets
  assert checked_tokens > 0
  stops = {cid: c.jm.t_stop for cid, c in cars.items()}
  commit_t = {cid: next(i for i, row in enumerate(trace) if {r[0]: r[1] for r in row}[cid] in (GO, COMMIT)) for cid in cars}
  return trace, cars, queue_order(stops), commit_t, shared, t


def check_fifo(cars, order, commit_t, geom):
  """a conflicting car ahead in the queue always goes first"""
  for i, a in enumerate(order):
    for b in order[i + 1:]:
      if geom.conflicting(cars[a].route, cars[b].route):
        assert commit_t[a] < commit_t[b], (order, commit_t)


@pytest.mark.parametrize("perm", list(itertools.permutations(ROUTES)))
def test_arrival_orders(cmap, geom, perm):
  """cars arrive 0.6 s apart (> tie_s) in every order: FIFO between conflicting cars, all clear, deterministic"""
  assign = {f"op{i}": r for i, r in enumerate(ROUTES)}
  by_route = {r: cid for cid, r in assign.items()}
  delays = {by_route[r]: 0.6 * k for k, r in enumerate(perm)}
  trace, cars, order, commit_t, shared, t_end = run(cmap, geom, assign, delays)
  assert [cars[c].route for c in order] == list(perm)
  check_fifo(cars, order, commit_t, geom)
  # leading run of one phase goes together (opposite movements share the grant)
  first_phase = cmap.routes[perm[0]].phase
  lead = list(itertools.takewhile(lambda r: cmap.routes[r].phase == first_phase, perm))
  first_go = min(commit_t.values())
  assert {cars[c].route for c in cars if commit_t[c] - first_go < 2.0 / DT} >= set(lead)
  if len(lead) > 1:
    assert shared
  assert t_end < 1000.0 + 60.0
  # deterministic
  trace2, *_ = run(cmap, geom, assign, delays)
  assert trace == trace2


@pytest.mark.parametrize("ids", list(itertools.permutations(range(4))))
def test_exact_ties(cmap, geom, ids):
  """all four stop at exactly the same time: the lower id goes first, its opposite with it if it is next in id order"""
  assign = {f"op{ids[i]}": r for i, r in enumerate(ROUTES)}
  delays = dict.fromkeys(assign, 0.0)
  trace, cars, order, commit_t, shared, t_end = run(cmap, geom, assign, delays)
  assert order == sorted(assign)
  check_fifo(cars, order, commit_t, geom)
  first = order[0]
  assert commit_t[first] == min(commit_t.values())
  if not geom.conflicting(cars[order[0]].route, cars[order[1]].route):
    assert shared and commit_t[order[1]] == commit_t[first]
  assert t_end < 1000.0 + 60.0
  trace2, *_ = run(cmap, geom, assign, delays)
  assert trace == trace2


@pytest.mark.parametrize("perm", list(itertools.permutations(ROUTES))[::5])
def test_near_ties(cmap, geom, perm):
  """stops 0.12 s apart: within tie_s of the first they tie (lower id first), the 4th (0.36 s) is a new cluster"""
  assign = {f"op{i}": r for i, r in enumerate(ROUTES)}
  by_route = {r: cid for cid, r in assign.items()}
  delays = {by_route[r]: 0.12 * k for k, r in enumerate(perm)}
  trace, cars, order, commit_t, shared, t_end = run(cmap, geom, assign, delays)
  expect = sorted(by_route[r] for r in perm[:3]) + [by_route[perm[3]]]
  assert order == expect
  check_fifo(cars, order, commit_t, geom)
  assert t_end < 1000.0 + 60.0


def test_deadlock_breaker(cmap, geom):
  """the queue head (E_W, cooperative) can't pass its gap test: a non-cooperative S_N car sits at its line. Nobody moves; after
  deadlock_s the lowest (t_stop, id) whose gap test passes (us, N_S: S_N doesn't conflict with us) takes the token"""
  sig = SignalHealth(UNTRUSTED, "conflict")

  def at_line(route):
    r = cmap.routes[route]
    x, z, psi = r.track.pose(r.stop_line_s - 3.3)
    return x, z, psi, r.stop_line_s - 3.3

  x, z, psi, s = at_line("N_S")
  ego = EgoState(0, "op0", "N_S", x, z, psi, 0.0, 0.0, s, 0.0, True)
  xo, zo, psio, so = at_line("E_W")
  head = Actor("op2", "E_W", xo, zo, psio, 0.0, so, 0.0, l3={"mode": ACTIVE, "jx_state": STOPPED, "t_stop": 990.0, "token": False})
  xn, zn, psin, sn = at_line("S_N")
  rogue = Actor("op1", "S_N", xn, zn, psin, 0.0, sn, 0.0)
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  t, got = 1000.0, None
  while t < 1030.0:
    jo = jm.step(ego, [head, rogue], sig, t)
    if jo.token:
      got = t
      break
    assert jo.state == STOPPED and jo.brake_hold
    t += DT
  assert got is not None and got - 1000.0 >= JP.deadlock_s - 1e-6 and jm.breaker
  assert jo.state == GO
  # the same with the head able to go: it is lower, so we don't break
  jm = JunctionManager("op0", "N_S", cmap, JP, geom)
  t = 1000.0
  while t < 1030.0:
    assert not jm.step(ego, [head], sig, t).token
    t += DT
