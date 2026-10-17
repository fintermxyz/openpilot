"""Kinematic closed loop of the crossroads, no engine and no openpilot. SIMULATION ONLY.

N point-mass cars on the real CrossroadsMap routes (s advances by v*dt, first-order accel lag), 100 Hz physics. An in-process
director runs crossroads.SignalController (with its stuck_green fault) and broadcasts WorldMsg at 20 Hz with a delivery
latency; each car runs the same L3Host / Arbiter code as the bridge at 25 Hz on a fake clock. The "advanced controller" is an
openpilot-like longitudinal model that reproduces what openpilot does at this junction: cruises at the limit, eases down to a
seeded 2.5-8 m/s near the junction and drives through, blind to lights and cross traffic, following a lead in its own lane.

The judge is the director's SAT footprint overlap plus PET per conflicting pair, red entries, box standstills and deadlocks.
"""
import heapq
import itertools
import math
import os
import random
from dataclasses import dataclass, field

import numpy as np

from openpilot.tools.sim.bridge.zoomengine.crossroads import CAR_HALF, ROUTES, CrossroadsMap, SignalController, footprint, overlap
from openpilot.tools.sim.l3.msgs import EM, Actor, EgoState, WorldMsg
from openpilot.tools.sim.l3.params import PARAMS, L3Params

MAP_PATH = os.environ.get("CROSSROADS_MAP", os.path.join(os.environ.get("ZOOMENGINE_DIR", os.path.expanduser("~/Documents/zoom/ZoomEngine-main")),
                                                         "assets/maps/crossroads.built.json"))

PHYS_HZ = 100
DIRECTOR_EVERY = 5      # 20 Hz world broadcast
TICK_EVERY = 4          # 25 Hz L3Host.tick
TAU = 0.3               # s first-order accel lag (engine + pedals ECU)
CLOCK0 = 1000.0         # fake time.monotonic() at trial start (never 0: catches "if t:" bugs)
HOLD_A = -6.0           # brake hold: this until stopped
BOX_STOP_V = 0.2
EM_GRACE = 2.0          # s: a box standstill this soon after EM is an emergency stop, judged separately
STALL_S = 30.0          # s with cars still to cross and nobody crossing: deadlock
CAR_R2 = (2.0 * math.hypot(*CAR_HALF)) ** 2  # centre distance below which SAT runs

_CMAP: CrossroadsMap | None = None
_GEOM = None


def crossroads_map() -> CrossroadsMap:
  global _CMAP
  if _CMAP is None:
    _CMAP = CrossroadsMap(MAP_PATH)
  return _CMAP


def conflict_geometry():
  global _GEOM
  if _GEOM is None:
    from openpilot.tools.sim.l3.junction import ConflictGeometry
    _GEOM = ConflictGeometry(crossroads_map(), PARAMS.junction)
  return _GEOM


# ---- openpilot-like advanced controller ----

@dataclass
class OpLike:
  """longitudinal model of openpilot at the crossroads: speed target from the map (it slows for the junction it can see in the
  road layout, not for lights or cross traffic) and IDM car-following in its own lane"""
  v_cruise: float = 13.41
  v_junction: float = 5.0   # seeded 2.5-8: what openpilot slowed to in the engine runs
  b_ease: float = 1.5       # m/s^2 ease-down rate towards the junction
  k_v: float = 0.8
  a_max: float = 1.5
  a_min: float = -3.0
  # IDM
  T: float = 1.2
  s0: float = 2.5
  b_idm: float = 2.0

  def accel(self, s: float, v: float, stop_line_s: float, junction_end: float, lead_gap: float | None, lead_v: float) -> float:
    d = stop_line_s - s
    if d > 0:
      v_des = min(self.v_cruise, math.sqrt(self.v_junction ** 2 + 2 * self.b_ease * d))
    elif s < junction_end + 3.0:
      v_des = self.v_junction
    else:
      v_des = self.v_cruise
    a = self.k_v * (v_des - v)
    if lead_gap is not None:
      s_star = self.s0 + max(0.0, v * self.T + v * (v - lead_v) / (2 * math.sqrt(self.a_max * self.b_idm)))
      a_idm = self.a_max * (1 - (v / self.v_cruise) ** 4 - (s_star / max(lead_gap, 0.1)) ** 2)
      a = min(a, a_idm)
    return float(np.clip(a, self.a_min, self.a_max))


# ---- cars ----

@dataclass
class SimCar:
  id: str
  route: str
  s: float
  v: float
  t_spawn: float
  ctrl: OpLike
  host: object | None = None
  a: float = 0.0
  a_cmd: float = 0.0
  x: float = 0.0
  z: float = 0.0
  psi: float = 0.0
  active: bool = False
  holding: bool = False
  em_t: float = -math.inf
  failsafe: bool = False
  s_hist: list = field(default_factory=list)  # (t, s) while active
  rec: dict = field(default_factory=dict)


def _l3_dict(host) -> dict | None:
  if host is None:
    return None
  try:
    st = host.status()
  except Exception:
    return None
  return {k: st.get(k) for k in ("mode", "jx_state", "t_stop", "token")}


def _cross_t(hist: np.ndarray, s_th: float) -> float | None:
  """first time the car's centre reached arc length s_th"""
  idx = np.nonzero(hist[:, 1] >= s_th)[0]
  return float(hist[idx[0], 0]) if len(idx) else None


def run_episode(seed: int, fault: str = "stuck_green", fault_at: float = 6.0, l3: bool = True, n_cars: int = 4, latency: float = 0.05,
                max_t: float = 90.0, stagger: bool = False, params: L3Params = PARAMS, trace: bool = False,
                l3_mask: set[int] | None = None, sig_fn=None, spawn: dict | None = None, v_junction: dict | None = None,
                ctrl: dict | None = None) -> dict:
  """one trial; returns the judge's verdict. fault: 'stuck_green' | 'none'. Cars beyond 4 queue behind on the same approaches.
  Scenario hooks: l3_mask (car indices running L3, default all when l3), sig_fn(t, controller) -> aspects (replaces the
  controller's), spawn {i: (s, v, t_spawn)}, v_junction {i: m/s}, ctrl {i: OpLike instance or subclass instance}"""
  assert fault in ("stuck_green", "none"), fault
  cmap = crossroads_map()
  rng = random.Random(seed)
  now = CLOCK0
  clock = lambda: now  # noqa: E731  every host reads the shared fake clock
  Host = None
  if l3:
    from openpilot.tools.sim.l3.host import L3Host
    Host = L3Host

  cars: list[SimCar] = []
  last_spawn: dict[str, float] = {}
  for i in range(n_cars):
    route = ROUTES[i % len(ROUTES)]
    r = cmap.routes[route]
    # stagger: each approach's first car 0-12 s late; cars behind on the same approach always 4+ s after the one ahead
    if route not in last_spawn:
      t_spawn = rng.uniform(0.0, 12.0) if stagger else 0.0
    else:
      t_spawn = last_spawn[route] + (rng.uniform(4.0, 10.0) if stagger else 5.0)
    last_spawn[route] = t_spawn
    oc = OpLike(v_cruise=cmap.speed_limit, v_junction=rng.uniform(2.5, 8.0))
    v0 = max(0.0, cmap.speed_limit + rng.uniform(-0.7, 0.7))
    s0 = r.spawn_s + rng.uniform(-3.0, 3.0)
    if ctrl is not None and i in ctrl:
      oc = ctrl[i]
    if v_junction is not None and i in v_junction:
      oc.v_junction = v_junction[i]
    if spawn is not None and i in spawn:
      s0, v0, t_spawn = spawn[i]
      last_spawn[route] = t_spawn
    car = SimCar(f"op{i}", route, s0, v0, t_spawn, oc)
    if Host is not None and (l3_mask is None or i in l3_mask):
      car.host = Host(car.id, route, cmap, params, clock=clock)
    car.rec = {"route": route, "v_junction": round(oc.v_junction, 2), "t_spawn": round(t_spawn, 2), "t_cross": None,
               "min_v_before_line": 99.0, "stopped_before_line": False, "red_entry": False, "box_stop": False, "box_stop_em": False,
               "em": False, "failsafe": False}
    cars.append(car)

  signals = SignalController()
  inbox: list[tuple[float, int, WorldMsg]] = []  # (deliver_t, seq, msg)
  seq = 0
  dt = 1.0 / PHYS_HZ
  collisions: list[dict] = []
  hit_pairs: set = set()
  shown = signals.aspects(0.0)
  t_last_cross = 0.0
  frames = []
  n_steps = int(round(max_t * PHYS_HZ))
  done_t = None

  for k in range(n_steps + 1):
    t = k * dt
    now = CLOCK0 + t
    for c in cars:
      if not c.active and t >= c.t_spawn - 1e-9:
        c.active = True
        c.x, c.z, c.psi = cmap.routes[c.route].track.pose(c.s)
    live = [c for c in cars if c.active]

    # director: signals + world broadcast (20 Hz)
    if k % DIRECTOR_EVERY == 0:
      signals.fault = fault == "stuck_green" and 0 <= fault_at <= t
      shown = signals.aspects(t) if sig_fn is None else sig_fn(t, signals)
      actors = [Actor(c.id, c.route, c.x, c.z, c.psi, c.v, c.s, now, c.a, CAR_HALF[0], CAR_HALF[1], "oracle", _l3_dict(c.host)) for c in live]
      msg = WorldMsg(seed, seq, now, dict(shown), actors)
      heapq.heappush(inbox, (now + latency, seq, msg))
      seq += 1
    while inbox and inbox[0][0] <= now + 1e-9:
      _, _, msg = heapq.heappop(inbox)
      for c in cars:
        if c.host is not None:
          c.host.feed(msg)

    # L3 supervisors (25 Hz)
    if k % TICK_EVERY == 0:
      for c in live:
        if c.host is not None:
          c.host.tick(EgoState(now, c.id, c.route, c.x, c.z, c.psi, c.v, c.a, c.s, 0.0, True, CAR_HALF[0], CAR_HALF[1]))
          if c.host.status().get("mode") == EM:
            c.rec["em"], c.em_t = True, t

    # advanced controller -> arbiter -> point mass (100 Hz)
    for c in live:
      r = cmap.routes[c.route]
      lead = min((o for o in live if o is not c and o.route == c.route and o.s > c.s), key=lambda o: o.s, default=None)
      gap = (lead.s - c.s - 2 * CAR_HALF[1]) if lead is not None else None
      a_op = c.ctrl.accel(c.s, c.v, r.stop_line_s, r.junction_s[1], gap, lead.v if lead is not None else 0.0)
      if c.host is not None:
        out = c.host.arbiter.apply(a_op, now, dt, v=c.v)
        a_cmd = out.accel
        if out.failsafe:
          c.failsafe = True
          c.rec["failsafe"] = True
        c.holding = out.brake_hold or (out.failsafe and c.v < 0.05)
        if c.holding:
          a_cmd = HOLD_A if c.v > 0 else 0.0
      else:
        a_cmd = a_op
      c.a_cmd = a_cmd
      c.a += (a_cmd - c.a) * dt / TAU
      v_new = c.v + c.a * dt
      if v_new <= 0.0:
        v_new = 0.0
        c.a = max(c.a, 0.0) if c.a_cmd >= 0 else 0.0
      c.s += 0.5 * (c.v + v_new) * dt
      c.v = v_new
      if c.s > r.track.length - 8.0:  # end of route: park
        c.s, c.v, c.a = min(c.s, r.track.length - 8.0), 0.0, 0.0
      c.x, c.z, c.psi = r.track.pose(c.s)
      c.s_hist.append((t, c.s))

    # judge (100 Hz)
    for c in live:
      r = cmap.routes[c.route]
      rec = c.rec
      front = c.s + CAR_HALF[1]  # stop line and red entry are judged on the front bumper (as junction.py's d_line)
      if front < r.stop_line_s:
        rec["min_v_before_line"] = min(rec["min_v_before_line"], c.v)
        if c.v < 0.5 and r.stop_line_s - front < 6.0:
          rec["stopped_before_line"] = True
          rec.setdefault("t_stop_line", round(t, 2))  # first standstill at the line: its arrival for the give-way order
      prev_front = (c.s_hist[-2][1] if len(c.s_hist) > 1 else c.s) + CAR_HALF[1]
      if prev_front < r.stop_line_s <= front and shown[r.phase] in ("red", "red_amber"):
        rec["red_entry"] = True
        rec["red_entry_t"] = round(t, 2)
      if "t_enter" not in rec and front >= r.junction_s[0]:
        rec["t_enter"] = round(t, 2)
      if rec["t_cross"] is None and c.s >= r.junction_s[1] + CAR_HALF[1]:
        rec["t_cross"] = round(t, 2)
        t_last_cross = t
      if r.junction_s[0] <= c.s <= r.junction_s[1] and c.v < BOX_STOP_V and not rec["box_stop"] and not rec["box_stop_em"]:
        # an emergency stop (EM in the last EM_GRACE s) that ended in the box is counted apart from a plain box standstill
        rec["box_stop_em" if t - c.em_t <= EM_GRACE else "box_stop"] = True
        rec["box_stop_t"] = round(t, 2)
    for a, b in itertools.combinations(live, 2):
      if (a.x - b.x) ** 2 + (a.z - b.z) ** 2 > CAR_R2 or (a.id, b.id) in hit_pairs:
        continue
      if overlap(footprint(a.x, a.z, a.psi), footprint(b.x, b.z, b.psi)):
        hit_pairs.add((a.id, b.id))
        collisions.append({"cars": [a.id, b.id], "t": round(t, 2), "speeds": [round(a.v, 2), round(b.v, 2)], "routes": [a.route, b.route],
                           "lights": dict(shown)})
    if trace and k % DIRECTOR_EVERY == 0:
      frames.append({"t": round(t, 2), "spat": dict(shown), "cars": {c.id: {"s": round(c.s, 2), "v": round(c.v, 2), "a": round(c.a_cmd, 2),
                     "l3": c.host.status() if c.host is not None else None} for c in live}})
    if len(live) == len(cars) and all(c.rec["t_cross"] is not None and c.s > cmap.routes[c.route].junction_s[1] + 10.0 for c in cars):
      done_t = t
      break

  # post-episode: PET for every conflicting pair that both crossed
  pets = []
  geom = conflict_geometry()
  hists = {c.id: np.asarray(c.s_hist) if c.s_hist else np.zeros((0, 2)) for c in cars}
  for a, b in itertools.combinations(cars, 2):
    iv = geom.conflicts.get(a.route, {}).get(b.route)
    if iv is None or not len(hists[a.id]) or not len(hists[b.id]):
      continue
    cin_a, cout_a, cin_b, cout_b = iv
    ta_in, ta_out = _cross_t(hists[a.id], cin_a), _cross_t(hists[a.id], cout_a)
    tb_in, tb_out = _cross_t(hists[b.id], cin_b), _cross_t(hists[b.id], cout_b)
    if None in (ta_in, ta_out, tb_in, tb_out):
      continue
    # negative: both occupied the conflict zone at once
    first = a.id if ta_out <= tb_out else b.id
    pets.append({"cars": [a.id, b.id], "first": first, "pet": round(max(tb_in - ta_out, ta_in - tb_out), 3)})

  crossed = [c for c in cars if c.rec["t_cross"] is not None]
  collided = {i for h in collisions for i in h["cars"]}
  stuck = [c.id for c in cars if c.rec["t_cross"] is None and c.id not in collided]
  out = {
    "seed": seed, "fault": fault, "fault_at": fault_at, "l3": l3, "n_cars": n_cars, "latency": latency, "stagger": stagger,
    "params_hash": params.hash(),
    "collisions": collisions, "n_collisions": len(collisions),
    "pets": pets, "min_pet": min((p["pet"] for p in pets), default=None),
    "box_stops": [c.id for c in cars if c.rec["box_stop"]],
    "box_stops_em": [c.id for c in cars if c.rec["box_stop_em"]],
    "em": [c.id for c in cars if c.rec["em"]],
    "red_entries": [c.id for c in cars if c.rec["red_entry"]],
    "failsafe": [c.id for c in cars if c.rec["failsafe"]],
    "arb_violations": sum(c.host.arbiter.stats["violations"] for c in cars if c.host is not None),
    "uncleared": stuck,
    "timeout": len(stuck) > 0,   # a car still not across at max_t
    "deadlock": len(stuck) > 0 and (done_t if done_t is not None else max_t) - t_last_cross >= STALL_S,  # ... and nobody crossed for STALL_S
    "t_clear": round(max(c.rec["t_cross"] for c in crossed), 2) if len(crossed) == len(cars) else None,
    "t_end": round(done_t if done_t is not None else max_t, 2),
    "t_last_cross": round(t_last_cross, 2),
    "cars": {c.id: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.rec.items()} for c in cars},
  }
  if trace:
    out["frames"] = frames
  return out


def run_episode_kw(kw: dict) -> dict:
  """picklable pool entry point"""
  return run_episode(**kw)


if __name__ == "__main__":
  import argparse
  import json
  ap = argparse.ArgumentParser(description="one kinematic crossroads episode")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--fault", default="stuck_green", choices=("stuck_green", "none"))
  ap.add_argument("--fault-at", type=float, default=6.0)
  ap.add_argument("--no-l3", action="store_true")
  ap.add_argument("--cars", type=int, default=4)
  ap.add_argument("--stagger", action="store_true")
  args = ap.parse_args()
  res = run_episode(args.seed, args.fault, args.fault_at, not args.no_l3, args.cars, stagger=args.stagger)
  print(json.dumps(res, indent=1))
