"""Junction manager for the UK signalised crossroads (ODD-X, research extension). SIMULATION ONLY. Pure and deterministic.

ConflictGeometry: per route pair, the arc-length intervals on each route where the two cars' swept rectangles overlap.
JunctionManager: FAR -> APPROACH -> {GO | STOP_AT_LINE -> STOPPED (queue)} -> COMMIT -> CLEARED, with
  - the approach speed envelope: until GO the car can always stop at the line with <= a_b (the core crash fix),
  - the GO gate: signal (or the unmarked token) AND the PET gap test against every conflicting car AND exit space,
  - the point of no return: GO lost when the stop at the line is no longer feasible -> emergency stop before the conflict zone
    if that is feasible at a_em, else COMMIT (keep >= v_clear through the box, never stop in it),
  - the unmarked protocol when the signal is UNTRUSTED (Highway Code 176, nobody has priority): mandatory stop at the line,
    queue by (t_stop, id), one token per conflicting pair (opposite through movements go together), 8 s deadlock breaker.

Conventions: s is the car centre's arc length on its own route (as the director judges it); d_line is the front bumper's distance
to the stop line. Conflict intervals are in centre s, so they already include both cars' lengths.
"""
import hashlib
import json
import math

import numpy as np

from openpilot.tools.sim.l3.msgs import (ACTIVE, APPROACH, CLEARED, COMMIT, CONSTRAINED, FAR, GO, STOP_AT_LINE, STOPPED, STOPPED_YIELD,
                                         TRUSTED_AMBER, TRUSTED_GREEN, TRUSTED_RED,
                                         TRUSTED_RED_AMBER, UNTRUSTED, Actor, EgoState, JunctionOut, SignalHealth)
from openpilot.tools.sim.l3.params import JunctionParams

INF = math.inf
SAMPLE_DS = 0.25          # m centreline sampling of the conflict geometry
CAR_HALF_L = 2.3
CAR_HALF_W = 0.92
GRANT_PET_MARGIN = 0.5    # s extra PET demanded to grant GO (re-checks use pet_min): no GO / no-GO chatter at the boundary
TOKEN_SETTLE_S = 0.8      # s own t_stop must have been broadcast this long before claiming the token (>= tie_s + feed_stale_s)
OTHER_BRAKE_MAX = 8.0     # m/s^2 the hardest another car may brake (latest exit: it may stop in the zone)
DECEL_A = 0.5             # m/s^2 "already decelerating"
COMMIT_BRAKE_MAX = 2.0    # m/s^2 COMMIT floor: openpilot may brake at most this hard in the box, easing to 0 at v_clear
STOPPED_V = 0.3           # m/s another car slower than this is stopped (it may launch: rho_other_stopped)
LIGHT_STOP_SLACK = 0.5   # m/s^2 a stop decided at amber_stop_a (GO lost on a light change) may then brake this much harder
YIELD_SUSTAIN_S = 0.5     # s a non-cooperative car must have been braking hard enough to stop at its line before it counts as yielding
LEAD_B = 2.5              # m/s^2 the braking a same-route lead check allows the ego (openpilot's comfortable stop), with rho reaction
LEAD_MARGIN = 4.0         # m standstill gap the lead check keeps
QUEUE_D = 4.0             # m a stopped car's front within this of its line (or past it, before the box) is queued
MOVE_AREA_D = 30.0        # m cars this close to their line (or in the box) count for the deadlock breaker's "nobody moving"
COOP_MODES = (ACTIVE, CONSTRAINED)
WAITING = (APPROACH, STOP_AT_LINE, STOPPED, STOPPED_YIELD)


# ---- kinematic envelopes (pure) ----

def time_to(d: float, v: float, a: float, v_max: float) -> float:
  """time to cover d accelerating at a from v up to v_max, then constant (inf if it never gets there)"""
  if d <= 0.0:
    return 0.0
  v = max(v, 0.0)
  if a <= 0.0 or v >= v_max:
    return d / v if v > 1e-9 else INF
  t1 = (v_max - v) / a
  d1 = 0.5 * (v + v_max) * t1
  if d <= d1:
    return (-v + math.sqrt(v * v + 2.0 * a * d)) / a
  return t1 + (d - d1) / v_max


def earliest_arrival(d: float, v: float, a0: float, rho: float, a_max: float, v_max: float) -> float:
  """a car reacting for rho (accel <= a0 meanwhile), then accelerating at a_max up to v_max: earliest time it covers d"""
  if d <= 0.0:
    return 0.0
  v = max(v, 0.0)
  t = time_to(d, v, a0, v_max)
  if t <= rho:
    return t
  # state after the reaction
  if a0 > 0.0 and v < v_max:
    t1 = min(rho, (v_max - v) / a0)
    s_r = v * t1 + 0.5 * a0 * t1 * t1 + (rho - t1) * min(v + a0 * t1, v_max)
    v_r = min(v + a0 * rho, v_max)
  else:
    s_r, v_r = v * rho, v
  return rho + time_to(d - s_r, v_r, a_max, v_max)


def latest_exit(d: float, v: float, b: float) -> float:
  """braking at b from v: time to cover d, inf if it can stop first"""
  if d <= 0.0:
    return 0.0
  v = max(v, 0.0)
  disc = v * v - 2.0 * b * d
  if disc <= 0.0:
    return INF
  return (v - math.sqrt(disc)) / b


class EgoProfiles:
  """the two ego speed profiles of the gap test.
  fast (planned): from v accelerate at a_go up to v_cross (if already faster, constant v). GO / COMMIT cap the ego to this.
  slow: the COMMIT floor, dv/dt = clip(k*(v_clear - v), -b_floor, a_go): openpilot may slow the car to v_clear, never below"""
  def __init__(self, p: JunctionParams, v_cross: float):
    self.p, self.v_cross = p, v_cross
    self.k = max(p.k_v, 1e-3)

  def fast_time(self, d: float, v: float) -> float:
    return time_to(d, v, self.p.a_go, max(self.v_cross, v))

  def slow_s(self, t: float, v0: float) -> float:
    p, k, vc = self.p, self.k, self.p.v_clear
    v_hi, v_lo = vc + COMMIT_BRAKE_MAX / k, vc - p.a_go / k
    s, v = 0.0, max(v0, 0.0)
    if v > v_hi:  # braking at the floor's limit
      t1 = min(t, (v - v_hi) / COMMIT_BRAKE_MAX)
      s += v * t1 - 0.5 * COMMIT_BRAKE_MAX * t1 * t1
      v -= COMMIT_BRAKE_MAX * t1
      t -= t1
    elif v < v_lo:  # launching at a_go
      t1 = min(t, (v_lo - v) / p.a_go)
      s += v * t1 + 0.5 * p.a_go * t1 * t1
      v += p.a_go * t1
      t -= t1
    if t > 0.0:  # exponential approach to v_clear
      s += vc * t + (v - vc) * (1.0 - math.exp(-k * t)) / k
    return s

  def slow_time(self, d: float, v: float) -> float:
    if d <= 0.0:
      return 0.0
    lo, hi = 0.0, 1.0
    while self.slow_s(hi, v) < d:
      lo, hi = hi, hi * 2.0
      if hi > 1e5:
        return INF
    for _ in range(60):
      mid = 0.5 * (lo + hi)
      if self.slow_s(mid, v) < d:
        lo = mid
      else:
        hi = mid
    return hi

  def fast_accel(self, v: float) -> float:
    """GO / COMMIT accel cap: never faster than the planned profile"""
    return min(self.p.a_go, max(0.0, self.k * (self.v_cross - v)))

  def floor_accel(self, v: float) -> float:
    """COMMIT accel floor: the slow profile"""
    return min(self.p.a_go, max(-COMMIT_BRAKE_MAX, self.k * (self.p.v_clear - v)))


# ---- geometry ----

_GEOM_CACHE: dict[str, dict] = {}


def _rects_overlap(ca, fa, ra, cb, fb, rb, hl: float, hw: float) -> np.ndarray:
  """separating axis test of N rectangle pairs (centres c, unit forward f and right r, all (N, 2)), equal half sizes"""
  dc = cb - ca
  sep = np.zeros(len(ca), dtype=bool)
  for n in (fa, ra, fb, rb):
    proj = np.abs(np.sum(dc * n, axis=1))
    ext = (hl * (np.abs(np.sum(fa * n, axis=1)) + np.abs(np.sum(fb * n, axis=1))) +
           hw * (np.abs(np.sum(ra * n, axis=1)) + np.abs(np.sum(rb * n, axis=1))))
    sep |= proj > ext
  return ~sep


def _close_pairs(ca: np.ndarray, cb: np.ndarray, reach2: float, block: int = 64) -> tuple[np.ndarray, np.ndarray]:
  """index pairs (i, j) with |ca[i] - cb[j]|^2 < reach2. Blocks of samples are prefiltered by their bounding boxes, so only the
  few block pairs near the junction (or along a close parallel route) get a dense distance matrix: ~1 MB, not ~120 MB"""
  reach = math.sqrt(reach2)
  def blocks(c):
    return [(k, c[k:k + block], c[k:k + block].min(axis=0), c[k:k + block].max(axis=0)) for k in range(0, len(c), block)]
  out_a, out_b = [], []
  bb = blocks(cb)
  for ka, pa, lo_a, hi_a in blocks(ca):
    for kb, pb, lo_b, hi_b in bb:
      gap = np.maximum(0.0, np.maximum(lo_b - hi_a, lo_a - hi_b))
      if float(gap @ gap) >= reach2 or np.any(gap >= reach):
        continue
      i, j = np.nonzero(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2) < reach2)
      out_a.append(i + ka)
      out_b.append(j + kb)
  if not out_a:
    return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
  return np.concatenate(out_a), np.concatenate(out_b)


class ConflictGeometry:
  """conflicts[a][b] = (cin_a, cout_a, cin_b, cout_b): centre arc lengths on routes a and b between which the two cars' swept
  rectangles (half width 0.92 + margin, half length 2.3) overlap. Only overlapping pairs appear. Cached by map + params"""
  def __init__(self, cmap, params: JunctionParams):
    self.half_w = CAR_HALF_W + params.swept_half_w_margin
    self.half_l = CAR_HALF_L
    key = hashlib.sha256(json.dumps([cmap.data.get("routes"), self.half_w, self.half_l, SAMPLE_DS], sort_keys=True).encode()).hexdigest()
    if key not in _GEOM_CACHE:
      _GEOM_CACHE[key] = self._build(cmap)
    self.conflicts: dict[str, dict[str, tuple[float, float, float, float]]] = _GEOM_CACHE[key]
    self.routes = cmap.routes

  def _build(self, cmap) -> dict:
    samples = {}
    for name, r in cmap.routes.items():
      ss = np.arange(0.0, r.track.length + 1e-9, SAMPLE_DS)
      pose = np.array([r.track.pose(float(s)) for s in ss])
      f = np.stack([np.sin(pose[:, 2]), np.cos(pose[:, 2])], axis=1)
      rt = np.stack([-np.cos(pose[:, 2]), np.sin(pose[:, 2])], axis=1)
      samples[name] = (ss, pose[:, :2], f, rt)
    reach2 = (2.0 * math.hypot(self.half_w, self.half_l)) ** 2
    out: dict[str, dict] = {n: {} for n in cmap.routes}
    names = sorted(cmap.routes)
    for i, a in enumerate(names):
      for b in names[i + 1:]:
        sa, ca, fa, ra = samples[a]
        sb, cb, fb, rb = samples[b]
        ia, ib = _close_pairs(ca, cb, reach2)
        if len(ia) == 0:
          continue
        hit = _rects_overlap(ca[ia], fa[ia], ra[ia], cb[ib], fb[ib], rb[ib], self.half_l, self.half_w)
        if not hit.any():
          continue
        ia, ib = ia[hit], ib[hit]
        # pad by one sample: the true boundary lies within a sample of the last overlapping one
        cin_a, cout_a = float(sa[ia].min() - SAMPLE_DS), float(sa[ia].max() + SAMPLE_DS)
        cin_b, cout_b = float(sb[ib].min() - SAMPLE_DS), float(sb[ib].max() + SAMPLE_DS)
        out[a][b] = (cin_a, cout_a, cin_b, cout_b)
        out[b][a] = (cin_b, cout_b, cin_a, cout_a)
    return out

  def conflicting(self, a: str, b: str) -> bool:
    return b in self.conflicts.get(a, {})


# ---- junction manager ----

class JunctionManager:
  def __init__(self, ego_id: str, route: str, cmap, params: JunctionParams, geometry: ConflictGeometry | None = None):
    self.ego_id, self.route, self.p = ego_id, route, params
    self.cmap = cmap
    self.geom = geometry if geometry is not None else ConflictGeometry(cmap, params)
    self.r = cmap.routes[route]
    self.v_lim = float(cmap.speed_limit)
    self.v_cross = min(self.v_lim, params.v_cross_max)
    self.prof = EgoProfiles(params, self.v_cross)
    cins = [c[0] for c in self.geom.conflicts.get(route, {}).values()]
    self.cin_any = min(cins) if cins else self.r.junction_s[0]
    self.reset()

  def reset(self) -> None:
    self.state = FAR
    self.em = False
    self.b_stop = self.p.a_b  # braking the current stop may use: a_b, or amber_stop_a + slack after a light change while going
    self.token = False
    self.breaker = False
    self.stop_since: float | None = None
    self.t_stop: float | None = None
    self.t_stop_known: float | None = None
    self.seen_stop: dict[str, float] = {}   # other car id -> since when observed stopped at its line
    self.decel_since: dict[str, float] = {}  # other car id -> since when observed braking hard enough to stop at its line
    self.now: float | None = None
    self.last_move: float | None = None

  # ---- helpers ----
  @staticmethod
  def coop(x: Actor) -> bool:
    return x.source != "unknown" and isinstance(x.l3, dict) and x.l3.get("mode") in COOP_MODES

  def d_line_of(self, x) -> float:
    return self.cmap.routes[x.route].stop_line_s - (x.s + x.half_l)

  def queued_stop(self, x) -> bool:
    """stopped at its line: front within QUEUE_D of it, or past it but not yet through the box"""
    return x.v < self.p.stop_v and self.d_line_of(x) < QUEUE_D and x.s < self.cmap.routes[x.route].junction_s[1]

  def stopping_for_line(self, x: Actor) -> bool:
    """braking at least as hard as stopping before its line needs (projected stop point at or before the line), and that is
    at most a_other_brake. Easing down towards a crossing speed is not this: its projected stop point is past the line"""
    d = self.d_line_of(x)
    if d <= 0.0 or x.v < STOPPED_V:
      return False
    need = x.v * x.v / (2.0 * d)
    return need <= self.p.a_other_brake and -x.a >= max(DECEL_A, need)

  def observe(self, actors: list[Actor], now: float) -> None:
    """per-tick bookkeeping of the other cars: observed stops at their lines and sustained braking for them"""
    self.now = now
    for x in actors:
      if self.queued_stop(x):
        self.seen_stop.setdefault(x.id, now)
      else:
        self.seen_stop.pop(x.id, None)
      if x.source != "unknown" and self.stopping_for_line(x):
        self.decel_since.setdefault(x.id, now)
      else:
        self.decel_since.pop(x.id, None)

  def yielding(self, x: Actor, signal: SignalHealth) -> bool:
    """the other car will not enter: right of way is given, not taken"""
    if x.source == "unknown":
      return False
    d = self.d_line_of(x)
    can_stop = d > 0.0 and x.v * x.v / (2.0 * d) <= self.p.a_other_brake
    if self.coop(x) and not x.l3.get("token"):
      st = x.l3.get("jx_state")
      if st in (STOPPED, STOPPED_YIELD) and x.v < STOPPED_V:
        return True
      if st == STOP_AT_LINE and (can_stop or x.v < STOPPED_V and d > -0.5):
        return True
    # any car (cooperative or not) seen braking to stop before its line for YIELD_SUSTAIN_S
    t0 = self.decel_since.get(x.id)
    if t0 is not None and self.now is not None and self.now - t0 >= YIELD_SUSTAIN_S and self.stopping_for_line(x):
      return True
    # our head trusted green / amber: the cross phase is red, a car stopped at its line is waiting for it
    if signal.state in (TRUSTED_GREEN, TRUSTED_AMBER) and x.v < STOPPED_V and d > -0.5:
      return True
    return False

  def other_window(self, x: Actor, cin_o: float, cout_o: float) -> tuple[float, float]:
    """[earliest entry, latest exit] of another car in its conflict interval. unknown (stale): as early as physically possible,
    never gone"""
    p = self.p
    v = max(x.v, 0.0)
    v_max = max(v, p.v_other_factor * self.v_lim)
    if x.source == "unknown":
      return earliest_arrival(cin_o - x.s, v, p.a_other_max, 0.0, p.a_other_max, v_max), INF
    rho = p.rho_other_stopped if v < STOPPED_V else p.rho_other
    a0 = min(max(x.a, 0.0), p.a_other_max)
    return earliest_arrival(cin_o - x.s, v, a0, rho, p.a_other_max, v_max), latest_exit(cout_o - x.s, v, OTHER_BRAKE_MAX)

  def ego_window(self, s: float, v: float, cin: float, cout: float) -> tuple[float, float]:
    return self.prof.fast_time(cin - s, v), self.prof.slow_time(cout - s, v)

  def gap_test(self, route: str, s: float, v: float, others: list[Actor], signal: SignalHealth, pet: float) -> tuple[bool, str]:
    """PET gap test of a car on `route` at (s, v) on the ego profiles against every conflicting car. (ok, blocking id)"""
    for x in others:
      c = self.geom.conflicts.get(route, {}).get(x.route)
      if c is None:
        continue
      cin_e, cout_e, cin_o, cout_o = c
      if s > cout_e:
        continue
      if x.source != "unknown" and x.s > cout_o:
        continue
      if self.yielding(x, signal):
        continue
      t_in_o, t_out_o = self.other_window(x, cin_o, cout_o)
      if t_in_o == INF:
        continue
      t_in_e, t_out_e = self.ego_window(s, v, cin_e, cout_e)
      if t_out_e + pet < t_in_o or t_in_e - pet > t_out_o:
        continue
      return False, x.id
    return True, ""

  def lead_blocks(self, ego: EgoState, actors: list[Actor]) -> str | None:
    """a car ahead on our route the ego could not stop behind at LEAD_B (after rho) if it kept the keep-moving floor, assuming
    the lead brakes at OTHER_BRAKE_MAX (unknown: already stopped). The floor must never push us into it"""
    v = max(ego.v, 0.0)
    d_ego = v * self.p.rho + v * v / (2.0 * LEAD_B)
    for x in actors:
      if x.route != self.route or x.s <= ego.s:
        continue
      v_l = 0.0 if x.source == "unknown" else max(x.v, 0.0)
      gap = (x.s - x.half_l) - (ego.s + ego.half_l)
      if gap < d_ego - v_l * v_l / (2.0 * OTHER_BRAKE_MAX) + LEAD_MARGIN:
        return x.id
    return None

  def keep_floor(self, ego: EgoState, actors: list[Actor], reasons: list[str]) -> float | None:
    """the COMMIT / launch floor, unless a lead ahead on our route is too close for it: then openpilot's braking for the lead
    wins (stopping behind it, even in the box, beats running into it)"""
    who = self.lead_blocks(ego, actors)
    if who is not None:
      reasons.append(f"lead {who}: no keep-moving floor")
      return None
    return self.prof.floor_accel(ego.v)

  def exit_clear(self, ego: EgoState, actors: list[Actor]) -> tuple[bool, str]:
    """room to clear (rules 174, 176): no slow car on our route ahead of us, in the box or within L_ego + exit_gap past it"""
    end = self.r.junction_s[1] + 2.0 * ego.half_l + self.p.exit_gap
    for x in actors:
      if x.route == self.route and x.s > ego.s and x.s - x.half_l < end and (x.v < self.p.exit_v or x.source == "unknown"):
        return False, x.id
    return True, ""

  def order(self, entries: list[tuple[float, str]]) -> dict[str, tuple[int, str]]:
    """queue order: by t_stop; stop times within tie_s of a cluster's first are a tie, broken by the lower id.
    Every car clusters the same set (all broadcast t_stops, never removed), so all agree on the order"""
    keys, start, idx = {}, None, -1
    for t, cid in sorted(entries):
      if start is None or t - start > self.p.tie_s:
        start, idx = t, idx + 1
      keys[cid] = (idx, cid)
    return keys

  def stop_distance(self, v: float, a_now: float, b: float) -> float:
    """stopping distance commanding -b from now through a first-order lag rho, the car's accel now a_now:
    v(t) ~ v + c - b t with c = (a_now + b) rho, s = (v + c)^2 / 2b - c rho (= v rho + v^2 / 2b - b rho^2 / 2 from a_now = 0)"""
    rho = self.p.rho
    c = max(a_now + b, 0.0) * rho
    return (v + c) ** 2 / (2.0 * b) - c * rho

  def stop_feasible(self, v: float, a_now: float, d: float, b: float) -> bool:
    return self.stop_distance(max(v, 0.0), a_now, b) <= d

  def envelope(self, v: float, d_line: float, stop: bool, b: float | None = None) -> float:
    """approach speed envelope (and the STOP profile): accel cap, never below -b (default a_b)"""
    p = self.p
    b = p.a_b if b is None else b
    x = d_line - p.line_margin - v * p.rho
    v_cap = math.sqrt(2.0 * b * max(x, 0.0))
    ff = -b * min(v / v_cap, 1.0) if v_cap > 1e-6 else -b
    a = p.k_v * (v_cap - v) + ff
    if stop:
      a = min(a, -v * v / (2.0 * max(d_line - p.line_margin, 0.3)) - 0.5 * (v - v_cap))
    return max(a, -b)

  # ---- unmarked protocol ----
  def compute_token(self, ego: EgoState, actors: list[Actor], signal: SignalHealth, now: float) -> tuple[bool, list[str]]:
    p = self.p
    if self.t_stop is None or self.t_stop_known is None or now - self.t_stop_known < TOKEN_SETTLE_S:
      return False, ["settling" if self.t_stop is not None else "no stop yet"]
    if self.token and self.state in (GO, COMMIT):
      return True, []
    entries = [(self.t_stop, self.ego_id)]
    for x in actors:
      if self.coop(x) and x.l3.get("t_stop") is not None:
        entries.append((float(x.l3["t_stop"]), x.id))
    keys = self.order(entries)
    me = keys[self.ego_id]
    why = []
    for x in actors:
      c = self.geom.conflicts.get(self.route, {}).get(x.route)
      if c is None:
        continue
      if x.source != "unknown" and x.s > c[3]:
        continue  # through its conflict with us
      if self.coop(x):
        st = x.l3.get("jx_state")
        if st == CLEARED:
          continue
        if st in (GO, COMMIT) or x.l3.get("token"):
          why.append(f"{x.id} going")
        elif x.id in keys and keys[x.id] < me and st in WAITING:
          why.append(f"{x.id} first")
      elif self.queued_stop(x) or x.source == "unknown":
        # non-cooperative waiting car: it goes first unless we clearly stopped before it (the gap test still protects us)
        t_seen = self.seen_stop.get(x.id)
        if x.source == "unknown" or t_seen is None or t_seen + p.stop_t <= self.t_stop + p.tie_s:
          why.append(f"{x.id} waiting")
    if not why:
      return True, []
    # deadlock breaker: nobody moved for deadlock_s -> the lowest (t_stop, id) whose gap test passes goes
    if self.last_move is not None and now - self.last_move >= p.deadlock_s:
      cands = [(keys[self.ego_id], self.ego_id, self.route, ego.s, ego.v)]
      for x in actors:
        if x.id in keys and self.coop(x) and x.l3.get("jx_state") in WAITING:
          cands.append((keys[x.id], x.id, x.route, x.s, x.v))
      ego_actor = Actor(self.ego_id, self.route, ego.x, ego.z, ego.psi, ego.v, ego.s, now, ego.a, ego.half_w, ego.half_l,
                        l3={"mode": ACTIVE, "jx_state": STOPPED, "t_stop": self.t_stop, "token": False})
      for _, cid, route, s, v in sorted(cands):
        others = [a for a in actors if a.id != cid] + ([ego_actor] if cid != self.ego_id else [])
        if self.gap_test(route, s, v, others, signal, p.pet_min)[0]:
          if cid == self.ego_id:
            self.breaker = True
            return True, ["deadlock breaker"]
          break
    return False, why

  # ---- step ----
  def step(self, ego: EgoState, actors: list[Actor], signal: SignalHealth, now: float) -> JunctionOut:
    p = self.p
    if self.state == CLEARED and ego.s < self.r.junction_s[0] - 2.0 * ego.half_l:
      # back before the junction after clearing it: the car was reset (routes are one way). A reset races the engine's physics
      # car, so a tick may see the previous trial's end-of-route pose and clear the fresh FSM: re-arm it
      self.reset()
    d_line = self.r.stop_line_s - (ego.s + ego.half_l)
    actors = [x for x in actors if x.id != self.ego_id]

    # bookkeeping: observed stops / braking of the other cars, nobody-moving timer, our own stop
    self.observe(actors, now)
    moving = ego.v >= p.stop_v or any(x.v >= p.stop_v and self.d_line_of(x) < MOVE_AREA_D and x.s < self.cmap.routes[x.route].junction_s[1]
                                      for x in actors)
    if self.last_move is None or moving:
      self.last_move = now
    if self.queued_stop(ego):
      if self.stop_since is None:
        self.stop_since = now
    elif self.t_stop is None:
      self.stop_since = None
    unmarked = signal.state == UNTRUSTED
    if unmarked and self.t_stop is None and self.stop_since is not None and now - self.stop_since >= p.stop_t:
      self.t_stop, self.t_stop_known = self.stop_since + p.stop_t, now

    def out(state: str, a_cap: float = INF, a_floor: float | None = None, hold: bool = False, reasons: list[str] | None = None) -> JunctionOut:
      self.state = state
      return JunctionOut(state, a_cap, a_floor, hold, self.em, self.token, self.t_stop, d_line, reasons or [])

    # through the box
    if self.state == CLEARED or ego.s > self.r.junction_s[1] + 2.0 * ego.half_l:
      self.em, self.token = False, False
      return out(CLEARED)
    if self.state == COMMIT:
      self.em = False
      reasons = ["commit"]
      return out(COMMIT, self.prof.fast_accel(ego.v), self.keep_floor(ego, actors, reasons), reasons=reasons)
    # emergency stop before the conflict zone: until stopped
    if self.em:
      if ego.v < p.hold_v:
        self.em = False
        return out(STOPPED, -p.a_b, None, True, ["em stopped"])
      return out(STOP_AT_LINE, -p.a_em, -p.a_em, False, ["emergency stop"])
    if d_line > p.approach_d:
      return out(FAR)

    # GO gate
    reasons: list[str] = []
    a_stop_need = ego.v * ego.v / (2.0 * max(d_line, 0.01))
    if signal.state == TRUSTED_GREEN:
      sig_ok = True
    elif self.state == GO and signal.state in (TRUSTED_AMBER, TRUSTED_RED, TRUSTED_RED_AMBER):
      # rule 175, already going: the change to amber (and the red after it) stops us unless stopping at the line would need more
      # than amber_stop_a, judged with the same lag model as the STOP profile. That stop may then use amber_stop_a + slack, so a
      # normal light change is never an emergency, and a car that goes on is not one that could have stopped (no red entries
      # from the amber dilemma zone, which a_b alone leaves open at 30 mph with a 3 s amber)
      sig_ok = d_line <= 0.0 or not self.stop_feasible(ego.v, ego.a, d_line, p.amber_stop_a)
      if not sig_ok:
        self.b_stop = max(self.b_stop, p.amber_stop_a + LIGHT_STOP_SLACK)
    elif signal.state == TRUSTED_AMBER:
      # rule 175: go on only if stopping would need more than amber_stop_a
      sig_ok = d_line <= 0.0 or (a_stop_need > p.amber_stop_a and self.b_stop <= p.a_b)  # once stopping for it, stay stopping
    elif unmarked:
      self.token, why = self.compute_token(ego, actors, signal, now)
      sig_ok = self.token
      reasons += ["unmarked"] + why
    else:
      sig_ok = False
    if not unmarked:
      self.token = False
    if not sig_ok:
      reasons.append(f"signal {signal.state}")
    go = sig_ok
    if go:
      pet = p.pet_min + (0.0 if self.state == GO else GRANT_PET_MARGIN)
      ok, who = self.gap_test(self.route, ego.s, ego.v, actors, signal, pet)
      if not ok:
        reasons.append(f"gap {who}")
      ok2, who2 = self.exit_clear(ego, actors)
      if not ok2:
        reasons.append(f"exit {who2}")
      go = ok and ok2

    stop_ok = self.stop_feasible(ego.v, ego.a, d_line, self.b_stop)
    em_ok = self.stop_feasible(ego.v, ego.a, self.cin_any - 1.0 - ego.s, p.a_em)
    if go:
      if not em_ok:
        reasons = ["commit"]
        return out(COMMIT, self.prof.fast_accel(ego.v), self.keep_floor(ego, actors, reasons), reasons=reasons)
      # launching from the line: keep it going (openpilot may not resume by itself)
      reasons = ["go"]
      floor = self.keep_floor(ego, actors, reasons) if d_line < QUEUE_D and ego.v < p.v_clear else None
      return out(GO, self.prof.fast_accel(ego.v), floor, reasons=reasons)

    # no GO
    if ego.v < p.hold_v and d_line < p.hold_d:
      self.b_stop = p.a_b
      return out(STOPPED, -p.a_b, None, True, reasons)
    if stop_ok:
      must_stop = not sig_ok
      return out(STOP_AT_LINE if must_stop else APPROACH, self.envelope(ego.v, d_line, must_stop, self.b_stop), None, False, reasons)
    # point of no return passed without GO
    if em_ok:
      self.em = True
      return out(STOP_AT_LINE, -p.a_em, -p.a_em, False, reasons + ["point of no return: emergency stop"])
    reasons.append("point of no return: commit")
    return out(COMMIT, self.prof.fast_accel(ego.v), self.keep_floor(ego, actors, reasons), reasons=reasons)
