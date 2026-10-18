import math
import os
from dataclasses import dataclass, field

import numpy as np

from openpilot.tools.sim.bridge.zoomengine.crossroads import CrossroadsMap, Route
from openpilot.tools.sim.bridge.zoomengine.scene import car_commands, deg
from openpilot.tools.sim.bridge.zoomengine.track import Track

# A road-kit town (ZoomEngine tools/roadkit/town_map.py -> assets/maps/town.built.json): four signalised crossroads joined
# by streets and looping "lobes". openpilot drives the four loop routes (straight through every junction); this module
# holds what the director needs for the rest of the town: the lane graph for NPC traffic, footways for pedestrians,
# per-junction UK signals with a pedestrian stage, and generated scenery (buildings, trees, street furniture, parked cars).

ROUTES = ("V_cw", "V_ccw", "H_cw", "H_ccw")
LANE_W = 3.65
# engine window layout (fractions of the window, from the top left): each car's road and wide camera side by side, cars 1-2
# along the top, 3-4 along the bottom, each car's status line next to its views; the town overview in the middle band
PIP_W = 0.24
PIP_H = PIP_W * 1208 / 1928 * 16 / 9      # camera aspect on a 16:9 window
PIP_ROW_Y = (0.02, 0.98 - PIP_H)
PIP_COL_X = (0.008, 0.252, 0.504, 0.748)  # car 1/3 road, wide; car 2/4 road, wide


def town_layout(car: int) -> tuple[list[float], list[float], list[float]]:
  """(road camera pip, wide camera pip, status line position) of town car 0-3"""
  row, half = car // 2, car % 2
  y = PIP_ROW_Y[row]
  road = [PIP_COL_X[2 * half], y, PIP_W, PIP_H]
  wide = [PIP_COL_X[2 * half + 1], y, PIP_W, PIP_H]
  info = [PIP_COL_X[2 * half], y + PIP_H + 0.008 if row == 0 else y - 0.035]
  return road, wide, info


FOOTWAY_D = 4.7          # footway centre from the carriageway centre line (kerb face 3.65, footway 2 m)
CROSSING_D = 10.8        # pedestrian crossings: across each junction arm, between the stop line (12) and the mouth (9.65)


class TownMap(CrossroadsMap):
  def __init__(self, built_json: str):
    super().__init__(built_json)
    # the openpilot routes are closed loops
    for name, r in self.data["routes"].items():
      track = Track.from_points(np.asarray(r["points"], dtype=float), step=0.5, closed=True)
      self.routes[name] = Route(name, r["phase"], track, float(r["stop_line_s"]), tuple(r["junction_s"]), self.routes[name].spawn_s)
    self.stops = {name: r.get("stops", []) for name, r in self.data["routes"].items()}
    self.left_hand = self.data.get("drive_side", "left") == "left"
    # the openpilot cars' routes, car 0 first: the map's list, or TOWN_CAR_ROUTES=C_cw,C_ccw (launch_town.sh CAR_ROUTES)
    override = os.environ.get("TOWN_CAR_ROUTES", "")
    self.car_routes = [r for r in override.split(",") if r] or list(self.data.get("car_routes", ROUTES))
    unknown = [r for r in self.car_routes if r not in self.routes]
    if unknown:
      raise ValueError(f"no route {unknown} in {built_json}: it has {sorted(self.routes)}")
    self.graph = LaneGraph(self.data["lane_graph"], self.left_hand)
    self.junctions = {}   # id -> centre (x, z)
    for st in self.data["statics"]:
      if st["section"].startswith("crossroads"):
        self.junctions[st["id"]] = (st["position"][0], st["position"][2])


# ---- lane graph (NPC traffic) ----

@dataclass
class Lane:
  id: str
  pts: np.ndarray          # (n, 2) x, z
  s: np.ndarray            # arc length at each point
  length: float
  next: list
  junction: str | None
  movement: str | None
  phase: str | None
  stop_s: float | None
  vmax: float              # comfortable speed from the tightest curvature (2.5 m/s^2 sideways)
  ys: np.ndarray           # road height at each point (bridges, ramps)


class LaneGraph:
  def __init__(self, data: dict, left_hand: bool = True):
    self.left_hand = left_hand
    self.lanes: dict[str, Lane] = {}
    for lid, d in data.items():
      p = np.asarray(d["points"], dtype=float)[:, [0, 2]]
      s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
      h = np.unwrap(np.arctan2(np.diff(p[:, 0]), np.diff(p[:, 1])))
      ds = np.maximum(np.diff(s)[1:], 1e-3)
      k = np.abs(np.diff(h)) / ds if len(h) > 1 else np.zeros(1)
      kmax = float(np.max(np.convolve(k, np.ones(5) / 5, mode="same"))) if len(k) else 0.0
      vmax = math.sqrt(2.5 / kmax) if kmax > 1e-4 else 99.0
      self.lanes[lid] = Lane(lid, p, s, float(s[-1]), list(d["next"]), d.get("junction"), d.get("movement"), d.get("phase"),
                             d.get("stop_s"), vmax, np.asarray(d["points"], dtype=float)[:, 1])

  def height(self, lid: str, s: float) -> tuple[float, float]:
    """(road height, pitch in radians, nose up positive) at arc length s along a lane"""
    ln = self.lanes[lid]
    s = min(max(s, 0.0), ln.length)
    y = float(np.interp(s, ln.s, ln.ys))
    ds = 2.0
    grade = (np.interp(min(s + ds, ln.length), ln.s, ln.ys) - np.interp(max(s - ds, 0.0), ln.s, ln.ys)) / (2 * ds)
    return y, math.atan(float(grade))

  def pose(self, lid: str, s: float) -> tuple[float, float, float]:
    """(x, z, heading) at arc length s along a lane"""
    ln = self.lanes[lid]
    s = min(max(s, 0.0), ln.length)
    i = int(np.clip(np.searchsorted(ln.s, s) - 1, 0, len(ln.s) - 2))
    t = (s - ln.s[i]) / max(ln.s[i + 1] - ln.s[i], 1e-6)
    p = ln.pts[i] + t * (ln.pts[i + 1] - ln.pts[i])
    d = ln.pts[i + 1] - ln.pts[i]
    return float(p[0]), float(p[1]), math.atan2(d[0], d[1])

  def streets(self) -> list[dict]:
    """every street between two junctions once: its centre line (from one direction's lanes) and the junctions at its ends"""
    out, seen = [], set()
    for ln in self.lanes.values():
      if ln.junction is None or ln.movement != "through":
        continue
      for nxt in ln.next:   # a through movement leaving a junction: follow the street to the next junction
        chain, cur = [], nxt
        while cur and self.lanes[cur].junction is None:
          chain.append(cur)
          cands = self.lanes[cur].next
          cur = cands[0] if cands else None
        if not chain or cur is None:
          continue
        key = frozenset(c.split(".")[0] for c in chain)
        if key in seen:
          continue
        seen.add(key)
        pts = np.concatenate([self.lanes[c].pts if i == 0 else self.lanes[c].pts[1:] for i, c in enumerate(chain)])
        t = np.gradient(pts, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        right = np.stack([-t[:, 1], t[:, 0]], axis=1)       # right of travel: (-cos, sin) for heading (sin, cos)
        # the street centre line: right of the lane in left-hand traffic (UK), left of it in right-hand traffic (US)
        centre = pts + right * (LANE_W / 2) * (1.0 if self.left_hand else -1.0)
        # elevated: any part of it on a bridge or ramp (no footway pedestrians or street furniture down at ground level)
        elevated = any(self.lanes[c].ys.max() > 1.0 for c in chain)
        out.append({"centre": centre, "left": -right, "from": ln.junction, "to": self.lanes[cur].junction, "elevated": elevated})
    return out


# ---- signals: two-phase cycle with a pedestrian stage, per junction ----

class JunctionSignals:
  """NS green -> amber -> all red -> PEDESTRIANS (all vehicle red) -> EW red+amber -> EW green -> amber -> all red -> NS
  red+amber. red_amber=0 (US: green, yellow, red only) leaves the red+amber steps out. offset staggers the junctions."""
  def __init__(self, offset: float = 0.0, green: float = 14.0, amber: float = 3.0, all_red: float = 2.0, ped: float = 10.0,
               red_amber: float = 2.0):
    self.offset = offset
    seq = [(green, "NS", "green"), (amber, "NS", "amber"), (all_red, None, "red"), (ped, "PED", "red"), (red_amber, "EW", "red_amber"),
           (green, "EW", "green"), (amber, "EW", "amber"), (all_red, None, "red"), (red_amber, "NS", "red_amber")]
    self.seq = [step for step in seq if step[0] > 0]
    self.cycle = sum(d for d, _, _ in self.seq)

  def state(self, t: float) -> tuple[dict[str, str], bool, float]:
    """({phase: aspect}, pedestrian stage, seconds left in it)"""
    t = (t + self.offset) % self.cycle
    for dur, phase, aspect in self.seq:
      if t < dur:
        if phase in ("NS", "EW"):
          other = "EW" if phase == "NS" else "NS"
          return {phase: aspect, other: "red"}, False, dur - t
        return {"NS": "red", "EW": "red"}, phase == "PED", dur - t
      t -= dur
    return {"NS": "red", "EW": "red"}, False, 0.0


# ---- NPC cars ----

@dataclass
class Npc:
  id: str
  kind: str
  lane: str
  s: float
  v: float
  v_des: float
  plan: list = field(default_factory=list)   # the lanes after the current one, chosen ahead
  wrecked: float = 0.0                         # seconds since a crash (0: fine)
  lat: float = 0.0                             # sideways offset after a crash
  lat_v: float = 0.0
  x: float = 0.0
  z: float = 0.0
  psi: float = 0.0


def idm(v: float, v0: float, gap: float, dv: float, a_max: float = 1.6, b: float = 2.5, s0: float = 2.5, T: float = 1.3) -> float:
  s_star = s0 + max(0.0, v * T + v * dv / (2 * math.sqrt(a_max * b)))
  return a_max * (1 - (v / max(v0, 0.1)) ** 4 - (s_star / max(gap, 0.1)) ** 2)


# ---- pedestrians ----

SKIN = ([0.87, 0.72, 0.6], [0.6, 0.45, 0.33], [0.36, 0.25, 0.18], [0.95, 0.82, 0.7])
CLOTHES = ([0.15, 0.2, 0.45], [0.55, 0.1, 0.1], [0.2, 0.35, 0.2], [0.85, 0.75, 0.3], [0.3, 0.3, 0.32], [0.9, 0.9, 0.88], [0.4, 0.2, 0.45])
TROUSERS = ([0.12, 0.12, 0.14], [0.2, 0.25, 0.4], [0.35, 0.3, 0.25], [0.15, 0.15, 0.3])
PED_MASS = 75.0


def pedestrian_commands(pid: str, x: float, z: float, psi: float, rng: np.random.Generator, h: float = 1.0) -> list[dict]:
  """a walker: torso box (the collision box, 75 kg) with head, and legs / arms hung from hip / shoulder pivots so they swing
  from the joint (rotate the '<id>_hip{l,r}' / '<id>_sh{l,r}' children about x)"""
  shirt = [float(c) for c in CLOTHES[rng.integers(len(CLOTHES))]]
  legs = [float(c) for c in TROUSERS[rng.integers(len(TROUSERS))]]
  skin = [float(c) for c in SKIN[rng.integers(len(SKIN))]]
  cmds = [{"action": "create", "type": "cube", "id": pid, "position": [x, 1.18 * h, z], "rotation": [0, deg(psi), 0],
           "scale": [0.2, 0.3 * h, 0.12], "color": shirt, "roughness": 0.9, "mass": PED_MASS}]

  def child(name, typ, pos, parent=pid, **kw):
    cmds.append({"action": "create", "type": typ, "id": f"{pid}_{name}", "parent": parent, "position": pos, **kw})

  child("head", "sphere", [0, 0.45 * h, 0], scale=[0.11, 0.13, 0.11], segments=12, color=skin, roughness=0.8)
  for side, sx in (("l", 0.1), ("r", -0.1)):
    child(f"hip{side}", "cube", [sx, -0.3 * h, 0], scale=[0.01, 0.01, 0.01], color=legs)
    child(f"leg{side}", "cylinder", [0, -0.42 * h, 0], parent=f"{pid}_hip{side}", radius=0.075, height=0.84 * h, segments=8, color=legs,
          roughness=0.9)
    child(f"sh{side}", "cube", [sx * 2.3, 0.26 * h, 0], scale=[0.01, 0.01, 0.01], color=shirt)
    child(f"arm{side}", "cylinder", [0, -0.3 * h, 0], parent=f"{pid}_sh{side}", radius=0.055, height=0.6 * h, segments=8, color=shirt,
          roughness=0.9)
  return cmds


@dataclass
class Ped:
  id: str
  street: int
  side: float              # +1 left footway of the street's direction, -1 right
  s: float                 # along the street's footway
  dir: float               # +1 / -1 along it
  speed: float
  state: str = "walk"      # walk | cross | down
  cross_t: float = 0.0     # crossing: progress 0..1
  cross_from: tuple = (0.0, 0.0)
  cross_to: tuple = (0.0, 0.0)
  timer: float = 0.0
  phase: float = 0.0       # gait phase
  h: float = 1.0           # height scale
  x: float = 0.0
  z: float = 0.0
  psi: float = 0.0


# ---- scenery ----

TOWER = ([0.62, 0.64, 0.66], [0.72, 0.7, 0.66], [0.32, 0.38, 0.44], [0.55, 0.5, 0.45], [0.82, 0.8, 0.76], [0.22, 0.26, 0.3])
GLASS = ([0.1, 0.14, 0.18], [0.12, 0.18, 0.22], [0.16, 0.2, 0.26])
DECK_UNDERSIDE_GAP = 0.9 + 0.8   # a building under a bridge stays this far below the road surface (slab + clearance)
BRICK = ([0.55, 0.27, 0.2], [0.62, 0.36, 0.26], [0.48, 0.3, 0.24], [0.78, 0.72, 0.62], [0.88, 0.85, 0.78], [0.6, 0.55, 0.5])
ROOF = ([0.25, 0.22, 0.22], [0.35, 0.22, 0.18], [0.2, 0.22, 0.26])
SHOP = ([0.15, 0.3, 0.55], [0.6, 0.12, 0.12], [0.12, 0.45, 0.25], [0.85, 0.6, 0.1], [0.4, 0.15, 0.45])


class Scenery:
  """buildings along every street (outside the footways, never on a road), street trees, lamp posts, bins, benches, post
  boxes, bollards, bus shelters, a roadworks pen, and car parks with parked cars. The map's "scenery_style": "town" (a
  row of houses and shops along each street) or "city" (a planned city: city_scenery.CityPlanner)"""
  def __init__(self, tmap: TownMap, rng: np.random.Generator):
    self.rng = rng
    self.cmds: list[dict] = []
    if tmap.data.get("scenery_style", "town") == "city":
      from openpilot.tools.sim.bridge.zoomengine.city_scenery import CityPlanner
      plan = CityPlanner(tmap, rng)
      self.cmds, self.boxes = plan.cmds, plan.boxes
      return
    pts = np.concatenate([np.column_stack([ln.pts, ln.ys]) for ln in tmap.graph.lanes.values()])
    high = pts[:, 2] > 3.0                          # bridge decks: buildings may stand under them, low enough
    self.road = pts[~high, :2]
    self.deck = pts[high]
    self.boxes: list[np.ndarray] = [self._rect(p[0], p[2], 0.0, 1.6, 1.6) for p in tmap.data.get("piers", [])]
    self.n = 0
    self.streets = [st for st in tmap.graph.streets() if not st["elevated"]]
    for st in self.streets:
      self._street(st)
    self._car_parks(tmap)

  def _deck_cap(self, poly: np.ndarray) -> float:
    """the tallest a building on this footprint may be: under a bridge deck, the deck's underside less a clearance"""
    if len(self.deck) == 0:
      return math.inf
    c = poly.mean(axis=0)
    r = float(np.max(np.linalg.norm(poly - c, axis=1)))
    near = self.deck[np.linalg.norm(self.deck[:, :2] - c, axis=1) < r + LANE_W / 2 + 2.3 + 2.0]
    return math.inf if len(near) == 0 else float(near[:, 2].min()) - DECK_UNDERSIDE_GAP

  def _id(self, tag: str) -> str:
    self.n += 1
    return f"sc_{tag}{self.n}"

  def _clear_of_road(self, poly: np.ndarray, margin: float) -> bool:
    c = poly.mean(axis=0)
    r = float(np.max(np.linalg.norm(poly - c, axis=1)))
    d = np.linalg.norm(self.road - c, axis=1)
    near = self.road[d < r + margin + 6.0]
    if len(near) == 0:
      return True
    # distance from road points to the polygon (sample its edges densely)
    edge = np.concatenate([np.linspace(poly[i], poly[(i + 1) % len(poly)], 12) for i in range(len(poly))] + [c[None]])
    return float(np.min(np.linalg.norm(near[:, None] - edge[None], axis=2))) > LANE_W / 2 + margin

  def _free(self, poly: np.ndarray) -> bool:
    from openpilot.tools.sim.bridge.zoomengine.crossroads import overlap
    return all(not overlap(poly, b) for b in self.boxes)

  @staticmethod
  def _rect(cx, cz, psi, hw, hl):
    f = np.array([math.sin(psi), math.cos(psi)])
    r = np.array([-math.cos(psi), math.sin(psi)])
    c = np.array([cx, cz])
    return np.array([c + f * hl + r * hw, c + f * hl - r * hw, c - f * hl - r * hw, c - f * hl + r * hw])

  def _street(self, st: dict) -> None:
    rng = self.rng
    c, left = st["centre"], st["left"]
    n = len(c)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    high_street = rng.random() < 0.4
    for side in (1.0, -1.0):
      pos = float(rng.uniform(6.0, 14.0))
      while pos < s[-1] - 10.0:
        i = int(np.searchsorted(s, pos))
        i = min(max(i, 1), n - 2)
        p, nrm = c[i], left[i] * side
        tang = c[i + 1] - c[i - 1]
        face = math.atan2(-nrm[0], -nrm[1])          # buildings face the street
        shop = high_street and rng.random() < 0.7
        w = float(rng.uniform(7.0, 11.0) if shop else rng.uniform(6.0, 9.0))
        dep = float(rng.uniform(8.0, 12.0))
        setback = 1.5 if shop else float(rng.uniform(3.0, 6.0))
        d = FOOTWAY_D + 1.0 + setback + dep / 2
        bc = p + nrm * d
        poly = self._rect(bc[0], bc[1], face, w / 2, dep / 2)
        if self._clear_of_road(poly, 3.0) and self._free(poly) and self._deck_cap(poly) > 7.5:
          self.boxes.append(poly)
          self._building(bc, face, w, dep, shop)
        # street furniture on this footway, at its back edge
        fp = p + nrm * (FOOTWAY_D + 0.75)
        k = rng.random()
        if k < 0.35:
          self._lamp(fp, face)
        elif k < 0.5:
          self._tree(p + nrm * (FOOTWAY_D + 1.6))
        elif k < 0.58:
          self._bin(fp)
        elif k < 0.64:
          self._bench(fp, face)
        elif k < 0.67:
          self._postbox(fp)
        elif k < 0.70 and np.linalg.norm(tang) > 0:
          self._bus_shelter(p + nrm * (FOOTWAY_D + 0.9), face)
        pos += w + float(rng.uniform(0.5, 4.0))
      # bollards where the street meets its junctions
      for i in (2, n - 3):
        for k in range(3):
          b = c[i] + left[i] * side * (FOOTWAY_D - 0.6) + (c[min(i + 1, n - 1)] - c[max(i - 1, 0)]) * 0.5 * (k - 1)
          self.cmds.append({"action": "create", "type": "cylinder", "id": self._id("bollard"), "position": [float(b[0]), 0.55, float(b[1])],
                            "radius": 0.09, "height": 0.9, "segments": 8, "color": [0.15, 0.15, 0.15], "roughness": 0.6})
    # a roadworks pen on some streets: barriers and cones on the verge behind the footway
    if rng.random() < 0.35:
      i = int(rng.integers(n // 3, 2 * n // 3))
      nrm = left[i] * (1 if rng.random() < 0.5 else -1)
      base = c[i] + nrm * (FOOTWAY_D + 2.2)
      tang = (c[i + 1] - c[i - 1]) / (np.linalg.norm(c[i + 1] - c[i - 1]) + 1e-9)
      for k in range(-3, 4):
        q = base + tang * k * 1.2
        self.cmds.append({"action": "create", "type": "cone", "id": self._id("cone"), "position": [float(q[0]), 0.35, float(q[1])],
                          "radius": 0.2, "height": 0.7, "segments": 10, "color": [1.0, 0.4, 0.05], "emissive": [0.3, 0.1, 0.0]})
      self.cmds.append({"action": "create", "type": "cube", "id": self._id("skip"), "position": [float(base[0] + nrm[0] * 2), 0.6, float(base[1] + nrm[1] * 2)],
                        "rotation": [0, deg(math.atan2(tang[0], tang[1])), 0], "scale": [0.9, 0.6, 1.8], "color": [0.85, 0.65, 0.05]})

  def _building(self, c, face, w, dep, shop):
    rng = self.rng
    h = float(rng.uniform(4.5, 6.5) if not shop else rng.uniform(6.0, 11.0))
    bid = self._id("shop" if shop else "house")
    col = [float(v) for v in (BRICK[rng.integers(len(BRICK))])]
    self.cmds.append({"action": "create", "type": "cube", "id": bid, "position": [float(c[0]), h / 2, float(c[1])], "rotation": [0, deg(face), 0],
                      "scale": [w / 2, h / 2, dep / 2], "color": col, "roughness": 0.9})

    def child(name, pos, scale, color, typ="cube", **kw):
      self.cmds.append({"action": "create", "type": typ, "id": f"{bid}_{name}", "parent": bid, "position": pos, "scale": scale,
                        "color": color, **kw})

    front = dep / 2 + 0.02
    if shop:
      sc = [float(v) for v in SHOP[rng.integers(len(SHOP))]]
      child("window", [0, -h / 2 + 1.6, front], [w / 2 - 0.6, 1.1, 0.03], [0.1, 0.13, 0.16], metallic=0.6, roughness=0.1)
      child("fascia", [0, -h / 2 + 3.1, front + 0.02], [w / 2 - 0.3, 0.35, 0.04], sc, roughness=0.6)
      child("awning", [0, -h / 2 + 2.75, front + 0.6], [w / 2 - 0.5, 0.05, 0.6], sc, rotation=[-12, 0, 0], roughness=0.8)
      for fl in range(1, int(h // 3)):
        for k in (-1, 0, 1):
          child(f"w{fl}{k}", [k * w * 0.3, -h / 2 + fl * 3.0 + 1.6, front], [0.55, 0.65, 0.03], [0.12, 0.15, 0.2], metallic=0.6, roughness=0.12)
      child("roof", [0, h / 2 + 0.25, 0], [w / 2, 0.25, dep / 2], [0.3, 0.3, 0.32], roughness=0.9)   # parapet
    else:
      child("door", [w * 0.25, -h / 2 + 1.05, front], [0.5, 1.05, 0.03], [0.1, 0.18, 0.35] if rng.random() < 0.5 else [0.45, 0.08, 0.08])
      for fl in range(int(h // 2.7)):
        for k in (-0.25, 0.0) if fl == 0 else (-0.25, 0.25):
          child(f"w{fl}{k}", [k * w, -h / 2 + 1.6 + fl * 2.7, front], [0.6, 0.6, 0.03], [0.12, 0.15, 0.2], metallic=0.6, roughness=0.12)
      # gable roof along the frontage (a square prism, its lower half inside the walls; it stops at the gable walls)
      r = (dep / 2 + 0.35) / math.sqrt(2.0)
      child("roof", [0, h / 2, 0], [w / 2 - 0.03, r, r], [float(v) for v in ROOF[rng.integers(len(ROOF))]], rotation=[45, 0, 0], roughness=0.8)
      if rng.random() < 0.5:
        child("chimney", [w * 0.3, h / 2 + dep * 0.3, -dep * 0.15], [0.35, 0.9, 0.35], [0.45, 0.25, 0.2])

  def _lamp(self, p, face):
    lid = self._id("lamp")
    self.cmds.append({"action": "create", "type": "cylinder", "id": lid, "position": [float(p[0]), 3.0, float(p[1])], "radius": 0.07, "height": 6.0,
                      "segments": 8, "color": [0.3, 0.31, 0.32], "metallic": 0.7, "roughness": 0.5})
    self.cmds.append({"action": "create", "type": "cube", "id": f"{lid}_head", "parent": lid, "position": [0, 3.0, 0], "rotation": [0, deg(face), 0],
                      "scale": [0.12, 0.06, 0.5], "color": [0.3, 0.31, 0.32]})

  def _tree(self, p):
    h = float(self.rng.uniform(5.0, 9.0))
    tid = self._id("tree")
    green = [float(self.rng.uniform(0.08, 0.16)), float(self.rng.uniform(0.24, 0.38)), float(self.rng.uniform(0.06, 0.12))]
    self.cmds.append({"action": "create", "type": "cylinder", "id": tid, "position": [float(p[0]), h * 0.3, float(p[1])], "radius": 0.18,
                      "height": h * 0.6, "segments": 8, "color": [0.3, 0.2, 0.12], "roughness": 0.95})
    r = h * 0.3
    self.cmds.append({"action": "create", "type": "sphere", "id": f"{tid}_crown", "position": [float(p[0]), h * 0.68, float(p[1])],
                      "scale": [r, r * 0.9, r], "segments": 12, "color": green, "roughness": 0.95})

  def _bin(self, p):
    self.cmds.append({"action": "create", "type": "cylinder", "id": self._id("bin"), "position": [float(p[0]), 0.45, float(p[1])], "radius": 0.28,
                      "height": 0.9, "segments": 12, "color": [0.12, 0.25, 0.15], "roughness": 0.7})

  def _bench(self, p, face):
    self.cmds.append({"action": "create", "type": "cube", "id": self._id("bench"), "position": [float(p[0]), 0.45, float(p[1])],
                      "rotation": [0, deg(face + math.pi / 2), 0], "scale": [0.3, 0.05, 0.9], "color": [0.45, 0.3, 0.18], "roughness": 0.9})

  def _postbox(self, p):
    self.cmds.append({"action": "create", "type": "cylinder", "id": self._id("postbox"), "position": [float(p[0]), 0.75, float(p[1])], "radius": 0.25,
                      "height": 1.5, "segments": 14, "color": [0.75, 0.04, 0.04], "roughness": 0.4})

  def _bus_shelter(self, p, face):
    sid = self._id("shelter")
    self.cmds.append({"action": "create", "type": "cube", "id": sid, "position": [float(p[0]), 2.4, float(p[1])], "rotation": [0, deg(face), 0],
                      "scale": [1.8, 0.06, 0.7], "color": [0.25, 0.27, 0.3], "metallic": 0.5})
    self.cmds.append({"action": "create", "type": "cube", "id": f"{sid}_back", "parent": sid, "position": [0, -1.2, -0.65],
                      "scale": [1.8, 1.15, 0.03], "color": [0.6, 0.7, 0.75], "metallic": 0.3, "roughness": 0.1})

  def _car_parks(self, tmap: TownMap) -> None:
    """an asphalt car park inside each block between the streets, part filled with parked cars"""
    js = list(tmap.junctions.values())
    xs, zs = sorted({j[0] for j in js}), sorted({j[1] for j in js})
    cx, cz = (xs[0] + xs[-1]) / 2, (zs[0] + zs[-1]) / 2
    lots = [(cx, cz)] + [(cx + dx, cz + dz) for dx, dz in ((150.0, 0.0), (-150.0, 0.0), (0.0, 150.0), (0.0, -150.0))]
    for k, (lx, lz) in enumerate(lots):
      pad = self._rect(lx, lz, 0.0, 22.0, 14.0)
      if not self._clear_of_road(pad, 4.0) or not self._free(pad):
        continue
      self.boxes.append(pad)
      self.cmds.append({"action": "create", "type": "cube", "id": f"sc_carpark{k}", "position": [lx, 0.005, lz], "scale": [22.0, 0.02, 14.0],
                        "color": [0.16, 0.16, 0.17], "roughness": 0.9})
      for row, rz in enumerate((-8.0, 8.0)):
        for slot in range(-6, 7):
          if self.rng.random() < 0.35:
            continue
          x = lx + slot * 3.0
          psi = 0.0 if row == 0 else math.pi
          col = [float(v) for v in (([0.75, 0.75, 0.78], [0.08, 0.08, 0.09], [0.55, 0.05, 0.05], [0.1, 0.2, 0.45], [0.9, 0.9, 0.92])
                                    [self.rng.integers(5)])]
          self.cmds += car_commands(f"sc_parked{k}_{row}_{slot}", "sedan" if self.rng.random() < 0.7 else "suv", col, x, lz + rz, psi)
