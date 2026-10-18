import json
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from openpilot.tools.sim.bridge.zoomengine.scene import CAR_COLORS, car_lite_commands, deg
from openpilot.tools.sim.bridge.zoomengine.town import BRICK, DECK_UNDERSIDE_GAP, GLASS, ROOF, SHOP, TOWER

# Scenery for a road-kit city (town.py Scenery with the map's "scenery_style": "city", e.g. assets/maps/city_us.built.json).
# A land-use planner rather than rows of buildings: a 2 m land grid splits the ground into blocks (downtown grid blocks, the
# lobes, the loop infield, the outer area); access corridors (alleys, mid-block cuts, courts, park links) are reserved
# first, then parcels are cut along every street side and filled by recipes (houses with driveways and back gardens,
# townhouse rows on a rear lane, downtown towers on paved forecourts, parks, surface lots, a school, retail...), then the
# interiors and the verges. Every building's door must reach the footway over paved flats (checked by a BFS: unreachable
# buildings are removed), nothing solid stands on a road or footway, near a bridge pier, too tall under a deck, or taller
# than 1 m where it would block a junction's sight lines.


def _m(c, r=0.9, m=0.0) -> dict:
  return {"color": [float(v) for v in c], "roughness": r, "metallic": m}


# palette: every entry a full material, few of them so identical shapes batch into one draw
ASPHALT = _m((0.17, 0.17, 0.18))
CONCRETE = _m((0.62, 0.61, 0.58))
PAVER = _m((0.55, 0.50, 0.45))
GRAVEL = _m((0.66, 0.60, 0.48))
LAWN = _m((0.30, 0.50, 0.22), 0.95)
PITCH = _m((0.22, 0.52, 0.20), 0.95)
SAND = _m((0.80, 0.72, 0.52), 0.95)
LINE = _m((0.92, 0.92, 0.90), 0.6)
WATER = _m((0.16, 0.30, 0.38), 0.05, 0.2)
TRUNK = _m((0.30, 0.20, 0.12), 0.95)
CROWN = (_m((0.12, 0.32, 0.10), 0.95), _m((0.16, 0.38, 0.12), 0.95))
CONIFER = _m((0.08, 0.24, 0.10), 0.95)
HEDGE = _m((0.12, 0.28, 0.10), 0.95)
SIDING = tuple(_m(c, 0.85) for c in ((0.88, 0.87, 0.82), (0.86, 0.80, 0.64), (0.55, 0.62, 0.68), (0.58, 0.64, 0.52), (0.70, 0.62, 0.50)))
BRICKS = (_m(BRICK[0]), _m(BRICK[1]))
ROOFS = tuple(_m(c, 0.8) for c in ROOF)
DOORS = (_m((0.10, 0.18, 0.35), 0.6), _m((0.45, 0.08, 0.08), 0.6), _m((0.10, 0.28, 0.16), 0.6))
TRIM = _m((0.92, 0.92, 0.90), 0.6)
FENCE_WOOD = _m((0.52, 0.38, 0.24))
FENCE_IRON = _m((0.1, 0.1, 0.1), 0.5, 0.6)
POOL = _m((0.3, 0.6, 0.8), 0.1)
TOWERS = tuple(_m(TOWER[i], 0.85) for i in (0, 1, 2, 4))
GLASSES = tuple(_m(GLASS[i], 0.1, 0.6) for i in (0, 1))
STONE = _m((0.70, 0.68, 0.62))
SHOPS = tuple(_m(c, 0.6) for c in SHOP)
PLAY = (SHOPS[1], SHOPS[3], SHOPS[0])
METAL = _m((0.30, 0.31, 0.32), 0.5, 0.7)
ROOFTOP = _m((0.28, 0.28, 0.30))
DARK = _m((0.06, 0.06, 0.07), 0.7, 0.3)
RUST = _m((0.62, 0.30, 0.22), 0.95)
CROPS = (_m((0.36, 0.52, 0.20), 0.95), _m((0.78, 0.68, 0.38), 0.95), _m((0.42, 0.32, 0.22), 0.95))
CARS = [[float(v) for v in c] for c in CAR_COLORS[:6]]

# flat layers (slab top y): never two materials at one height where they overlap
L0, L1, L2, L3 = 0.010, 0.020, 0.030, 0.040
WALK = ("path", "plaza", "lot")                  # flats people and cars use: the access network
SOFT = ("path", "lot", "water", "court")         # flats no prop or tree may stand on (by default)

CELL = 2.0               # land grid
JUNCTION_PAD = 26.0      # land grid: junction boxes blocked out to this half size
DP_MIN = 26.0           # house lots shallower than this become pocket parks
CORNER = 16.0            # frontages start this far from a junction box edge (corner plazas, lawns, sight lines)
SIGHT_JUNCTION = 15.0    # nothing solid taller than TALL within this of a junction box
SIGHT_BEND = 8.5         # ... nor this close to the centre line on the inside of a bend
TALL = 1.0
PIER_R = 0.7
PIER_CLEAR = 3.0
DECK_CLEAR = 0.4         # building tops this far below the deck cap
EXPRESS_HOMES = 30.0     # no dwelling this close to the expressway centre line
BUDGET = (15000, 18000)  # objects on city_us: the engine's draw budget for scenery
TREE_SIZES = ((6.5, 1.6), (7.5, 2.0), (9.0, 2.4))   # (height, crown radius): crown bottom >= 3.2 m over footways


# ---- geometry (x, z) ----

def _ccw(poly: np.ndarray) -> np.ndarray:
  x, z = poly[:, 0], poly[:, 1]
  return poly if float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)) > 0 else poly[::-1].copy()


def _inside(pts: np.ndarray, poly: np.ndarray, margin: float = 0.0) -> np.ndarray:
  """points at least `margin` inside a convex ccw polygon (negative margin: that far outside still counts)"""
  e = np.roll(poly, -1, axis=0) - poly
  cr = e[None, :, 0] * (pts[:, None, 1] - poly[None, :, 1]) - e[None, :, 1] * (pts[:, None, 0] - poly[None, :, 0])
  return np.all(cr / np.linalg.norm(e, axis=1)[None] >= margin - 1e-9, axis=1)


def _dist(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
  """distance from each point to a convex ccw polygon (0 inside)"""
  if len(pts) == 0:
    return np.zeros(0)
  ab = np.roll(poly, -1, axis=0) - poly
  ap = pts[:, None, :] - poly[None]
  t = np.clip(np.einsum("nki,ki->nk", ap, ab) / np.einsum("ki,ki->k", ab, ab), 0.0, 1.0)
  d = np.linalg.norm(ap - t[..., None] * ab[None], axis=2).min(axis=1)
  d[_inside(pts, poly)] = 0.0
  return d


def _poly_dist(a: np.ndarray, b: np.ndarray) -> float:
  if not _sep(a, b):
    return 0.0
  return float(min(_dist(a, b).min(), _dist(b, a).min()))


def _sep(a: np.ndarray, b: np.ndarray, gap: float = 0.0) -> bool:
  """separating axis test of two convex polygons: True when apart by more than gap (touching counts as apart at gap 0).
  (crossroads.overlap takes quads only and counts touching as overlap: shared walls and bulbs need this)"""
  for poly in (a, b):
    e = np.roll(poly, -1, axis=0) - poly
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    pa, pb = a @ n.T, b @ n.T
    if np.any(pa.max(axis=0) + gap <= pb.min(axis=0) + 1e-6) or np.any(pb.max(axis=0) + gap <= pa.min(axis=0) + 1e-6):
      return True
  return False


def _rect(c, psi: float, hx: float, hz: float) -> np.ndarray:
  """footprint of a box at c turned psi (engine rotation y): local x half hx, local z half hz"""
  ex = np.array([math.cos(psi), -math.sin(psi)]) * hx
  ez = np.array([math.sin(psi), math.cos(psi)]) * hz
  c = np.asarray(c, dtype=float)
  return np.array([c + ex + ez, c - ex + ez, c - ex - ez, c + ex - ez])


def _octagon(c, r: float) -> np.ndarray:
  a = np.arange(8) * math.pi / 4 + math.pi / 8
  return np.asarray(c, dtype=float) + np.stack([np.cos(a), np.sin(a)], axis=1) * r / math.cos(math.pi / 8)


def _nearest_on(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
  ab = np.roll(poly, -1, axis=0) - poly
  t = np.clip(np.einsum("ki,ki->k", p[None] - poly, ab) / np.einsum("ki,ki->k", ab, ab), 0.0, 1.0)
  q = poly + t[:, None] * ab
  return q[int(np.argmin(np.linalg.norm(q - p, axis=1)))]


class Frame:
  """a local frame on the ground: a across (ex), d deeper (v, away from the street); objects turned `face` look back
  along -v (their local +z), so a child at local z = -dd sits dd deeper"""
  def __init__(self, o, v):
    self.o = np.asarray(o, dtype=float)
    v = np.asarray(v, dtype=float)
    self.v = v / np.linalg.norm(v)
    self.face = math.atan2(-self.v[0], -self.v[1])
    self.ex = np.array([math.cos(self.face), -math.sin(self.face)])

  def p(self, a: float, d: float) -> np.ndarray:
    return self.o + a * self.ex + d * self.v

  def rect(self, a0: float, a1: float, d0: float, d1: float) -> np.ndarray:
    return _ccw(np.array([self.p(a0, d0), self.p(a1, d0), self.p(a1, d1), self.p(a0, d1)]))

  def sub(self, a: float, d: float, turn: float = 0.0) -> "Frame":
    """a frame at (a, d) turned by `turn` (radians, +pi/2: its v along this frame's -ex)"""
    v = self.v * math.cos(turn) - self.ex * math.sin(turn)
    return Frame(self.p(a, d), v)


class _PointHash:
  def __init__(self, pts: np.ndarray, cell: float = 8.0):
    self.pts, self.cell = pts, cell
    b = defaultdict(list)
    for i, k in enumerate(map(tuple, np.floor(pts[:, :2] / cell).astype(int))):
      b[k].append(i)
    self.b = {k: np.array(v) for k, v in b.items()}

  def near(self, lo, hi, r: float) -> np.ndarray:
    i0, j0 = (int(math.floor(v)) for v in (np.asarray(lo) - r) / self.cell)
    i1, j1 = (int(math.floor(v)) for v in (np.asarray(hi) + r) / self.cell)
    idx = [self.b[(i, j)] for i in range(i0, i1 + 1) for j in range(j0, j1 + 1) if (i, j) in self.b]
    return self.pts[np.concatenate(idx)] if idx else self.pts[:0]


class Shape:
  __slots__ = ("poly", "lo", "hi", "cat", "purpose", "y", "key", "top", "g", "alive", "crown")

  def __init__(self, poly, cat, g, top=0.0, purpose=None, y=None, key=None, crown=None):
    self.poly, self.cat, self.g, self.top = poly, cat, g, top
    self.lo, self.hi = poly.min(axis=0), poly.max(axis=0)
    self.purpose, self.y, self.key, self.crown, self.alive = purpose, y, key, crown, True


class _Index:
  """20 m buckets of shapes"""
  def __init__(self, cell: float = 20.0):
    self.cell, self.b = cell, defaultdict(list)

  def _keys(self, lo, hi):
    for i in range(int(math.floor(lo[0] / self.cell)), int(math.floor(hi[0] / self.cell)) + 1):
      for j in range(int(math.floor(lo[1] / self.cell)), int(math.floor(hi[1] / self.cell)) + 1):
        yield (i, j)

  def add(self, sh: Shape) -> None:
    for k in self._keys(sh.lo, sh.hi):
      self.b[k].append(sh)

  def query(self, lo, hi):
    seen = set()
    for k in self._keys(lo, hi):
      for sh in self.b.get(k, ()):
        if sh.alive and id(sh) not in seen and sh.lo[0] < hi[0] and sh.hi[0] > lo[0] and sh.lo[1] < hi[1] and sh.hi[1] > lo[1]:
          seen.add(id(sh))
          yield sh


@dataclass
class Group:
  kind: str
  parcel: object = None
  cmds: list = field(default_factory=list)
  shapes: list = field(default_factory=list)
  doors: list = field(default_factory=list)
  alive: bool = True


@dataclass
class Parcel:
  side: object
  s0: float
  s1: float
  depth: float
  quad: np.ndarray
  frame: Frame
  width: float
  kind: str
  rng: np.random.Generator
  theme: dict
  gap: float               # the access flats start this far in front of the parcel front (over the verge)
  index: int = 0
  corner: bool = False
  drive_a: float | None = None


# ---- land grid ----

class LandGrid:
  """2 m raster over the ground (less 10 m): road (lane discs + junction squares), deck caps, block labels by 4-connected
  flood fill; and a 1 m occupancy raster of what has been reserved (corridors, parcels, interiors)"""
  def __init__(self, ground: dict, low: np.ndarray, high: np.ndarray, junctions, road_r: float, deck_r: float):
    cx, cz = float(ground["center"][0]), float(ground["center"][2])
    w, d = float(ground["size"][0]), float(ground["size"][1])
    self.x0, self.z0 = cx - w / 2 + 10.0, cz - d / 2 + 10.0
    self.nx, self.nz = int((w - 20.0) // CELL), int((d - 20.0) // CELL)
    road = np.zeros((self.nz, self.nx), bool)
    for (i, j), _ in self._stamp(low, road_r):
      road[j, i] = True
    for jx, jz in junctions:
      i0, j0 = self._ij(np.array([[jx - JUNCTION_PAD, jz - JUNCTION_PAD]]))[0]
      i1, j1 = self._ij(np.array([[jx + JUNCTION_PAD, jz + JUNCTION_PAD]]))[0]
      road[max(j0, 0):j1 + 1, max(i0, 0):i1 + 1] = True
    self.road = road
    self.cap = np.full((self.nz, self.nx), np.inf)
    if len(high):
      for (i, j), k in self._stamp(high[:, :2], deck_r):
        np.minimum.at(self.cap, (j, i), high[k, 2] - DECK_UNDERSIDE_GAP)
    self.lab, self.n_labels, self.edge = self._label(~road)
    self.occ = np.zeros((self.nz * 2, self.nx * 2), bool)

  def _ij(self, pts: np.ndarray) -> np.ndarray:
    return np.floor((pts - [self.x0, self.z0]) / CELL).astype(int)

  def _stamp(self, pts: np.ndarray, r: float):
    """cells within r of any point: yields ((i, j) arrays, point index array) per disc offset"""
    ij = self._ij(pts)
    k = np.arange(len(pts))
    n = int(math.ceil(r / CELL))
    for di in range(-n, n + 1):
      for dj in range(-n, n + 1):
        if (di * CELL) ** 2 + (dj * CELL) ** 2 > r * r:
          continue
        i, j = ij[:, 0] + di, ij[:, 1] + dj
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.nz)
        yield (i[ok], j[ok]), k[ok]

  def _label(self, free: np.ndarray):
    lab = np.zeros(free.shape, int)
    k = 0
    for j0, i0 in zip(*np.nonzero(free), strict=True):
      if lab[j0, i0]:
        continue
      k += 1
      lab[j0, i0] = k
      q = deque([(j0, i0)])
      while q:
        j, i = q.popleft()
        for jj, ii in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
          if 0 <= jj < self.nz and 0 <= ii < self.nx and free[jj, ii] and not lab[jj, ii]:
            lab[jj, ii] = k
            q.append((jj, ii))
    cnt = np.bincount(lab.ravel(), minlength=k + 1)
    small = cnt * CELL * CELL < 400.0
    small[0] = True
    lab[small[lab]] = 0
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist()) - {0}
    return lab, k, edge

  def lab_at(self, pts: np.ndarray) -> np.ndarray:
    ij = self._ij(pts.reshape(-1, 2))
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.nz)
    out = np.zeros(len(ij), int)
    out[ok] = self.lab[ij[ok, 1], ij[ok, 0]]
    return out.reshape(pts.shape[:-1])

  def cap_at(self, pts: np.ndarray) -> np.ndarray:
    ij = self._ij(pts.reshape(-1, 2))
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.nz)
    out = np.full(len(ij), np.inf)
    out[ok] = self.cap[ij[ok, 1], ij[ok, 0]]
    return out.reshape(pts.shape[:-1])

  def occ_at(self, pts: np.ndarray) -> np.ndarray:
    ij = np.floor(pts.reshape(-1, 2) - [self.x0, self.z0]).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx * 2) & (ij[:, 1] >= 0) & (ij[:, 1] < self.nz * 2)
    out = np.ones(len(ij), bool)
    out[ok] = self.occ[ij[ok, 1], ij[ok, 0]]
    return out.reshape(pts.shape[:-1])

  def reserve(self, poly: np.ndarray, pad: float = 0.0) -> None:
    lo = np.floor(poly.min(axis=0) - pad - [self.x0, self.z0]).astype(int)
    hi = np.ceil(poly.max(axis=0) + pad - [self.x0, self.z0]).astype(int)
    lo, hi = np.maximum(lo, 0), np.minimum(hi, [self.nx * 2, self.nz * 2])
    if np.any(hi <= lo):
      return
    ii, jj = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
    c = np.stack([ii.ravel() + 0.5 + self.x0, jj.ravel() + 0.5 + self.z0], axis=1)
    m = _inside(c, poly, -pad - 0.5)
    self.occ[jj.ravel()[m], ii.ravel()[m]] = True

  def free_mask(self, label: int, erode: int = 1) -> np.ndarray:
    """the label's 2 m cells nothing has reserved, eroded"""
    o = self.occ.reshape(self.nz, 2, self.nx, 2).any(axis=(1, 3))
    m = (self.lab == label) & ~o
    for _ in range(erode):
      e = m.copy()
      e[1:] &= m[:-1]
      e[:-1] &= m[1:]
      e[:, 1:] &= m[:, :-1]
      e[:, :-1] &= m[:, 1:]
      m = e
    return m

  def mask_at(self, mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ij = self._ij(pts.reshape(-1, 2))
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.nz)
    out = np.zeros(len(ij), bool)
    out[ok] = mask[ij[ok, 1], ij[ok, 0]]
    out &= ~self.occ_at(pts.reshape(-1, 2))
    return out.reshape(pts.shape[:-1])

  def cells(self, mask: np.ndarray) -> np.ndarray:
    j, i = np.nonzero(mask)
    return np.stack([self.x0 + (i + 0.5) * CELL, self.z0 + (j + 0.5) * CELL], axis=1)

  def fit_rect(self, mask: np.ndarray, frame: Frame, sizes, target=None, step: int = 4):
    """the largest of sizes [(across, deep)] (in that order) that fits in mask, axes along the frame, centres on a
    `step` m lattice, the one nearest target: (centre frame (a, d both centred), across, deep) or None"""
    pts = self.cells(mask)
    if len(pts) == 0:
      return None
    rel = pts - frame.o
    a, d = rel @ frame.ex, rel @ frame.v
    a0, d0 = math.floor(a.min()) - 2, math.floor(d.min()) - 2
    na, nd = int(math.ceil(a.max() - a0)) + 3, int(math.ceil(d.max() - d0)) + 3
    aa, dd = np.meshgrid(a0 + 0.5 + np.arange(na), d0 + 0.5 + np.arange(nd))
    world = frame.o + aa[..., None] * frame.ex + dd[..., None] * frame.v
    f = self.mask_at(mask, world).astype(np.int32)
    s = np.zeros((nd + 1, na + 1), np.int64)
    s[1:, 1:] = f.cumsum(axis=0).cumsum(axis=1)
    tgt = None if target is None else np.array([(target - frame.o) @ frame.ex, (target - frame.o) @ frame.v])
    for w, h in sizes:
      w, h = int(round(w)), int(round(h))
      if w > na or h > nd:
        continue
      win = s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w]
      vj, vi = np.nonzero(win == w * h)
      keep = ((vi + a0) % step == 0) & ((vj + d0) % step == 0)
      vj, vi = vj[keep], vi[keep]
      if len(vi) == 0:
        continue
      ca, cd = a0 + vi + w / 2.0, d0 + vj + h / 2.0
      k = 0 if tgt is None else int(np.argmin((ca - tgt[0]) ** 2 + (cd - tgt[1]) ** 2))
      return Frame(frame.p(float(ca[k]), float(cd[k])), frame.v), float(w), float(h)
    return None


# ---- street sides ----

class Side:
  """one side of a street (or a court): its centre line, the normal to this side, arc length, and the frontage line
  `front` m out (the back of the verge on a street)"""
  def __init__(self, c: np.ndarray, n: np.ndarray, front: float, gap: float, name: str):
    keep = np.concatenate([[True], np.linalg.norm(np.diff(c, axis=0), axis=1) > 1e-6])
    c, n = c[keep], n[keep]
    self.c, self.n, self.front, self.gap, self.name = c, n / np.linalg.norm(n, axis=1, keepdims=True), front, gap, name
    self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    self.len = float(self.s[-1])
    t = np.gradient(c, self.s, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    k = np.einsum("ij,ij->i", np.gradient(t, self.s, axis=0), self.n)
    self.k = np.convolve(k, np.ones(9) / 9, mode="same")    # curvature, > 0 where this side is the inside of a bend
    self.curved = bool(np.max(np.abs(self.k)) > 1 / 300)
    self.label = 0
    self.zone = ""
    self.kind = ""
    self.prog = ""
    self.theme: dict = {}
    self.lots: list = []
    self.street = None

  def at(self, s, d=0.0):
    s = np.clip(np.asarray(s, dtype=float), 0.0, self.len)
    p = np.stack([np.interp(s, self.s, self.c[:, 0]), np.interp(s, self.s, self.c[:, 1])], axis=-1)
    n = np.stack([np.interp(s, self.s, self.n[:, 0]), np.interp(s, self.s, self.n[:, 1])], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return p + n * np.asarray(d, dtype=float)[..., None], n

  def frame(self, s: float) -> Frame:
    p, n = self.at(s, self.front)
    return Frame(p, n)


def _profile_values(tmap) -> dict:
  v = {"lane_width": 3.66, "footway_width": 1.524, "junction_half_size": 20.11}
  path = os.path.join(tmap.root, "assets", "roadkit", tmap.data.get("profile", "us"), "library.json")
  if os.path.exists(path):
    with open(path) as f:
      v.update(json.load(f).get("values", {}))
  return v


# ---- the planner ----

class CityPlanner:
  def __init__(self, tmap, rng: np.random.Generator, verbose: bool = False):
    self.tmap, self.rng, self.verbose = tmap, rng, verbose
    v = _profile_values(tmap)
    self.lane_half = v["lane_width"] / 2
    self.footway = v["footway_width"]
    self.foot = v["lane_width"] + v["footway_width"]     # back of the footway from the street centre line
    self.verge = self.foot + 2.0                          # back of the verge: parcels start here
    self.jh = v["junction_half_size"]
    pts = np.concatenate([np.column_stack([ln.pts, ln.ys]) for ln in tmap.graph.lanes.values()])
    self.low = pts[pts[:, 2] < 3.0, :2]
    self.high = pts[pts[:, 2] >= 3.0]
    self.low_hash = _PointHash(self.low)
    self.high_hash = _PointHash(self.high)
    self.deck_r = self.lane_half + self.footway + 1.25    # deck edge (lane centre + half lane + footway/parapet) + 1 m
    self.piers = np.array([[p[0], p[2]] for p in tmap.data.get("piers", [])], dtype=float).reshape(-1, 2)
    self.jcentres = list(tmap.junctions.values())
    rot = {st["id"]: math.radians(st.get("rotation", [0, 0, 0])[1]) for st in tmap.data["statics"]}
    self.jboxes = [_rect(c, -rot.get(j, 0.0), self.jh, self.jh) for j, c in tmap.junctions.items()]
    streets = tmap.graph.streets()
    self.streets = [st for st in streets if not st["elevated"]]
    ex = [st["centre"] for st in streets if st["elevated"]]
    self.express_hash = _PointHash(np.concatenate(ex) if ex else np.zeros((0, 2)))
    self.centre_hash = _PointHash(np.concatenate([st["centre"] for st in self.streets]))
    self.index = _Index()
    self.groups: list[Group] = []
    self.g = -1
    self.n = 0
    self.stats: dict = defaultdict(int)
    self.notes: list[str] = []
    self.r = rng
    self.block_plazas: dict = {}
    self._grid()
    self._sides()
    self._corridors()
    self._frontages()
    self._interiors()
    self._verges()
    self._validate()
    self.cmds = [c for gr in self.groups if gr.alive for c in gr.cmds]
    self.boxes = [sh.poly for gr in self.groups if gr.alive for sh in gr.shapes if sh.cat == "bldg"]
    budget = "" if BUDGET[0] <= len(self.cmds) <= BUDGET[1] else f" (outside the {BUDGET[0]}-{BUDGET[1]} budget)"
    n_bldg = sum(1 for gr in self.groups if gr.alive and gr.doors)
    print(f"city scenery: {len(self.cmds)} objects{budget}, {n_bldg} buildings, {self.removed} unreachable removed, " +
          f"rule violations {self.violations or 'none'}, solid overlaps {self.overlaps}", flush=True)

  # ---- bookkeeping ----

  def _id(self, tag: str) -> str:
    self.n += 1
    return f"sc_{tag}{self.n}"

  def _begin(self, kind: str, parcel=None) -> int:
    self.groups.append(Group(kind, parcel))
    self.g = len(self.groups) - 1
    return self.g

  def _abort(self, g: int) -> None:
    gr = self.groups[g]
    for sh in gr.shapes:
      sh.alive = False
    gr.shapes, gr.cmds, gr.doors, gr.alive = [], [], [], False

  def _cmds(self, cmds: list) -> None:
    """commands with no footprint of their own (e.g. cars on a garage roof)"""
    self.groups[self.g].cmds += cmds

  # ---- hard rules ----

  def _road_d(self, poly: np.ndarray) -> float:
    p = self.low_hash.near(poly.min(axis=0), poly.max(axis=0), 8.0)
    return float(_dist(p, poly).min()) if len(p) else math.inf

  def _cap(self, poly: np.ndarray) -> float:
    p = self.high_hash.near(poly.min(axis=0), poly.max(axis=0), self.deck_r)
    if len(p) == 0:
      return math.inf
    near = p[_dist(p[:, :2], poly) < self.deck_r]
    return float(near[:, 2].min()) - DECK_UNDERSIDE_GAP if len(near) else math.inf

  def _pier_d(self, poly: np.ndarray) -> float:
    if len(self.piers) == 0:
      return math.inf
    return float(_dist(self.piers, poly).min())

  def _junction_d(self, poly: np.ndarray) -> float:
    best = math.inf
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    for b in self.jboxes:
      blo, bhi = b.min(axis=0), b.max(axis=0)
      if np.all(lo < bhi + SIGHT_JUNCTION) and np.all(hi > blo - SIGHT_JUNCTION):
        best = min(best, _poly_dist(poly, b))
    return best

  def _bend_ok(self, poly: np.ndarray) -> bool:
    """not within SIGHT_BEND of a street centre line on the inside of a bend"""
    for sd in self.bend_sides:
      lo, hi = poly.min(axis=0), poly.max(axis=0)
      if np.any(lo > sd.bhi + SIGHT_BEND) or np.any(hi < sd.blo - SIGHT_BEND):
        continue
      d = _dist(sd.bend_c, poly)
      near = d < SIGHT_BEND
      if np.any(near) and np.any(np.einsum("ij,ij->i", poly.mean(axis=0) - sd.bend_c[near], sd.bend_n[near]) > 0):
        return False
    return True

  def _hard(self, poly, cat: str, top: float, crown=None) -> str | None:
    """the first hard rule a shape breaks, or None"""
    if cat == "flat":       # (a slab may run round a pier: lots under the deck)
      return "road" if self._road_d(poly) < self.lane_half + self.footway - 0.01 else None
    if self._road_d(poly) <= self.lane_half + 2.0:
      return "road"
    if self._pier_d(poly) < PIER_R + PIER_CLEAR:
      return "pier"
    cap = self._cap(crown if crown is not None else poly)
    if top > cap - DECK_CLEAR or (cat == "tree" and cap < math.inf):
      return "deck"
    if top > TALL:
      if self._junction_d(poly) < SIGHT_JUNCTION:
        return "junction"
      if not self._bend_ok(poly):
        return "bend"
    return None

  def _collides(self, poly, cat, purpose=None, y=None, key=None, avoid=SOFT, allow=("lawn",), crown=None) -> bool:
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    if crown is not None:
      lo, hi = np.minimum(lo, crown.min(axis=0)), np.maximum(hi, crown.max(axis=0))
    for sh in self.index.query(lo, hi):
      p = poly
      if sh.cat == "flat":
        if cat == "flat":
          water = "water" in (purpose, sh.purpose)       # nothing runs under a pond but grass
          if (sh.y != y or sh.key == key) and not (water and "lawn" not in (purpose, sh.purpose)):
            continue
        elif cat == "bldg":
          if sh.purpose in allow:
            continue
        elif sh.purpose not in avoid:
          continue
      elif cat == "flat":
        if sh.cat != "bldg" and purpose not in SOFT:
          continue
      elif crown is not None and sh.cat == "bldg":
        p = crown
      if not _sep(p, sh.poly):
        return True
    return False

  def _add(self, poly, cat: str, cmds: list, top: float = 0.0, purpose=None, y=None, key=None, avoid=SOFT, allow=("lawn",),
           within=None, crown=None, door=None, check=True) -> bool:
    poly = _ccw(np.asarray(poly, dtype=float))
    if check:
      if within is not None and not np.all(_inside(poly, within, -0.05)):
        self.stats["reject_within"] += 1
        return False
      why = self._hard(poly, cat, top, crown)
      if why:
        self.stats["reject_" + why] += 1
        return False
      if self._collides(poly, cat, purpose, y, key, avoid, allow, crown):
        self.stats["reject_overlap"] += 1
        return False
    sh = Shape(poly, cat, self.g, top, purpose, y, key, crown)
    self.index.add(sh)
    gr = self.groups[self.g]
    gr.shapes.append(sh)
    gr.cmds += cmds
    if door is not None:
      gr.doors.append(np.asarray(door, dtype=float))
    return True

  # ---- primitives ----

  @staticmethod
  def _box(oid, p, y, hx, hy, hz, psi, mat, **kw) -> dict:
    return {"action": "create", "type": "cube", "id": oid, "position": [round(float(p[0]), 3), round(y, 3), round(float(p[1]), 3)],
            "rotation": [0, deg(psi), 0], "scale": [round(hx, 3), round(hy, 3), round(hz, 3)], **mat, **kw}

  @staticmethod
  def _kid(pid, name, typ, pos, mat, **kw) -> dict:
    return {"action": "create", "type": typ, "id": f"{pid}_{name}", "parent": pid, "position": [round(float(v), 3) for v in pos], **mat, **kw}

  @staticmethod
  def _cyl(oid, p, y, r, h, mat, seg=12, **kw) -> dict:
    return {"action": "create", "type": "cylinder", "id": oid, "position": [round(float(p[0]), 3), round(y, 3), round(float(p[1]), 3)],
            "radius": r, "height": h, "segments": seg, **mat, **kw}

  def _flat(self, F: Frame, a0, a1, d0, d1, layer, mat, purpose, tag="pave", within=None, check=True) -> bool:
    poly = F.rect(a0, a1, d0, d1)
    c = F.p((a0 + a1) / 2, (d0 + d1) / 2)
    cmd = self._box(self._id(tag), c, layer - 0.01, (a1 - a0) / 2, 0.01, (d1 - d0) / 2, F.face, mat)
    return self._add(poly, "flat", [cmd], purpose=purpose, y=layer, key=tuple(mat["color"]), within=within, check=check)

  def _strip(self, p0, p1, width, layer, mat, purpose, tag="path", check=True) -> bool:
    """a straight flat from p0 to p1"""
    d = np.asarray(p1, dtype=float) - p0
    ln = float(np.linalg.norm(d))
    if ln < 0.3:
      return False
    psi = math.atan2(d[0], d[1])
    c = (np.asarray(p0) + p1) / 2
    poly = _rect(c, psi, width / 2, ln / 2)
    cmd = self._box(self._id(tag), c, layer - 0.01, width / 2, 0.01, ln / 2, psi, mat)
    return self._add(poly, "flat", [cmd], purpose=purpose, y=layer, key=tuple(mat["color"]), check=check)

  def _prop_box(self, tag, p, psi, hx, hy, hz, mat, y0=0.0, kids=(), avoid=SOFT, within=None, cat="prop", door=None, allow=("lawn",)):
    oid = self._id(tag)
    cmds = [self._box(oid, p, y0 + hy, hx, hy, hz, psi, mat)] + [self._kid(oid, *k[:4], **(k[4] if len(k) > 4 else {})) for k in kids]
    ok = self._add(_rect(p, psi, hx, hz), cat, cmds, top=y0 + 2 * hy + (max((k[2][1] for k in kids), default=0.0) if kids else 0.0),
                   avoid=avoid, within=within, door=door, allow=allow)
    return oid if ok else None

  def _tree(self, p, size=None, conifer=False, avoid=SOFT, within=None) -> bool:
    rng = self.r
    h, r = TREE_SIZES[int(rng.integers(3)) if size is None else size]
    tid = self._id("tree")
    th = round(h * (0.3 if conifer else 0.55), 2)
    cmds = [self._cyl(tid, p, th / 2, 0.18, th, TRUNK, seg=8)]
    if conifer:
      hc = round(h * 0.8, 2)
      cmds.append(self._kid(tid, "crown", "cone", [0, h - hc / 2 - th / 2, 0], CONIFER, radius=r, height=hc, segments=10))
    else:
      cmds.append(self._kid(tid, "crown", "sphere", [0, h - r * 0.9 - th / 2, 0], CROWN[int(rng.integers(2))], scale=[r, r * 0.9, r],
                            segments=10))
    p = np.asarray(p, dtype=float)
    return self._add(_rect(p, 0.0, 0.5, 0.5), "tree", cmds, top=h, avoid=avoid, within=within, crown=_rect(p, 0.0, r, r))

  def _lamp(self, p, face, h=6.0, park=False) -> bool:
    lid = self._id("lamp")
    cmds = [self._cyl(lid, p, h / 2, 0.07, h, METAL, seg=8)]
    if park:
      cmds.append(self._kid(lid, "globe", "sphere", [0, h / 2 + 0.2, 0], TRIM, scale=[0.25, 0.25, 0.25], segments=8, emissive=[0.4, 0.4, 0.35]))
    else:
      cmds.append(self._kid(lid, "head", "cube", [0, h / 2, 0], METAL, rotation=[0, deg(face), 0], scale=[0.12, 0.06, 0.5]))
    return self._add(_rect(p, 0.0, 0.15, 0.15), "prop", cmds, top=h + 0.3)

  def _bench(self, p, psi, avoid=("water", "court", "lot")) -> bool:
    return self._prop_box("bench", p, psi, 0.9, 0.05, 0.3, FENCE_WOOD, y0=0.4, avoid=avoid,
                          kids=[("back", "cube", [0, 0.3, -0.28], FENCE_WOOD, {"scale": [0.9, 0.25, 0.03]})]) is not None

  def _bin(self, p, avoid=("water", "court", "lot")) -> bool:
    return self._add(_rect(p, 0.0, 0.3, 0.3), "prop", [self._cyl(self._id("bin"), p, 0.45, 0.28, 0.9, _m((0.12, 0.25, 0.15), 0.7))],
                     top=0.9, avoid=avoid)

  def _planter(self, p, avoid=("water", "court", "lot")) -> bool:
    return self._prop_box("planter", p, 0.0, 0.8, 0.25, 0.8, STONE, avoid=avoid,
                          kids=[("shrub", "sphere", [0, 0.35, 0], HEDGE, {"scale": [0.7, 0.35, 0.7], "segments": 8})]) is not None

  def _flagpole(self, p, color) -> bool:
    fid = self._id("flag")
    cmds = [self._cyl(fid, p, 5.0, 0.06, 10.0, METAL, seg=8),
            self._kid(fid, "flag", "cube", [0.9, 4.2, 0], _m(color, 0.8), scale=[0.85, 0.5, 0.02])]
    return self._add(_rect(p, 0.0, 0.2, 0.2), "prop", cmds, top=10.0, avoid=("water", "court", "lot", "path"))

  def _fence(self, p0, p1, h, mat, tag="fence", avoid=("path", "lot", "water", "court", "plaza")) -> bool:
    d = np.asarray(p1, dtype=float) - p0
    ln = float(np.linalg.norm(d))
    if ln < 1.0:
      return False
    psi = math.atan2(-d[1], d[0])      # local x along the fence
    c = (np.asarray(p0) + p1) / 2
    return self._prop_box(tag, c, psi, ln / 2, h / 2, 0.04, mat, avoid=avoid) is not None

  def _car(self, p, psi, avoid=("water", "court", "lawn")) -> bool:
    rng = self.r
    kind = "sedan" if rng.random() < 0.7 else "suv"
    col = CARS[int(rng.integers(len(CARS)))]
    cid = self._id("car")
    return self._add(_rect(p, psi, 0.98, 2.42), "prop", car_lite_commands(cid, kind, col, round(float(p[0]), 3), round(float(p[1]), 3), psi),
                     top=1.8, avoid=avoid)

  # ---- 1. land grid and zones ----

  def _grid(self) -> None:
    road_r = self.lane_half + self.footway + 0.5
    self.grid = gr = LandGrid(self.tmap.data["ground"], self.low, self.high, self.jcentres, road_r, self.deck_r)
    js = np.array(self.jcentres)
    self.jmin, self.jmax = js.min(axis=0), js.max(axis=0)
    jc = js.mean(axis=0)
    self.hub = js[int(np.argmin(np.linalg.norm(js - jc, axis=1)))]       # the central junction (J11): skyline peak
    loops = {int(gr.lab_at(st["centre"].mean(axis=0)[None])[0]) for st in self.streets if st["from"] == st["to"]}
    labels = [k for k in range(1, gr.n_labels + 1) if np.any(gr.lab == k)]
    self.zone, self.lrng, self.ldir, self.lcent = {}, {}, {}, {}
    for k in labels:
      c = gr.cells(gr.lab == k).mean(axis=0)
      self.lcent[k] = c
      if k in gr.edge:
        z = "outer"
      elif self.jmin[0] < c[0] < self.jmax[0] and self.jmin[1] < c[1] < self.jmax[1]:
        z = "downtown"
      elif k in loops:
        z = "loop"
      else:
        z = "lobe"
      self.zone[k] = z
      d = c - jc
      self.ldir[k] = ("W" if d[0] > 0 else "E") if abs(d[0]) > abs(d[1]) else ("N" if d[1] > 0 else "S")   # N = +Z, W = +X
      self.lrng[k] = np.random.default_rng(int(self.rng.integers(2 ** 32)))
    self.vrng = np.random.default_rng(int(self.rng.integers(2 ** 32)))
    if self.verbose:
      print("land grid:", {k: (self.zone[k], self.ldir[k], int(np.sum(gr.lab == k)) * 4) for k in labels})

  # ---- 2. street sides ----

  def _sides(self) -> None:
    gr = self.grid
    self.sides: list[Side] = []
    self.bend_sides = []
    blocks = {}
    for k, z in self.zone.items():
      if z == "downtown":
        d = self.lcent[k] - self.hub
        blocks[k] = "cbd" if d[0] * d[1] < 0 else ("park" if d[1] < 0 else "civic")
    self.block_prog = blocks
    for si, st in enumerate(self.streets):
      for sign in (1.0, -1.0):
        sd = Side(st["centre"], st["left"] * sign, self.verge, 2.0 - 0.05, f"{st['from']}-{st['to']}{'+' if sign > 0 else '-'}")
        sd.street = si
        if sd.curved:
          m = sd.k > 1 / 300
          if np.any(m):
            sd.bend_c, sd.bend_n = sd.c[m], sd.n[m]
            sd.blo, sd.bhi = sd.bend_c.min(axis=0), sd.bend_c.max(axis=0)
            self.bend_sides.append(sd)
        p, _ = sd.at(np.linspace(0.1, 0.9, 9) * sd.len, 14.0)
        labs = gr.lab_at(p)
        labs = labs[labs > 0]
        if len(labs) == 0:
          continue
        sd.label = int(np.bincount(labs).argmax())
        sd.zone = self.zone[sd.label]
        if sd.zone == "downtown":
          sd.kind = {"cbd": "tower", "park": "mixed", "civic": "main"}[blocks[sd.label]]
        elif sd.zone == "lobe":
          sd.kind = "house" if sd.curved else "townhouse"
        elif sd.zone == "loop":
          sd.kind = "loop"
        else:
          sd.kind = "house_out" if sd.curved else "commercial"
          if not sd.curved:
            nm = sd.n.mean(axis=0)
            sd.prog = (("retail" if nm[0] > 0 else "apartments") if abs(nm[0]) > abs(nm[1]) else ("highschool" if nm[1] > 0 else "bigbox"))
        self.sides.append(sd)
    prio = {"tower": 0, "mixed": 0, "main": 0, "townhouse": 1, "house": 2, "house_out": 3, "commercial": 3, "loop": 4}
    self.sides.sort(key=lambda sd: (prio[sd.kind], -round(sd.len), sd.name))
    for sd in self.sides:
      rng = self.lrng[sd.label]
      walls = rng.choice(len(SIDING), 2, replace=False)
      sd.theme = {"w": float(rng.uniform(15.0, 19.0) if sd.kind == "house" else rng.uniform(17.0, 21.0)),
                  "setback": 6.0 + float(rng.uniform(-0.3, 0.3)), "walls": [SIDING[int(i)] for i in walls],
                  "roof": ROOFS[int(rng.integers(len(ROOFS)))], "crown": int(rng.integers(2)), "mix": rng.dirichlet((2.0, 2.0, 2.0)),
                  "fence": bool(rng.random() < 0.8), "door": DOORS[int(rng.integers(len(DOORS)))]}

  # ---- 3. access corridors ----

  def _block_frame(self, label: int):
    """downtown block: a frame on its longest side, a in [0, LA] along it, d in [0, LD] across the block (both between
    the frontage lines)"""
    sides = [sd for sd in self.sides if sd.label == label]
    base = max(sides, key=lambda sd: (sd.len, sd.name))
    F = base.frame(base.len / 2)
    pts = np.concatenate([sd.at(sd.s, sd.front)[0] for sd in sides])
    rel = pts - F.o
    a, d = rel @ F.ex, rel @ F.v
    return Frame(F.p(float(a.min()), 0.0), F.v), float(a.max() - a.min()), float(d.max())

  def _corridors(self) -> None:
    self.blocks, self.courts, self.links = {}, [], defaultdict(list)
    gr = self.grid
    for label, prog in sorted(self.block_prog.items()):
      self.r = self.lrng[label]
      F, LA, LD = self._block_frame(label)
      self.blocks[label] = (F, LA, LD, prog)
      self._begin("corridor")
      park = prog == "park"
      mat, layer, cw = (PAVER, L2, 8.0) if park else (ASPHALT, L1, 6.0)
      quads = [(LA / 2 - cw / 2, LA / 2 + cw / 2, -1.95, LD + 1.95), (-1.95, LA + 1.95, LD / 2 - cw / 2, LD / 2 + cw / 2),
               (36.0, LA - 36.0, 36.0, 42.0), (36.0, LA - 36.0, LD - 42.0, LD - 36.0), (36.0, 42.0, 36.0, LD - 36.0),
               (LA - 42.0, LA - 36.0, 36.0, LD - 36.0)]
      for a0, a1, d0, d1 in quads:
        if self._flat(F, a0, a1, d0, d1, layer, mat, "path", tag="alley"):
          gr.reserve(F.rect(a0, a1, d0, d1), 1.0)
    for sd in self.sides:
      if sd.zone != "lobe":
        continue
      self.r = self.lrng[sd.label]
      if sd.kind == "townhouse":
        self._court(sd, 0.28 if self.ldir[sd.label] == "E" else 0.5)
      elif sd.kind == "house":
        for f in (1 / 3, 2 / 3):
          F = sd.frame(sd.len * f)
          self._begin("corridor")
          if self._flat(F, -2.0, 2.0, -sd.gap, 36.0, L2, CONCRETE, "path", tag="link"):
            gr.reserve(F.rect(-2.0, 2.0, -sd.gap, 36.0), 1.0)
            self.links[sd.label].append(F.p(0.0, 36.0))
    self.osites = []
    for label in sorted(self.zone):
      if self.zone[label] == "outer":
        self.r = self.lrng[label]
        self._outer_sites(label)

  def _outer_sites(self, label: int) -> None:
    """B's outer programme behind the frontage band, its drive reserved now so the frontage lots leave the gap:
    warehouses (commercial may stand 20 m from the expressway, homes 30 m)"""
    gr = self.grid
    far = np.zeros_like(gr.road)           # cells within 52 m of a street centre line (the frontage band and its verge)
    c = np.concatenate([st["centre"][::3] for st in self.streets])
    for (i, j), _ in gr._stamp(c, 52.0):
      far[j, i] = True
    near_x = np.zeros_like(gr.road)
    e = self.express_hash.pts
    if len(e):
      for (i, j), _ in gr._stamp(e[::3], 22.0):
        near_x[j, i] = True
    base = gr.free_mask(label, erode=1) & ~far & ~near_x & ~np.isfinite(gr.cap)
    F0 = Frame(self.lcent[label], np.array([0.0, 1.0]))
    # (off house frontages: a drive through a commercial frontage would cut its programme short)
    for prog, target in (("works", (90.0, -90.0)), ("works", (-640.0, 330.0))):
      mask = base & gr.free_mask(label, erode=1)
      for size in ((76.0, 56.0), (56.0, 76.0), (64.0, 48.0), (48.0, 64.0), (56.0, 44.0), (44.0, 56.0)):
        site = gr.fit_rect(mask, F0, [size], np.array(target))
        if site is not None and np.linalg.norm(site[0].o - target) < 90.0:
          break
      else:
        continue
      Fs, w, d = site
      cs_ = self.centre_hash.near(Fs.o, Fs.o, 140.0)
      if len(cs_) == 0:
        self.notes.append(f"{prog} at {np.round(Fs.o)}: no street near")
        continue
      q = cs_[int(np.argmin(np.linalg.norm(cs_ - Fs.o, axis=1)))]
      u = q - Fs.o
      u = np.array([np.sign(u[0]), 0.0]) if abs(u[0]) > abs(u[1]) else np.array([0.0, np.sign(u[1])])
      F = Frame(Fs.o, -u)                     # the site's front (d = -depth/2) faces the street
      if abs(u[0]) > 0.5:
        w, d = d, w
      start = F.p(0.0, -d / 2)
      # the drive: straight on toward the street until the footway
      k = np.arange(0.0, 140.0, 0.5)
      pts = start[None] + np.outer(k, u)
      cd = np.array([np.min(np.linalg.norm(self.centre_hash.near(pt, pt, 8.0) - pt, axis=1)) if len(self.centre_hash.near(pt, pt, 8.0)) else 99.0
                     for pt in pts])
      hit = np.nonzero(cd < self.foot + 0.05)[0]
      if len(hit) == 0:
        self.notes.append(f"{prog} at {np.round(Fs.o)}: no street ahead")
        continue
      end = pts[hit[0]] - u * 0.1
      mid = pts[:hit[0]]
      if np.any(gr.lab_at(mid[:-12]) != label) or np.any(gr.occ_at(mid)):
        self.notes.append(f"{prog} at {np.round(Fs.o)}: drive blocked")
        continue
      self._begin("corridor")
      if not any(self._strip(start, end - u * k, 7.0, L1, ASPHALT, "path", tag="drive") for k in (0.0, 0.5, 1.0)):
        self.notes.append(f"{prog} at {np.round(Fs.o)}: no drive")
        continue
      gr.reserve(_rect((start + end) / 2, math.atan2(u[0], u[1]), 3.5, float(np.linalg.norm(end - start)) / 2), 1.0)
      gr.reserve(F.rect(-w / 2, w / 2, -d / 2, d / 2), 1.0)
      self.osites.append((prog, F, w, d))

  def _court(self, sd: Side, f: float) -> None:
    """a cul-de-sac off the grid street into the lobe: 6 m lane, a bulb, house lots both sides"""
    gr = self.grid
    F = sd.frame(sd.len * f)
    dd = np.arange(0.5, 260.0, 1.0)
    pts = np.array([F.p(0.0, d) for d in dd])
    in_lobe = gr.lab_at(pts) == sd.label
    homes = self._express_d(pts) > EXPRESS_HOMES
    ok = in_lobe & homes
    free = float(dd[np.argmin(ok)]) if not ok.all() else 260.0
    stop = min(int(free), len(dd) - 1)
    far = 34.0 if homes[stop] else 0.0         # stopped by the lobe road (its lots to come), or by the expressway buffer
    r = 11.0
    lc = min(free - far - 30.0 - r - 2.0, 100.0)
    if lc < 75.0:
      self.notes.append(f"no court on {sd.name}: {free:.0f} m deep")
      return
    self._begin("corridor")
    if not self._flat(F, -3.0, 3.0, -sd.gap, lc, L1, ASPHALT, "path", tag="court"):
      return
    bulb = _octagon(F.p(0.0, lc), r)
    bid = self._id("bulb")
    cmd = {"action": "create", "type": "cylinder", "id": bid, "position": [round(float(bulb.mean(axis=0)[0]), 3), L1 - 0.01,
           round(float(bulb.mean(axis=0)[1]), 3)], "radius": 1.0, "height": 0.02, "segments": 24, "scale": [r, 1.0, r], **ASPHALT}
    self._add(bulb, "flat", [cmd], purpose="path", y=L1, key=tuple(ASPHALT["color"]))
    gr.reserve(F.rect(-3.0, 3.0, -sd.gap, lc), 1.0)
    gr.reserve(bulb, 1.0)
    self.courts.append((sd, F, lc, r))

  def _express_d(self, pts: np.ndarray) -> np.ndarray:
    e = self.express_hash.pts
    if len(e) == 0:
      return np.full(len(pts), np.inf)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    near = self.express_hash.near(lo, hi, EXPRESS_HOMES + 5.0)
    if len(near) == 0:
      return np.full(len(pts), np.inf)
    return np.min(np.linalg.norm(pts[:, None] - near[None], axis=2), axis=1)

  # ---- 4. frontage parcels ----

  def _runs(self, sd: Side, dp: float, dp_min: float, s_lo: float = CORNER, s_hi: float | None = None):
    """free stretches along a side: [(s0, s1, depth per 1 m sample, homes allowed)]"""
    gr = self.grid
    s_hi = sd.len - CORNER if s_hi is None else s_hi
    if s_hi - s_lo < 4.0:
      return []
    ss = np.arange(s_lo, s_hi, 1.0)
    ds = np.arange(0.5, dp + 0.01, 2.0)
    p, _ = sd.at(np.broadcast_to(ss[:, None], (len(ss), len(ds))), sd.front + ds[None])
    ok = (gr.lab_at(p) == sd.label) & ~gr.occ_at(p)
    depth = np.where(ok.all(axis=1), dp, np.argmin(ok, axis=1) * 2.0)
    depth = np.minimum(depth, dp)
    homes = (self._express_d(p[:, 0]) > EXPRESS_HOMES) & (self._express_d(p[:, -1]) > EXPRESS_HOMES)
    good = depth >= dp_min
    out, i = [], 0
    while i < len(ss):
      if not good[i]:
        i += 1
        continue
      j = i
      while j + 1 < len(ss) and good[j + 1] and homes[j + 1] == homes[i]:
        j += 1
      if j > i:
        out.append((float(ss[i]), float(ss[j]) + 1.0, depth[i:j + 1], bool(homes[i])))
      i = j + 1
    return out

  def _parcel(self, sd: Side, s0: float, s1: float, depth: float, kind: str, index: int = 0) -> Parcel:
    p0, n0 = sd.at(s0, sd.front)
    p1, n1 = sd.at(s1, sd.front)
    quad = _ccw(np.array([p0, p1, p1 + n1 * depth, p0 + n0 * depth]))
    F = sd.frame((s0 + s1) / 2)
    return Parcel(sd, s0, s1, depth, quad, F, float(np.linalg.norm(p1 - p0)), kind, self.r, sd.theme, sd.gap, index)

  def _split(self, run, w: float, w_min: float, w_max: float, sd: Side | None = None):
    """equal lots along a run; on the inside of a bend (sd given) the lots start wider so they keep their width deep in"""
    s0, s1, depth, _ = run
    ln = s1 - s0
    if ln < w_min:
      return []

    def width(s):
      if sd is None:
        return w
      k = float(np.interp(s, sd.s, sd.k))
      if k < 1 / 300:
        return w
      rf = 1.0 / k - sd.front
      return min(w * rf / max(rf - 16.0, 1.0), 25.0)

    cuts, s = [s0], s0
    while s < s1:
      s += width(s + w / 2)
      cuts.append(s)
    n = len(cuts) - 1
    if n > 1 and cuts[-1] - s1 > (cuts[-1] - cuts[-2]) / 2:
      cuts.pop()
      n -= 1
    cuts = s0 + (np.asarray(cuts) - s0) * ln / (cuts[-1] - s0)      # stretch to fill the run exactly
    out = []
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
      if b - a < w_min * 0.9 or b - a > w_max * 1.3:
        continue
      dd = depth[int(a - s0):max(int(math.ceil(b - s0)), int(a - s0) + 1)]
      out.append((float(a), float(b), float(np.floor(dd.min() / 2.0) * 2.0)))
    return out

  def _frontages(self) -> None:
    for sd in self.sides:
      self.r = self.lrng[sd.label]
      if sd.kind in ("tower", "mixed", "main"):
        self._downtown_side(sd)
      elif sd.kind == "townhouse":
        self._townhouse_side(sd)
      elif sd.kind in ("house", "house_out"):
        dp = 34.0 if sd.kind == "house" else 40.0
        self._house_side(sd, dp)
      elif sd.kind == "commercial":
        self._commercial_side(sd)
    for sd, F, lc, r in self.courts:
      self.r = self.lrng[sd.label]
      self._court_lots(sd, F, lc, r)

  def _reserve_lots(self, lots) -> None:
    for P in lots:
      self.grid.reserve(P.quad, 0.0)

  def _pocket(self, sd: Side, s0: float, s1: float, depth: float) -> None:
    if s1 - s0 < 8.0:
      return
    P = self._parcel(sd, s0, s1, min(depth, 30.0), "pocket")
    self.grid.reserve(P.quad, 0.0)
    self._pocket_park(P)

  # downtown

  def _downtown_side(self, sd: Side) -> None:
    gr = self.grid
    F0, LA, LD, prog = self.blocks[sd.label]
    rng = self.r
    # corner squares: paved to the frontage depth, nothing taller than 1 m (sight lines)
    for s0, s1 in ((0.0, CORNER), (sd.len - CORNER, sd.len)):
      for trim in (0.0, 3.0, 6.0):
        a, b = (s0 + trim, s1) if s0 == 0.0 else (s0, s1 - trim)
        P = self._parcel(sd, a, b, 35.0, "corner")
        self._begin("corner", P)
        if self._flat(P.frame, -P.width / 2, P.width / 2, -sd.gap, 35.0, L2, PAVER, "plaza", tag="corner"):
          gr.reserve(P.quad, 0.0)
          break
    w = {"tower": 36.0, "mixed": 38.0, "main": 12.0}[sd.kind]
    if sd.label not in self.block_plazas:      # two of the block's four corners get a corner plaza
      self.block_plazas[sd.label] = {int(k) for k in rng.choice(4, 2, replace=False)}
    plaza_j = self.block_plazas[sd.label]
    for run in self._runs(sd, 35.0, 30.0):
      lots = [self._parcel(sd, a, b, d, sd.kind, i) for i, (a, b, d) in enumerate(self._split(run, w, w * 0.7, w * 1.4))]
      self._reserve_lots(lots)
      for i, P in enumerate(lots):
        corner_j = self._block_corner(sd.label, P)
        P.corner = corner_j is not None and i in (0, len(lots) - 1)
        if sd.kind == "tower" and P.corner and corner_j in plaza_j:
          self._corner_plaza(P)
        elif sd.kind == "tower":
          self._tower(P)
        elif sd.kind == "mixed":
          self._midrise_front(P)
        else:
          self._shop(P)

  def _block_corner(self, label: int, P: Parcel):
    """which of the block's corners (0..3, by its frame) a lot touches, if any"""
    F, LA, LD, _ = self.blocks[label]
    best = None
    for k, (a, d) in enumerate(((0, 0), (LA, 0), (LA, LD), (0, LD))):
      if np.min(np.linalg.norm(P.quad - F.p(a, d), axis=1)) < CORNER + 12.0:
        best = k
    return best

  # lobes

  def _townhouse_side(self, sd: Side) -> None:
    """rows of up to 8 units (each with its rear lane), pocket parks between and at the ends"""
    for run in self._runs(sd, 33.0, 33.0):
      s0, s1, depth, homes = run
      ln = s1 - s0
      rows = max(1, int((ln + 4.0) // (8 * 6.5 + 10.0 + 4.0)))
      n = min(8, int((ln / rows - 10.0 - 4.0) // 6.5))
      if not homes or n < 4:
        self._pocket(sd, s0, s1, float(depth.min()))
        continue
      wrow = n * 6.5 + 10.0
      gap = (ln - rows * wrow) / (rows + 1)
      prev = s0
      for k in range(rows):
        a = s0 + gap + k * (wrow + gap)
        P = self._parcel(sd, a, a + wrow, 33.0, "townhouse")
        self.grid.reserve(P.quad, 0.0)
        self._townhouse_row(P, n)
        self._pocket(sd, prev, a - 1.0, 30.0)
        prev = a + wrow + 1.0
      self._pocket(sd, prev, s1, 30.0)

  def _house_side(self, sd: Side, dp: float) -> None:
    for run in self._runs(sd, dp, DP_MIN):
      s0, s1, depth, homes = run
      if not homes:
        n = max(1, int(math.ceil((s1 - s0) / 40.0)))
        for k in range(n):
          a, b = s0 + (s1 - s0) * k / n, s0 + (s1 - s0) * (k + 1) / n
          self._under_or_belt(sd, (a, b, depth[int(a - s0):max(int(b - s0), int(a - s0) + 1)], False))
        continue
      lots = [self._parcel(sd, a, b, d, "house", i) for i, (a, b, d) in enumerate(self._split(run, sd.theme["w"], 14.0, 25.0, sd))]
      if not lots:
        self._pocket(sd, s0, s1, float(depth.min()))
      self._reserve_lots(lots)
      for P in lots:
        if not self._house(P):
          self._pocket_park(P)
      sd.lots += lots

  def _court_lots(self, sd: Side, F: Frame, lc: float, r: float) -> None:
    gr = self.grid
    for sign in (1.0, -1.0):
      c = np.array([F.p(0.0, d) for d in np.arange(34.0, lc - r - 1.0, 1.0)])
      if len(c) < 20:
        continue
      n = np.tile(F.ex * sign, (len(c), 1))
      cs = Side(c, n, 4.5, 1.6, sd.name + ("court+" if sign > 0 else "court-"))
      cs.label, cs.zone, cs.kind, cs.theme = sd.label, "lobe", "court", sd.theme
      for run in self._runs(cs, 30.0, DP_MIN, 0.0, cs.len):
        if not run[3]:
          continue
        lots = [self._parcel(cs, a, b, d, "house", i) for i, (a, b, d) in enumerate(self._split(run, 17.0, 14.0, 22.0))]
        self._reserve_lots(lots)
        for P in lots:
          if not self._house(P):
            self._pocket_park(P)
    # the lot at the end of the bulb, facing back down the court
    o = F.p(0.0, lc + r + 1.5)
    Fe = Frame(o, F.v)
    quad = Fe.rect(-10.0, 10.0, 0.0, 30.0)
    probe = np.array([Fe.p(a, d) for a in (-9.5, 0.0, 9.5) for d in (0.5, 10.0, 20.0, 29.5)])
    if np.all(gr.lab_at(probe) == sd.label) and not np.any(gr.occ_at(probe)) and np.all(self._express_d(probe) > EXPRESS_HOMES):
      P = Parcel(sd, 0.0, 20.0, 30.0, quad, Fe, 20.0, "house", self.r, sd.theme, 2.0, 0)
      gr.reserve(quad, 0.0)
      if not self._house(P):
        self._pocket_park(P)

  def _under_or_belt(self, sd: Side, run) -> None:
    """a stretch of frontage too close to the expressway for homes: parking under the deck, else a strip of trees"""
    s0, s1, depth, _ = run
    d = float(min(depth.min(), 40.0))
    P = self._parcel(sd, s0, s1, d, "under")
    under = np.mean(np.isfinite(self.grid.cap_at(np.array([P.frame.p(a, dd) for a in np.linspace(-P.width / 2, P.width / 2, 7)
                                                              for dd in np.linspace(1.0, d - 1.0, 5)])))) > 0.2
    self.grid.reserve(P.quad, 0.0)
    if under and P.width >= 30.0 and d >= 24.0:
      self._begin("lot", P)
      self._surface_lot(P.frame, -P.width / 2 + 0.5, P.width / 2 - 0.5, 1.0, d - 0.5, 0.5, gap=sd.gap + 1.0, within=P.quad)
    elif under and P.width >= 18.0:
      self._community_garden(P)
    elif not under and P.width >= 26.0 and d >= 26.0 and self._express_d(P.quad).min() > 20.0 and self._works(P):
      pass
    else:
      self._begin("belt", P)
      for a in np.arange(-P.width / 2 + 4.0, P.width / 2 - 3.0, 9.0):
        for dd in (6.0, 15.0, 24.0):
          if dd < d - 3.0:
            self._tree(P.frame.p(a + float(self.r.uniform(-1.5, 1.5)), dd), conifer=self.r.random() < 0.4, within=P.quad)

  def _works(self, P: Parcel) -> bool:
    """a small workshop / self-storage block (commercial may stand 20 m from the expressway): roller doors on a concrete
    apron off the street"""
    F, rng = P.frame, P.rng
    self.r = rng
    g = self._begin("works", P)
    bw = min(P.width - 6.0, 34.0)
    if not self._flat(F, -bw / 2, bw / 2, -P.gap, 9.98, L2, CONCRETE, "path", tag="apron"):
      self._abort(g)
      return False
    wid = self._id("works")
    h = 5.0
    cmds = [self._box(wid, F.p(0.0, 17.0), h / 2, bw / 2, h / 2, 7.0, F.face, _m((0.66, 0.66, 0.68), 0.7, 0.2)),
            self._kid(wid, "band", "cube", [0, h / 2 - 0.4, 7.03], SHOPS[int(rng.integers(len(SHOPS)))], scale=[bw / 2, 0.35, 0.03])]
    for k in range(int(bw // 6.0)):
      cmds.append(self._kid(wid, f"door{k}", "cube", [-bw / 2 + 3.0 + 6.0 * k, -h / 2 + 1.6, 7.03], TRIM, scale=[1.6, 1.6, 0.03]))
    if not self._add(F.rect(-bw / 2, bw / 2, 10.0, 24.0), "bldg", cmds, top=h, within=P.quad, door=F.p(0.0, 9.7)):
      self._abort(g)
      return False
    if rng.random() < 0.6:
      self._car(F.p(-bw / 2 + 4.0, 4.0), F.face + math.pi / 2, avoid=("water", "court"))
    return True

  # ---- recipes: residential ----

  def _house(self, P: Parcel) -> bool:
    """2.1: a detached house with an attached garage, driveway, front path, back garden"""
    rng, t, F = P.rng, P.theme, P.frame
    self.r = rng
    g = self._begin("house", P)
    side = (-1.0, -1.0, 1.0, 1.0)[P.index % 4]       # driveways alternate sides in pairs
    first = int(rng.choice(3, p=t["mix"]))
    sb = t["setback"]
    gw, gd, gh = 3.8, 7.0, 3.0
    fit = None
    for typ in [("ranch", "colonial", "bungalow")[(first + k) % 3] for k in range(3)]:
      hw, hd, h = {"ranch": (14.0, 9.0, 3.2), "colonial": (10.0, 9.0, 6.0), "bungalow": (9.0, 12.0, 4.0)}[typ]
      for _ in range(3):        # shrink the width up to 3 x 10 %
        W = hw + gw + 0.05
        ah, ag = -side * (W / 2 - hw / 2), side * (W / 2 - gw / 2)
        body = F.rect(ah - hw / 2, ah + hw / 2, sb, sb + hd)
        gar = F.rect(ag - gw / 2, ag + gw / 2, sb + 1.5, sb + 1.5 + gd)
        if np.all(_inside(np.concatenate([body, gar]), P.quad, 1.0)):
          fit = (ah, ag)
          break
        hw *= 0.9
      if fit:
        break
    if fit is None:
      self._abort(g)
      return False
    ah, ag = fit
    wall = t["walls"][int(rng.integers(2))]
    bid = self._id("house")
    hc = F.p(ah, sb + hd / 2)
    kids = []
    front = hd / 2 + 0.02
    a_door = side * (hw / 2 - 1.6)                      # body-local x of the door: on the garage side
    kids.append(("door", "cube", [a_door, -h / 2 + 1.05, front], t["door"], {"scale": [0.5, 1.05, 0.03]}))
    nwin = 2 if typ == "bungalow" else 3
    for k in range(nwin):
      x = -side * (hw / 2 - 1.8 - k * (hw - 4.6) / max(nwin - 1, 1)) if nwin > 1 else 0.0
      if abs(x - a_door) < 1.4:
        continue
      kids.append((f"w{k}", "cube", [x, -h / 2 + 1.6, front], GLASSES[0], {"scale": [0.6, 0.6, 0.03]}))
    if typ == "colonial":
      for k, x in enumerate((-hw * 0.28, hw * 0.28)):
        kids.append((f"u{k}", "cube", [x, h / 2 - 1.3, front], GLASSES[0], {"scale": [0.55, 0.6, 0.03]}))
      r = (hd / 2 + 0.35) / math.sqrt(2.0)
      kids.append(("roof", "cube", [0, h / 2, 0], t["roof"], {"rotation": [45, 0, 0], "scale": [hw / 2 + 0.2, r, r]}))
      roof_top = h + hd / 2 + 0.35
      if rng.random() < 0.5:
        kids.append(("chimney", "cube", [hw * 0.3, h / 2 + hd * 0.3, -hd * 0.15], BRICKS[0], {"scale": [0.35, 0.9, 0.35]}))
    else:
      pitch = math.radians(22.0)
      half = hd / 2 + 0.3
      for k, sgn in enumerate((1.0, -1.0)):
        kids.append((f"roof{k}", "cube", [0, h / 2 + half / 2 * math.tan(pitch) + 0.05, sgn * half / 2], t["roof"],
                     {"rotation": [sgn * 22.0, 0, 0], "scale": [hw / 2 + 0.3, 0.08, half / 2 / math.cos(pitch)]}))
      roof_top = h + half * math.tan(pitch) + 0.2
    porch = typ == "bungalow" or rng.random() < 0.4
    if porch:
      for k, x in enumerate((a_door - 1.4, a_door + 1.4)):
        kids.append((f"post{k}", "cube", [x, -h / 2 + 1.25, front + 1.8], TRIM, {"scale": [0.08, 1.25, 0.08]}))
      kids.append(("porch", "cube", [a_door, -h / 2 + 2.55, front + 1.0], t["roof"], {"scale": [1.7, 0.06, 1.1]}))
    cmds = [self._box(bid, hc, h / 2, hw / 2, h / 2, hd / 2, F.face, wall)] + [self._kid(bid, *k[:4], **k[4]) for k in kids]
    door_p = F.p(ah + a_door, sb - 0.3)
    if not self._add(body, "bldg", cmds, top=roof_top + 1.0, within=P.quad, door=door_p):
      self._abort(g)
      return False
    gid = self._id("garage")
    gcmds = [self._box(gid, F.p(ag, sb + 1.5 + gd / 2), gh / 2, gw / 2, gh / 2, gd / 2, F.face, wall),
             self._kid(gid, "door", "cube", [0, -gh / 2 + 1.1, gd / 2 + 0.02], TRIM, scale=[1.5, 1.1, 0.03]),
             self._kid(gid, "roof", "cube", [0, gh / 2 + 0.06, 0], t["roof"], scale=[gw / 2 + 0.1, 0.06, gd / 2 + 0.1])]
    if not self._add(gar, "bldg", gcmds, top=gh + 0.15, within=P.quad):
      self._abort(g)
      return False
    # driveway from the footway (court) edge to the garage, the front path from the door onto it
    # (on the inside of a bend the frontage curves back toward the street at the lot's sides: start the drive a bit in)
    if not any(self._flat(F, ag - 1.6, ag + 1.6, -P.gap + k, sb + 1.5 - 0.02, L2, CONCRETE, "path", tag="drive") for k in (0.0, 0.4, 0.8)):
      self._abort(g)
      return False
    P.drive_a = ag
    a_d = ah + a_door
    edge = ag - side * 1.6
    if porch:
      self._flat(F, a_d - 1.6, a_d + 1.6, sb - 2.0, sb - 0.02, L2, CONCRETE, "path", tag="porch")
      d0, d1 = sb - 3.2, sb - 2.0
    else:
      d0, d1 = sb - 1.3, sb - 0.05
    lo, hi = sorted((a_d - side * 0.6, edge + side * 0.1))
    self._flat(F, lo, hi, d0, d1, L2, CONCRETE, "path", tag="walk")
    if rng.random() < 0.4:
      self._car(F.p(ag, sb + 1.5 - 2.7), F.face + math.pi)
    # back garden
    back = sb + hd
    self._flat(F, ah - 2.0, ah + 2.0, back + 0.05, back + 3.05, L2, PAVER, "yard", tag="patio")
    room = P.depth - back - 1.0
    k = rng.random()
    corner_a = -side * (P.width / 2 - 3.0)
    if room > 9.0 and k < 0.3:
      sid = self._id("shed")
      self._add(F.rect(corner_a - 1.25, corner_a + 1.25, P.depth - 3.2, P.depth - 1.2), "bldg",
                [self._box(sid, F.p(corner_a, P.depth - 2.2), 1.1, 1.25, 1.1, 1.0, F.face, FENCE_WOOD),
                 self._kid(sid, "roof", "cube", [0, 1.15, 0], t["roof"], scale=[1.4, 0.06, 1.15])], top=2.4, within=P.quad)
    elif room > 10.0 and k < 0.5:
      pc = F.p(ah, back + 7.0)
      pid = self._id("pool")
      self._add(_rect(pc, 0.0, 2.3, 2.3), "prop", [self._cyl(pid, pc, 0.6, 2.2, 1.2, TRIM, seg=20),
                                                  self._kid(pid, "water", "cylinder", [0, 0.55, 0], POOL, radius=2.05, height=0.12, segments=20)],
                top=1.2, within=P.quad)
    elif room > 9.0 and k < 0.65:
      sc = F.p(corner_a, P.depth - 4.0)
      self._prop_box("swing", sc, F.face, 1.6, 0.05, 0.05, PLAY[int(rng.integers(3))], y0=2.2, within=P.quad,
                     kids=[(f"leg{i}", "cube", [x, -1.1, 0], METAL, {"scale": [0.05, 1.1, 0.6]}) for i, x in enumerate((-1.55, 1.55))])
    tree_d = back + 4.0 + 2.4
    if tree_d < P.depth - 2.0:
      self._tree(F.p(side * (P.width / 2 - 4.0), tree_d + float(rng.uniform(0.0, max(0.0, P.depth - 2.0 - tree_d)))), within=P.quad)
    # fences: the back line and one shared side line, from the back of the house
    q = P.quad
    if t["fence"] and P.side.kind != "court":
      bl, br = self._edge_point(P, -1.0, P.depth - 0.1), self._edge_point(P, 1.0, P.depth - 0.1)
      u = (br - bl) / max(np.linalg.norm(br - bl), 1e-6)
      self._fence(bl + u * 0.15, br - u * 0.15, 1.8, FENCE_WOOD)
      s0, s1 = self._edge_point(P, 1.0, back + 0.5), self._edge_point(P, 1.0, P.depth - 0.3)
      inward = -P.frame.ex * 0.06
      self._fence(s0 + inward, s1 + inward, 1.8, FENCE_WOOD)
    for x in (-hw * 0.3, hw * 0.3):      # shrubs on the front lawn
      if rng.random() < 0.7:
        sp = F.p(ah + x, sb - 1.6 - float(rng.uniform(0.0, 1.5)))
        self._add(_rect(sp, 0.0, 0.7, 0.7), "prop", [self._box(self._id("shrub"), sp, 0.4, 0.7, 0.4, 0.7, 0.0, HEDGE) | {"type": "sphere", "segments": 8}],
                  top=0.8, within=q)
    if rng.random() < 0.35:
      lo_a = -side * (P.width / 2 - 0.8)
      hi_a = ag - side * 2.2
      lo_h, hi_h = sorted((lo_a, hi_a))
      if hi_h - lo_h > 3.0:
        self._prop_box("hedge", F.p((lo_h + hi_h) / 2, 0.8), F.face, (hi_h - lo_h) / 2, 0.3, 0.35, HEDGE, within=q)
    return True

  @staticmethod
  def _edge_point(P: Parcel, sgn: float, d: float) -> np.ndarray:
    """a point on the parcel's left (-1) or right (+1) side line, d deep (lot sides follow the street normals)"""
    q = P.quad
    fr = np.array([P.frame.p(sgn * P.width / 2, 0.0)])
    i = int(np.argmin(np.linalg.norm(q - fr, axis=1)))
    # the front corner on that side and the back corner after it
    cands = [q[(i + 1) % 4], q[(i - 1) % 4]]
    j = max(cands, key=lambda c: float((c - q[i]) @ P.frame.v))
    t = d / max(float((j - q[i]) @ P.frame.v), 1e-6)
    return q[i] + (j - q[i]) * t

  def _townhouse_row(self, P: Parcel, n: int) -> None:
    """2.2: n units on a 6.5 m pitch, garages on a rear lane joining the street at both row ends"""
    F, rng = P.frame, P.rng
    self.r = rng
    g = self._begin("townhouse", P)      # one group: the row stands or falls with its lane
    W = n * 6.5 + 10.0
    a0 = -W / 2
    ok = all([self._flat(F, a0, a0 + 5.0, -P.gap, 33.0, L1, ASPHALT, "path", tag="lane"),
              self._flat(F, a0 + W - 5.0, a0 + W, -P.gap, 33.0, L1, ASPHALT, "path", tag="lane"),
              self._flat(F, a0, a0 + W, 28.0, 33.0, L1, ASPHALT, "path", tag="lane")])
    if not ok:
      self._abort(g)
      return
    gid = self._id("garages")
    gl = n * 6.5 - 0.1
    gcmds = [self._box(gid, F.p(0.0, 25.0), 1.5, gl / 2, 1.5, 2.95, F.face, BRICKS[1]),
             self._kid(gid, "roof", "cube", [0, 1.56, 0], ROOFTOP, scale=[gl / 2, 0.06, 2.95])]
    for k in range(n):
      gcmds.append(self._kid(gid, f"d{k}", "cube", [a0 + 5.0 + 6.5 * k + 3.25, -0.4, -2.97], TRIM, scale=[1.3, 1.05, 0.03]))
    self._add(F.rect(-gl / 2, gl / 2, 22.05, 27.95), "bldg", gcmds, top=3.2)
    for k in range(n):
      ua = a0 + 5.0 + 6.5 * k + 3.25
      uid = self._id("terrace")
      pa = ua + (2.0 if k % 2 else -2.0)              # paths at the outer sides of each pair
      cmds = [self._box(uid, F.p(ua, 8.5), 4.5, 3.2, 4.5, 5.5, F.face, BRICKS[k // 2 % 2])]
      x = pa - ua
      cmds.append(self._kid(uid, "door", "cube", [x, -3.45, 5.52], P.theme["door"], scale=[0.5, 1.05, 0.03]))
      cmds.append(self._kid(uid, "stoop", "cube", [x, -4.35, 6.0], CONCRETE, scale=[0.8, 0.15, 0.5]))
      for fl in range(3):        # two windows a floor on the front, the ground floor's beside the door
        for j, wx in enumerate((-x,) if fl == 0 else (-x, x)):
          cmds.append(self._kid(uid, f"w{fl}{j}", "cube", [wx, -4.5 + 1.6 + fl * 3.0, 5.52], GLASSES[0], scale=[0.55, 0.65, 0.03]))
      cmds.append(self._kid(uid, "parapet", "cube", [0, 4.65, 0], ROOFTOP, scale=[3.2, 0.15, 5.5]))
      self._add(F.rect(ua - 3.2, ua + 3.2, 3.0, 14.0), "bldg", cmds, top=9.3, door=F.p(pa, 2.8))
      self._flat(F, pa - 0.6, pa + 0.6, -P.gap, 2.98, L2, CONCRETE, "path", tag="walk")
      if k % 2 == 1:
        left = ua - 6.5 + 2.0 + 0.7
        self._fence(F.p(left, 1.0), F.p(pa - 0.7, 1.0), 1.0, FENCE_IRON)
      if rng.random() < 0.5:
        self._tree(F.p(ua + float(rng.uniform(-1.0, 1.0)), 19.0), size=0)

  def _pocket_park(self, P: Parcel) -> None:
    """2.8: lawn, an L path to the verge, 3-5 trees, 2 benches, a bin, sometimes a play set"""
    F, rng = P.frame, P.rng
    self.r = rng
    self._begin("pocket", P)
    w, d = P.width, min(P.depth, 30.0)
    inset = 1.0 if w > 12 else 0.5
    if not self._flat(F, -w / 2 + inset, w / 2 - inset, 0.5, d - 0.5, L0, LAWN, "lawn", tag="lawn", within=P.quad):
      return
    self._flat(F, -0.75, 0.75, -P.gap, d * 0.5, L2, GRAVEL, "path", tag="path")
    self._flat(F, -0.75, min(w / 2 - 2.0, 8.0), d * 0.5 - 1.5, d * 0.5, L2, GRAVEL, "path", tag="path")
    for _ in range(int(rng.integers(3, 6))):
      self._tree(F.p(float(rng.uniform(-w / 2 + 2.5, w / 2 - 2.5)), float(rng.uniform(4.0, d - 3.0))), within=P.quad)
    self._bench(F.p(-2.0, d * 0.5 + 1.2), F.face)
    self._bench(F.p(3.0, d * 0.5 - 2.6), F.face + math.pi)
    self._bin(F.p(1.6, d * 0.5 + 1.0))
    if rng.random() < 0.5 and w > 14:
      self._play(F.p(-w / 4, d * 0.75), F.face)

  def _community_garden(self, P: Parcel) -> None:
    """under the deck: allotments, raised beds on gravel"""
    F = P.frame
    self._begin("garden", P)
    w, d = P.width, min(P.depth, 30.0)
    self._flat(F, -w / 2 + 1.0, w / 2 - 1.0, 0.5, d - 0.5, L2, GRAVEL, "path", tag="garden", within=P.quad)
    self._flat(F, -1.0, 1.0, -P.gap, 0.6, L2, GRAVEL, "path", tag="path")
    for a in np.arange(-w / 2 + 3.0, w / 2 - 2.5, 3.5):
      for dd in np.arange(3.0, d - 2.0, 4.0):
        if abs(a) < 1.6:
          continue
        self._prop_box("bed", F.p(a, dd), F.face, 1.0, 0.25, 1.5, FENCE_WOOD, avoid=("water", "court", "lot"),
                       kids=[("soil", "cube", [0, 0.26, 0], _m((0.25, 0.18, 0.12), 0.95), {"scale": [0.9, 0.02, 1.4]})])

  # ---- recipes: downtown ----

  def _tower(self, P: Parcel) -> None:
    """2.4: podium on a paved forecourt, a shaft on half the mid-block lots and every corner, skyline peaking at the hub"""
    F, rng, W = P.frame, P.rng, P.width
    self.r = rng
    self._begin("tower", P)
    plaza = (not P.corner) and rng.random() < 0.25
    sb = 8.0 if plaza else 2.0
    self._flat(F, -W / 2, W / 2, -P.gap, sb, L2, PAVER, "plaza", tag="forecourt", check=True)
    wd = W - 0.2
    hp = float(rng.choice((8.0, 11.0, 14.0)))
    col = TOWERS[int(rng.integers(len(TOWERS)))]
    glass = GLASSES[int(rng.integers(2))]
    shop = SHOPS[int(rng.integers(len(SHOPS)))]
    pid = self._id("tower")
    dep = 26.0
    cmds = [self._box(pid, F.p(0.0, sb + dep / 2), hp / 2, wd / 2, hp / 2, dep / 2, F.face, col)]

    def kid(name, pos, scale, mat, **kw):
      cmds.append(self._kid(pid, name, "cube", pos, mat, scale=scale, **kw))

    fz = dep / 2 + 0.03
    kid("gf", [0, -hp / 2 + 1.6, fz], [wd / 2 - 0.6, 1.2, 0.03], glass)
    kid("fascia", [0, -hp / 2 + 3.2, fz + 0.02], [wd / 2 - 0.3, 0.3, 0.04], shop)
    kid("awning", [0, -hp / 2 + 2.9, fz + 0.9], [wd / 2 - 0.8, 0.05, 0.9], shop)
    for k in range(min(3, int((hp - 4.0) // 3.2))):
      kid(f"pb{k}", [0, -hp / 2 + 5.3 + k * 3.2, fz], [wd / 2 - 0.5, 0.8, 0.03], glass)
    top = hp + 0.4
    kid("parapet", [0, hp / 2 + 0.2, 0], [wd / 2, 0.2, dep / 2], ROOFTOP)
    if P.corner or rng.random() < 0.5:
      hub_d = float(np.linalg.norm(F.o - self.hub))
      H = 24.0 + 46.0 * min(max(1.0 - hub_d / 350.0, 0.0), 1.0) * float(rng.uniform(0.6, 1.0))
      sw, sdp = W - 6.0, 20.0
      sy = hp / 2 + H / 2
      kid("shaft", [0, sy, 0], [sw / 2, H / 2, sdp / 2], col)
      floors = int(H // 3.5)
      levels = [f for f in range(floors) if f < 10 or f % 2 == 0][:12]
      faces = [(0.0, sdp / 2 + 0.03, sw / 2 - 0.6, 0)]
      if P.corner:
        end = self._corner_side(P)
        faces.append((end * (sw / 2 + 0.03), 0.0, sdp / 2 - 0.6, 90))
      for fi, (fx, fz2, half, rot) in enumerate(faces):
        for f in levels:
          y = hp / 2 + 1.8 + f * 3.5
          if y > hp / 2 + H - 1.0:
            break
          kid(f"s{fi}{f}", [fx, y, fz2], [half, 0.9, 0.03], glass, rotation=[0, rot, 0])
      kid("cap", [0, hp / 2 + H + 0.25, 0], [sw / 2, 0.25, sdp / 2], ROOFTOP)
      kid("plant", [sw * 0.15, hp / 2 + H + 1.6, -2.0], [min(4.0, sw / 4), 1.1, 3.0], METAL)
      top = hp + H + 2.7
    else:
      kid("plant", [wd * 0.2, hp / 2 + 1.4, -3.0], [3.0, 1.0, 2.5], METAL)
      top = hp + 2.4
    kid("loading", [wd * 0.3, -hp / 2 + 1.6, -dep / 2 - 0.03], [1.8, 1.6, 0.03], METAL)
    if not self._add(F.rect(-wd / 2, wd / 2, sb, sb + dep), "bldg", cmds, top=top, within=P.quad, door=F.p(0.0, sb - 0.3)):
      self._abort(self.g)
      return
    self._prop_box("dumpster", F.p(wd * 0.3 - 3.5, sb + dep + 2.0), F.face, 1.0, 0.7, 0.8, _m((0.15, 0.3, 0.2), 0.7), within=P.quad)
    if plaza:
      for k, a in enumerate((-W / 2 + 4.0, W / 2 - 4.0)):
        self._planter(F.p(a, 3.0 + k * 2.0))
        self._tree(F.p(a * 0.6, 5.0))
      self._bench(F.p(0.0, 6.5), F.face + math.pi)
      sid = self._id("sculpture")
      self._add(_rect(F.p(W / 4, 5.0), 0.0, 0.6, 0.6), "prop",
                [self._box(sid, F.p(W / 4, 5.0), 0.5, 0.6, 0.5, 0.6, F.face, STONE),
                 self._kid(sid, "orb", "sphere", [0, 1.4, 0], METAL, scale=[0.9, 0.9, 0.9], segments=12)], top=2.3)

  def _corner_side(self, P: Parcel) -> float:
    """which way (+1 / -1 along the frame's a) the lot's corner (the nearer junction) lies"""
    j = min(self.jcentres, key=lambda c: float(np.linalg.norm(np.asarray(c) - P.frame.o)))
    return 1.0 if float((np.asarray(j) - P.frame.o) @ P.frame.ex) > 0 else -1.0

  def _corner_plaza(self, P: Parcel) -> None:
    F, rng, W = P.frame, P.rng, P.width
    self.r = rng
    self._begin("plaza", P)
    self._flat(F, -W / 2, W / 2, -P.gap, P.depth, L2, PAVER, "plaza", tag="plaza")
    for a in np.linspace(-W / 2 + 4.0, W / 2 - 4.0, 4):
      self._planter(F.p(float(a), 4.0))
    for a in (-W / 4, W / 4):
      for d in (14.0, 26.0):
        self._tree(F.p(a, d))
    self._bench(F.p(0.0, 10.0), F.face + math.pi)
    self._bench(F.p(0.0, 20.0), F.face)
    self._flagpole(F.p(-W / 2 + 3.0, P.depth - 4.0), SHOP[0])
    self._bin(F.p(2.5, 10.0))

  def _shop(self, P: Parcel) -> None:
    """Main Street: 2-4 storey shops with flats over, a continuous frontage"""
    F, rng, W = P.frame, P.rng, P.width
    self.r = rng
    self._begin("shop", P)
    self._flat(F, -W / 2, W / 2, -P.gap, 2.0, L2, PAVER, "plaza", tag="forecourt")
    h = float(rng.choice((7.0, 10.0, 13.0)))
    dep, wd = 20.0, W - 0.1
    sid = self._id("shop")
    col = BRICKS[int(rng.integers(2))] if rng.random() < 0.6 else SIDING[int(rng.integers(len(SIDING)))]
    sc = SHOPS[int(rng.integers(len(SHOPS)))]
    fz = dep / 2 + 0.03
    cmds = [self._box(sid, F.p(0.0, 2.0 + dep / 2), h / 2, wd / 2, h / 2, dep / 2, F.face, col),
            self._kid(sid, "window", "cube", [0, -h / 2 + 1.6, fz], GLASSES[0], scale=[wd / 2 - 0.8, 1.1, 0.03]),
            self._kid(sid, "fascia", "cube", [0, -h / 2 + 3.15, fz + 0.02], sc, scale=[wd / 2 - 0.3, 0.35, 0.04]),
            self._kid(sid, "awning", "cube", [0, -h / 2 + 2.8, fz + 0.7], sc, scale=[wd / 2 - 0.5, 0.05, 0.7]),
            self._kid(sid, "parapet", "cube", [0, h / 2 + 0.3, fz - 0.15], col, scale=[wd / 2, 0.3, 0.15])]
    for fl in range(1, int(h // 3)):
      for k, x in enumerate((-wd * 0.25, wd * 0.25)):
        cmds.append(self._kid(sid, f"w{fl}{k}", "cube", [x, -h / 2 + fl * 3.0 + 1.6, fz], GLASSES[0], scale=[0.6, 0.7, 0.03]))
    if self._add(F.rect(-wd / 2, wd / 2, 2.0, 2.0 + dep), "bldg", cmds, top=h + 0.6, within=P.quad, door=F.p(0.0, 1.7)):
      if rng.random() < 0.3:
        self._prop_box("dumpster", F.p(0.0, 2.0 + dep + 3.0), F.face, 1.0, 0.7, 0.8, _m((0.15, 0.3, 0.2), 0.7), within=P.quad)
    else:
      self._abort(self.g)

  def _midrise_front(self, P: Parcel) -> None:
    """4-5 storey perimeter block: shops on the ground floor, flats over"""
    F, rng, W = P.frame, P.rng, P.width
    self.r = rng
    self._begin("midrise", P)
    self._flat(F, -W / 2, W / 2, -P.gap, 2.0, L2, PAVER, "plaza", tag="forecourt")
    h = float(rng.choice((14.0, 17.0)))
    dep, wd = 22.0, W - 0.2
    mid = self._id("block")
    col = BRICKS[int(rng.integers(2))] if rng.random() < 0.5 else TOWERS[int(rng.integers(len(TOWERS)))]
    sc = SHOPS[int(rng.integers(len(SHOPS)))]
    fz = dep / 2 + 0.03
    cmds = [self._box(mid, F.p(0.0, 2.0 + dep / 2), h / 2, wd / 2, h / 2, dep / 2, F.face, col),
            self._kid(mid, "gf", "cube", [0, -h / 2 + 1.6, fz], GLASSES[1], scale=[wd / 2 - 0.8, 1.2, 0.03]),
            self._kid(mid, "fascia", "cube", [0, -h / 2 + 3.15, fz + 0.02], sc, scale=[wd / 2 - 0.3, 0.3, 0.04]),
            self._kid(mid, "awning", "cube", [0, -h / 2 + 2.85, fz + 0.8], sc, scale=[wd / 2 - 1.0, 0.05, 0.8]),
            self._kid(mid, "parapet", "cube", [0, h / 2 + 0.25, 0], ROOFTOP, scale=[wd / 2, 0.25, dep / 2])]
    for k in range(int((h - 4.0) // 3.2)):
      cmds.append(self._kid(mid, f"b{k}", "cube", [0, -h / 2 + 5.4 + k * 3.2, fz], GLASSES[0], scale=[wd / 2 - 0.8, 0.75, 0.03]))
    if not self._add(F.rect(-wd / 2, wd / 2, 2.0, 2.0 + dep), "bldg", cmds, top=h + 0.5, within=P.quad, door=F.p(0.0, 1.7)):
      self._abort(self.g)

  # ---- 5. interiors ----

  def _interiors(self) -> None:
    for label in sorted(self.blocks):
      self.r = self.lrng[label]
      self._downtown_core(label)
    for label in sorted(self.zone):
      self.r = self.lrng[label]
      z = self.zone[label]
      if z == "lobe":
        self._lobe_interior(label)
      elif z == "loop":
        self._loop_park(label)
    for prog, F, w, d in self.osites:
      self.r = self.vrng
      getattr(self, "_site_" + prog)(F, w, d)
    self._expressway_belts()
    for label in sorted(self.zone):
      if self.zone[label] == "outer":
        self.r = self.lrng[label]
        self._outer_interior(label)

  def _downtown_core(self, label: int) -> None:
    F, LA, LD, prog = self.blocks[label]
    cw = 8.0 if prog == "park" else 6.0
    if prog == "park":
      self._central_park(F, LA, LD, cw)
      return
    rng = self.r
    # quadrants between the alley ring and the cuts, each fronting the vertical cut
    quads = []
    for qa in (0, 1):
      for qd in (0, 1):
        a0, a1 = (43.0, LA / 2 - cw / 2 - 1.0) if qa == 0 else (LA / 2 + cw / 2 + 1.0, LA - 43.0)
        d0, d1 = (43.0, LD / 2 - cw / 2 - 1.0) if qd == 0 else (LD / 2 + cw / 2 + 1.0, LD - 43.0)
        # front on the vertical cut: v points away from it
        if qa == 0:
          Fq = Frame(F.p(a1, (d0 + d1) / 2), -F.ex)
        else:
          Fq = Frame(F.p(a0, (d0 + d1) / 2), F.ex)
        quads.append((Fq, d1 - d0, a1 - a0))
    progs = {"cbd": ["garage", "lot", "office", "plaza"], "civic": ["cityhall", "library", "courtyard", "lot"]}[prog]
    order = [progs[int(i)] for i in rng.permutation(4)]
    for (Fq, w, d), what in zip(quads, order, strict=True):
      self._begin(what)
      for poly in (Fq.rect(-w / 2, w / 2, 0.0, d),):
        self.grid.reserve(poly, 0.0)
      getattr(self, "_core_" + what)(Fq, w, d)

  def _core_lot(self, F: Frame, w: float, d: float) -> None:
    self._surface_lot(F, -w / 2, w / 2, 1.0, d, 0.7, gap=2.0)

  def _core_garage(self, F: Frame, w: float, d: float) -> None:
    """2.8: a 4-level parking garage, its apron on the cut, cars on the roof"""
    rng = self.r
    gw, gd, h = min(60.0, w - 2.0), min(36.0, d - 9.0), 12.0
    gid = self._id("garage")
    cmds = [self._box(gid, F.p(0.0, 8.0 + gd / 2), h / 2, gw / 2, h / 2, gd / 2, F.face, CONCRETE)]
    for lv in range(4):
      for k, (z, half) in enumerate(((gd / 2 + 0.02, gw / 2 - 0.4), (-gd / 2 - 0.02, gw / 2 - 0.4))):
        cmds.append(self._kid(gid, f"o{lv}{k}", "cube", [0, -h / 2 + 1.6 + lv * 3.0, z], DARK, scale=[half, 0.7, 0.03]))
    cmds.append(self._kid(gid, "barrier", "cube", [-gw / 4, -h / 2 + 1.0, gd / 2 + 1.5], SHOPS[3], scale=[1.6, 0.05, 0.05]))
    if not self._add(F.rect(-gw / 2, gw / 2, 8.0, 8.0 + gd), "bldg", cmds, top=h + 1.6, door=F.p(-gw / 4, 7.7)):
      return
    self._flat(F, -gw / 2, gw / 2, -1.0, 7.98, L1, ASPHALT, "path", tag="apron")
    for k in range(6):
      p = F.p(-gw / 2 + 6.0 + k * 3.0, 8.0 + gd / 2 + (5.0 if k % 2 else -5.0))
      kind = "sedan" if rng.random() < 0.7 else "suv"
      cmds = car_lite_commands(self._id("car"), kind, CARS[int(rng.integers(len(CARS)))], round(float(p[0]), 3), round(float(p[1]), 3), F.face)
      cmds[0]["position"][1] = round(cmds[0]["position"][1] + h, 3)
      self._cmds(cmds)

  def _core_office(self, F: Frame, w: float, d: float) -> None:
    rng = self.r
    bw, bd, h = min(50.0, w - 6.0), 18.0, float(rng.choice((19.0, 23.0, 27.0)))
    self._flat(F, -w / 2 + 1.0, w / 2 - 1.0, -1.0, 10.0, L2, PAVER, "plaza", tag="plaza")
    oid = self._id("office")
    col = TOWERS[int(rng.integers(len(TOWERS)))]
    cmds = [self._box(oid, F.p(0.0, 10.0 + bd / 2), h / 2, bw / 2, h / 2, bd / 2, F.face, col),
            self._kid(oid, "canopy", "cube", [0, -h / 2 + 3.2, bd / 2 + 1.5], METAL, scale=[3.0, 0.1, 1.5]),
            self._kid(oid, "cap", "cube", [0, h / 2 + 0.2, 0], ROOFTOP, scale=[bw / 2, 0.2, bd / 2])]
    for side in (1.0, -1.0):
      for k in range(6):
        cmds.append(self._kid(oid, f"g{int(side > 0)}{k}", "cube", [0, -h / 2 + 2.0 + k * (h - 3.0) / 6, side * (bd / 2 + 0.03)],
                              GLASSES[1], scale=[bw / 2 - 0.6, 0.9, 0.03]))
    if not self._add(F.rect(-bw / 2, bw / 2, 10.0, 10.0 + bd), "bldg", cmds, top=h + 0.4, door=F.p(0.0, 9.7)):
      return
    for a in (-bw / 2 + 4.0, -bw / 4, bw / 4, bw / 2 - 4.0):
      self._planter(F.p(a, 6.0))
    for a in (-w / 2 + 4.0, w / 2 - 4.0):
      self._tree(F.p(a, 4.0))

  def _core_plaza(self, F: Frame, w: float, d: float) -> None:
    self._flat(F, -w / 2 + 1.0, w / 2 - 1.0, -1.0, d - 1.0, L2, PAVER, "plaza", tag="plaza")
    self._fountain(F.p(0.0, d / 2), 5.0)
    for a in np.linspace(-w / 2 + 5.0, w / 2 - 5.0, 5):       # a grid of trees in grates, open round the fountain
      for dd in np.linspace(5.0, d - 5.0, 5):
        if abs(a) > 14.0 or abs(dd - d / 2) > 14.0:
          if self._tree(F.p(float(a), float(dd))):
            self._flat(F, float(a) - 0.6, float(a) + 0.6, float(dd) - 0.6, float(dd) + 0.6, L3, DARK, "grate", tag="grate")
    for k in range(4):
      ang = k * math.pi / 2
      self._bench(F.p(math.sin(ang) * 9.0, d / 2 + math.cos(ang) * 9.0), F.face + ang)
    self._bin(F.p(7.0, d / 2 + 7.0))

  def _core_cityhall(self, F: Frame, w: float, d: float) -> None:
    """city hall on a paved plaza with three flagpoles and a fountain, lawn round the back"""
    self._flat(F, -w / 2 + 1.0, w / 2 - 1.0, 20.0, d - 1.0, L0, LAWN, "lawn", tag="lawn")
    self._flat(F, -w / 2 + 1.0, w / 2 - 1.0, -1.0, 19.98, L2, PAVER, "plaza", tag="plaza")
    bw, bd, h = min(50.0, w - 6.0), 25.0, 15.0
    hid = self._id("cityhall")
    cmds = [self._box(hid, F.p(0.0, 20.0 + bd / 2), h / 2, bw / 2, h / 2, bd / 2, F.face, STONE),
            self._kid(hid, "pediment", "cube", [0, h / 2 + 0.6, bd / 2 - 2.0], STONE, scale=[bw / 4, 0.6, 2.0]),
            self._kid(hid, "dome", "sphere", [0, h / 2 + 1.5, 0], _m((0.45, 0.55, 0.5), 0.5, 0.4), scale=[5.0, 4.0, 5.0], segments=16)]
    for k in range(6):
      cmds.append(self._kid(hid, f"col{k}", "cylinder", [-7.5 + k * 3.0, -h / 2 + 5.0, bd / 2 + 1.5], STONE, radius=0.45, height=10.0, segments=10))
    for k in range(4):
      cmds.append(self._kid(hid, f"w{k}", "cube", [0, -h / 2 + 2.5 + k * 3.2, bd / 2 + 0.03], GLASSES[0], scale=[bw / 2 - 1.0, 0.7, 0.03]))
    if not self._add(F.rect(-bw / 2, bw / 2, 20.0, 20.0 + bd), "bldg", cmds, top=h + 5.5, door=F.p(0.0, 19.7)):
      return
    for k, a in enumerate((-6.0, 0.0, 6.0)):
      self._flagpole(F.p(a, 4.0), (SHOP[0], TRIM["color"], SHOP[1])[k])
    self._fountain(F.p(0.0, 11.0), 5.0)
    for a in (-w / 2 + 4.0, w / 2 - 4.0):
      self._tree(F.p(a, 8.0))
      self._bench(F.p(a * 0.7, 14.0), F.face)

  def _core_library(self, F: Frame, w: float, d: float) -> None:
    """a library on the cut and a church behind it"""
    self._flat(F, -w / 2 + 1.0, w / 2 - 1.0, -1.0, 6.0, L2, PAVER, "plaza", tag="plaza")
    lid = self._id("library")
    cmds = [self._box(lid, F.p(-w / 2 + 19.0, 6.0 + 12.5), 5.0, 17.5, 5.0, 12.5, F.face, BRICKS[1]),
            self._kid(lid, "glass", "cube", [0, -1.5, 12.53], GLASSES[1], scale=[15.0, 2.5, 0.03]),
            self._kid(lid, "cap", "cube", [0, 5.2, 0], ROOFTOP, scale=[17.5, 0.2, 12.5])]
    self._add(F.rect(-w / 2 + 1.5, -w / 2 + 36.5, 6.0, 31.0), "bldg", cmds, top=10.4, door=F.p(-w / 2 + 19.0, 5.7))
    # the church: nave along the depth, a path from the plaza
    ca = w / 2 - 10.0
    self._flat(F, ca - 1.5, ca + 1.5, 5.9, 18.0, L2, PAVER, "path", tag="path")
    cid = self._id("church")
    cmds = [self._box(cid, F.p(ca, 18.0 + 15.0), 5.5, 7.0, 5.5, 15.0, F.face, STONE)]
    r = 7.4 / math.sqrt(2.0)
    cmds.append(self._kid(cid, "roof", "cube", [0, 5.5, 0], ROOFS[0], rotation=[0, 0, 45], scale=[r, r, 15.2]))
    cmds.append(self._kid(cid, "tower", "cube", [0, 6.5, 15.0 - 2.5], STONE, scale=[2.5, 12.0, 2.5]))
    cmds.append(self._kid(cid, "spire", "cone", [0, 18.5 + 4.0, 15.0 - 2.5], ROOFS[0], radius=2.6, height=8.0, segments=8))
    cmds.append(self._kid(cid, "window", "cube", [0, 2.0, 15.03], GLASSES[0], scale=[1.2, 2.0, 0.03]))
    if self._add(F.rect(ca - 7.0, ca + 7.0, 18.0, 48.0), "bldg", cmds, top=34.5, door=F.p(ca, 17.7)):
      for dd in (24.0, 32.0, 40.0):
        self._tree(F.p(ca - 11.0, dd))

  def _core_courtyard(self, F: Frame, w: float, d: float) -> None:
    """downtown residential: a U of 5-storey flats round a courtyard garden, its residents' lot in front"""
    self._surface_lot(F, -w / 2 + 1.0, w / 2 - 1.0, 1.0, 19.0, 0.6, gap=2.0)
    h = 16.0
    bars = [(-w / 2 + 1.5, w / 2 - 1.5, 19.05, 33.0, 0), (-w / 2 + 1.5, -w / 2 + 14.5, 33.05, d - 1.0, 1),
            (w / 2 - 14.5, w / 2 - 1.5, 33.05, d - 1.0, 2)]
    col = BRICKS[0]
    for a0, a1, d0, d1, k in bars:
      bid = self._id("flats")
      ca, cd = (a0 + a1) / 2, (d0 + d1) / 2
      cmds = [self._box(bid, F.p(ca, cd), h / 2, (a1 - a0) / 2, h / 2, (d1 - d0) / 2, F.face, col),
              self._kid(bid, "cap", "cube", [0, h / 2 + 0.2, 0], ROOFTOP, scale=[(a1 - a0) / 2, 0.2, (d1 - d0) / 2])]
      if k == 0:
        for f in range(4):
          cmds.append(self._kid(bid, f"g{f}", "cube", [0, -h / 2 + 2.0 + f * 3.4, (d1 - d0) / 2 + 0.03], GLASSES[0],
                                scale=[(a1 - a0) / 2 - 1.0, 0.8, 0.03]))
          cmds.append(self._kid(bid, f"c{f}", "cube", [0, -h / 2 + 2.0 + f * 3.4, -(d1 - d0) / 2 - 0.03], GLASSES[0],
                                scale=[(a1 - a0) / 2 - 8.0, 0.8, 0.03]))
        door = F.p(0.0, 18.8)
      else:
        door = None
      self._add(F.rect(a0, a1, d0, d1), "bldg", cmds, top=h + 0.4, door=door)
    self._flat(F, -w / 2 + 14.6, w / 2 - 14.6, 33.1, d - 1.0, L0, LAWN, "lawn", tag="lawn")
    for _ in range(5):
      self._tree(F.p(float(self.r.uniform(-w / 2 + 18.0, w / 2 - 18.0)), float(self.r.uniform(37.0, d - 4.0))))
    self._bench(F.p(0.0, 36.0), F.face + math.pi)

  def _fountain(self, p, r: float) -> bool:
    fid = self._id("fountain")
    cmds = [self._cyl(fid, p, 0.25, r, 0.5, STONE, seg=24),
            self._kid(fid, "water", "cylinder", [0, 0.2, 0], WATER, radius=r - 0.4, height=0.2, segments=24),
            self._kid(fid, "spout", "cylinder", [0, 1.0, 0], STONE, radius=0.4, height=2.0, segments=10),
            self._kid(fid, "jet", "cone", [0, 2.6, 0], _m((0.75, 0.85, 0.9), 0.1), radius=0.6, height=1.4, segments=10)]
    return self._add(_rect(p, 0.0, r, r), "prop", cmds, top=3.3, avoid=("water", "court", "lot"))

  def _central_park(self, F: Frame, LA: float, LD: float, cw: float) -> None:
    """2.5: the town square inside the park block's ring: four lawns, the promenades (the block's cuts) meeting at a
    paved plaza with a clock tower, diagonal paths, a pond, a bandstand, a playground, rows of trees"""
    rng = self.r
    self._begin("park")
    a_c, d_c = LA / 2, LD / 2
    pc = F.p(a_c, d_c)
    self._flat(F, a_c - 20.0, a_c + 20.0, d_c - 20.0, d_c + 20.0, L2, PAVER, "plaza", tag="plaza")
    tid = self._id("clock")
    cmds = [self._box(tid, pc, 9.0, 2.0, 9.0, 2.0, F.face, STONE),
            self._kid(tid, "roof", "cone", [0, 10.5, 0], ROOFS[0], radius=3.0, height=3.0, segments=4, rotation=[0, 45, 0])]
    for k in range(4):
      ang = k * math.pi / 2
      cmds.append(self._kid(tid, f"face{k}", "cylinder", [math.sin(ang) * 2.03, 6.5, math.cos(ang) * 2.03], TRIM, radius=1.1, height=0.04,
                            segments=16, rotation=[90, deg(ang), 0]))
    self._add(_rect(pc, F.face, 2.0, 2.0), "bldg", cmds, top=22.0, allow=("lawn", "plaza", "path"))
    quads = []
    for qa in (0, 1):
      for qd in (0, 1):
        a0, a1 = (43.0, a_c - cw / 2 - 0.5) if qa == 0 else (a_c + cw / 2 + 0.5, LA - 43.0)
        d0, d1 = (43.0, d_c - cw / 2 - 0.5) if qd == 0 else (d_c + cw / 2 + 0.5, LD - 43.0)
        quads.append((a0, a1, d0, d1, qa, qd))
        self._flat(F, a0, a1, d0, d1, L0, LAWN, "lawn", tag="lawn")
        self.grid.reserve(F.rect(a0, a1, d0, d1), 0.0)
    features = [int(i) for i in rng.permutation(4)]
    for k, (a0, a1, d0, d1, qa, qd) in enumerate(quads):
      # the diagonal: from the plaza corner out to the ring corner
      ca = (a_c + (20.5 if qa else -20.5), d_c + (20.5 if qd else -20.5))
      oa = (a1 - 6.0 if qa else a0 + 6.0, d1 - 6.0 if qd else d0 + 6.0)
      self._strip(F.p(*ca), F.p(*oa), 4.0, L2, GRAVEL, "path")
      # each lawn its own feature, set off the diagonal (which runs through the lawn's middle)
      sa, sd_ = (1.0 if qa else -1.0), (1.0 if qd else -1.0)
      fp = F.p((a0 + a1) / 2 + sa * 13.0, (d0 + d1) / 2 - sd_ * 13.0)
      what = features[k]
      if what == 0:
        dv = F.ex * sa + F.v * sd_                      # long axis along the diagonal: local x = (cos, -sin)
        self._pond(fp, 14.0, 8.0, math.atan2(-dv[1], dv[0]))
      elif what == 1:
        bid = self._id("bandstand")
        cmds = [self._cyl(bid, fp, 0.4, 5.0, 0.8, STONE, seg=8),
                self._kid(bid, "roof", "cone", [0, 4.6, 0], ROOFS[1], radius=5.6, height=2.4, segments=8)]
        for j in range(8):
          ang = j * math.pi / 4
          cmds.append(self._kid(bid, f"post{j}", "cylinder", [math.sin(ang) * 4.6, 1.8, math.cos(ang) * 4.6], TRIM, radius=0.12, height=2.8, segments=6))
        self._add(_rect(fp, 0.0, 5.0, 5.0), "prop", cmds, top=5.8, avoid=SOFT)
      elif what == 2:
        self._playground(Frame(fp, F.v))
      else:
        oid = self._id("obelisk")
        self._add(_rect(fp, 0.0, 1.2, 1.2), "prop", [self._box(oid, fp, 0.5, 1.2, 0.5, 1.2, F.face, STONE),
                                                   self._kid(oid, "shaft", "cube", [0, 6.0, 0], STONE, scale=[0.6, 5.5, 0.6])], top=12.0)
      # perimeter trees: a double row along the outer edges every 10 m
      ea = a1 - 3.0 if qa else a0 + 3.0
      ed = d1 - 3.0 if qd else d0 + 3.0
      for t in np.arange(a0 + 5.0, a1 - 4.0, 10.0):
        self._tree(F.p(float(t), ed), size=2)
        self._tree(F.p(float(t) + 5.0, ed + (-6.0 if qd else 6.0)), size=1)
      for t in np.arange(d0 + 5.0, d1 - 4.0, 10.0):
        self._tree(F.p(ea, float(t)), size=2)
      # two clumps of trees on each lawn, 7 m apart at least
      clumps = [F.p(float(rng.uniform(a0 + 14.0, a1 - 14.0)), float(rng.uniform(d0 + 14.0, d1 - 14.0))) for _ in range(2)]
      placed = []
      for _ in range(40):
        p = clumps[len(placed) % 2] + rng.normal(0.0, 7.0, 2)
        rel = p - F.o
        if not (a0 + 4.0 < rel @ F.ex < a1 - 4.0 and d0 + 4.0 < rel @ F.v < d1 - 4.0) or any(np.linalg.norm(p - q) < 7.0 for q in placed):
          continue
        if self._tree(p, conifer=rng.random() < 0.15):
          placed.append(p)
        if len(placed) >= 16:
          break
    # benches and lamps along the promenades, bins at the plaza
    for t in np.arange(-60.0, 61.0, 15.0):
      if abs(t) < 22.0:
        continue
      self._bench(F.p(a_c + cw / 2 + 0.8, d_c + t), F.face - math.pi / 2)
      self._bench(F.p(a_c + t, d_c - cw / 2 - 0.8), F.face)
      self._lamp(F.p(a_c - cw / 2 - 0.6, d_c + t), F.face, h=4.0, park=True)
      self._lamp(F.p(a_c + t, d_c + cw / 2 + 0.6), F.face, h=4.0, park=True)
    for a, d in ((-14.0, -14.0), (14.0, -14.0), (-14.0, 14.0), (14.0, 14.0)):
      self._bin(F.p(a_c + a, d_c + d))
      self._bench(F.p(a_c + a * 1.2, d_c + d * 0.6), F.face + (math.pi / 2 if a > 0 else -math.pi / 2))
    # lamps and benches along the perimeter walk (the ring), on the park side
    for t in np.arange(55.0, LA - 54.0, 25.0):
      for dd, turn in ((42.6, 0.0), (LD - 42.6, math.pi)):
        if abs(t - a_c) > cw:
          self._lamp(F.p(float(t), dd), F.face, h=4.0, park=True)
          self._bench(F.p(float(t) + 6.0, dd + (0.8 if turn == 0.0 else -0.8)), F.face + turn)
    for t in np.arange(55.0, LD - 54.0, 25.0):
      for aa in (42.6, LA - 42.6):
        if abs(t - d_c) > cw:
          self._lamp(F.p(aa, float(t)), F.face, h=4.0, park=True)

  def _pond(self, p, rx: float, rz: float, psi: float) -> bool:
    pid = self._id("pond")
    cmds = [{"action": "create", "type": "cylinder", "id": pid, "position": [round(float(p[0]), 3), 0.0, round(float(p[1]), 3)],
             "rotation": [0, deg(psi), 0], "radius": 1.0, "height": 0.12, "segments": 32, "scale": [rx, 1.0, rz], **CONCRETE},
            self._kid(pid, "water", "cylinder", [0, 0.01, 0], WATER, radius=1.0, height=0.12, segments=32, scale=[rx - 1.0, 1.0, rz - 1.0]),
            self._kid(pid, "island", "cylinder", [rx * 0.3, 0.05, 0], LAWN, radius=1.0, height=0.2, segments=12,
                      scale=[min(3.0, rx * 0.2), 1.0, min(3.0, rz * 0.25)])]
    if not self._add(_rect(p, psi, rx, rz), "flat", cmds, purpose="water", y=0.07, key="water"):
      return False
    isl = np.asarray(p) + np.array([math.cos(psi), -math.sin(psi)]) * rx * 0.3
    self._tree(isl, avoid=(), size=1)
    return True

  def _playground(self, F: Frame) -> None:
    """SAND pad 16 x 12 with swings, a slide, a climbing frame and a see-saw (PLAY colours)"""
    self._flat(F, -8.0, 8.0, -6.0, 6.0, L1, SAND, "court", tag="sand")
    av = ("water", "path", "lot")
    self._prop_box("swings", F.p(-4.0, -2.5), F.face, 2.0, 0.06, 0.06, PLAY[0], y0=2.3, avoid=av,
                   kids=[(f"leg{i}", "cube", [x, -1.15, 0], METAL, {"scale": [0.06, 1.15, 0.7]}) for i, x in enumerate((-1.95, 1.95))] +
                        [(f"seat{i}", "cube", [x, -1.8, 0], DARK, {"scale": [0.25, 0.03, 0.12]}) for i, x in enumerate((-0.8, 0.8))])
    self._prop_box("slide", F.p(4.0, -2.5), F.face, 0.6, 1.0, 0.6, PLAY[1], avoid=av,
                   kids=[("chute", "cube", [0, -0.2, 1.9], PLAY[2], {"rotation": [35, 0, 0], "scale": [0.35, 0.04, 1.6]})])
    self._prop_box("frame", F.p(-3.5, 3.0), F.face, 1.5, 0.75, 1.5, PLAY[2], y0=0.0, avoid=av,
                   kids=[("deck", "cube", [0, 0.8, 0], PLAY[1], {"scale": [1.5, 0.05, 1.5]})])
    self._prop_box("seesaw", F.p(4.0, 3.5), F.face, 0.15, 0.25, 0.15, METAL, avoid=av,
                   kids=[("plank", "cube", [0, 0.27, 0], PLAY[0], {"rotation": [8, 0, 0], "scale": [0.15, 0.04, 2.0]})])

  def _play(self, p, psi) -> None:
    self._prop_box("climb", p, psi, 1.2, 0.7, 1.2, PLAY[int(self.r.integers(3))], avoid=("water", "path", "lot"),
                   kids=[("deck", "cube", [0, 0.75, 0], PLAY[1], {"scale": [1.2, 0.05, 1.2]})])

  def _court_bball(self, F: Frame) -> None:
    """a basketball court 28 x 15 with lines and hoops"""
    self._flat(F, -14.0, 14.0, -7.5, 7.5, L1, CONCRETE, "court", tag="court")
    for a0, a1, d0, d1 in ((-14.0, 14.0, -7.5, -7.4), (-14.0, 14.0, 7.4, 7.5), (-14.0, -13.9, -7.5, 7.5), (13.9, 14.0, -7.5, 7.5),
                           (-0.05, 0.05, -7.5, 7.5)):
      self._flat(F, a0, a1, d0, d1, L3, LINE, "line", tag="line")
    for k, a in enumerate((-14.8, 14.8)):
      hid = self._id("hoop")
      self._add(_rect(F.p(a, 0.0), 0.0, 0.2, 0.2), "prop",
                [self._cyl(hid, F.p(a, 0.0), 1.6, 0.08, 3.2, METAL, seg=8),
                 self._kid(hid, "board", "cube", [-0.6 if k else 0.6, 1.6, 0], TRIM, rotation=[0, deg(F.face) + 90, 0], scale=[0.9, 0.55, 0.03])],
                top=3.8, avoid=("water", "lot"))

  def _surface_lot(self, F: Frame, a0, a1, d0, d1, occupancy, gap=1.0, within=None, entry=True, max_bays=120) -> int:
    """2.7: an asphalt lot in modules (5.5 bay + 7 aisle + 5.5 bay), a circulation lane at the a0 end, its entry apron
    from d0 back `gap` m to the access edge; bay lines, end islands with trees, part filled with cars"""
    rng = self.r
    if a1 - a0 < 20.0 or d1 - d0 < 18.0:
      return 0
    if not self._flat(F, a0, a1, d0, d1, L1, ASPHALT, "lot", tag="lot", within=None):
      return 0
    if entry:
      self._flat(F, a0, a0 + 7.0, d0 - gap, d0 + 0.5, L1, ASPHALT, "path", tag="apron")
      self._prop_box("sign", F.p(a0 + 8.5, d0 - gap + 1.0), F.face, 0.6, 0.4, 0.05, SHOPS[0], y0=1.6, avoid=("water", "court", "lot"),
                     kids=[("pole", "cube", [0, -1.2, 0], METAL, {"scale": [0.05, 0.8, 0.05]})])
    bays = 0
    for m in range(int((d1 - d0) // 18.0)):
      for row, (r0, r1) in enumerate(((d0 + 18.0 * m, d0 + 18.0 * m + 5.5), (d0 + 18.0 * m + 12.5, d0 + 18.0 * m + 18.0))):
        n = int((a1 - a0 - 7.5) // 2.7)
        for b in range(n):
          if bays >= max_bays:
            break
          ba = a0 + 7.0 + 2.7 * b
          bay = F.rect(ba, ba + 2.7, r0, r1)
          if self._pier_d(bay) < PIER_R + PIER_CLEAR:
            continue
          if b % 11 == 10 or b == n - 1:
            if self._flat(F, ba + 0.2, ba + 2.5, r0 + 0.2, r1 - 0.2, L2, LAWN, "lawn", tag="island", check=True):
              self._tree(F.p(ba + 1.35, (r0 + r1) / 2), avoid=("water", "path", "court"))
            continue
          self._flat(F, ba - 0.05, ba + 0.05, r0, r1, L3, LINE, "line", tag="bay")
          bays += 1
          if rng.random() < occupancy:
            self._car(F.p(ba + 1.35, (r0 + r1) / 2), F.face + (math.pi if row == 0 else 0.0), avoid=("water", "court", "path"))
    self.stats["bays"] += bays
    return bays

  # lobes, loop, outer

  def _lobe_interior(self, label: int) -> None:
    gr = self.grid
    side = max((sd for sd in self.sides if sd.label == label and not sd.curved), key=lambda sd: sd.len, default=None)
    if side is None:
      return
    F = side.frame(side.len / 2)
    target = self.lcent[label]
    if self.ldir[label] == "E":
      site = gr.fit_rect(gr.free_mask(label), F, [(72.0, 76.0), (64.0, 76.0), (76.0, 64.0)], target)
      if site:
        self._school(*site)
      else:
        self.notes.append(f"no room for the school in lobe {label}")
    # under the deck: a park-and-ride lot along it
    near = np.isfinite(gr.cap)
    for _ in range(6):            # 12 m either side of the deck
      near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0) | np.roll(near, 1, 1) | np.roll(near, -1, 1)
    mask = gr.free_mask(label) & near
    if mask.sum() > 50:
      for sz in ([(34.0, 120.0), (34.0, 90.0), (34.0, 60.0), (120.0, 34.0), (90.0, 34.0), (60.0, 34.0)]):
        site = gr.fit_rect(mask, F, [sz], target)
        if site and self._park_and_ride(*site):
          self.notes.append(f"park and ride {sz} in lobe {label}")
          break
    mask = gr.free_mask(label)
    dims = (140.0, 120.0, 100.0, 88.0, 76.0, 64.0, 52.0, 44.0, 36.0, 30.0)
    sizes = sorted(((w, h) for w in dims for h in dims if w * h >= 1200.0 and max(w, h) <= 2.5 * min(w, h)), key=lambda s: (-s[0] * s[1], s))
    site = gr.fit_rect(mask, F, sizes, target)
    if site:
      self._neighbourhood_park(*site, label)
    # smaller amenities where there is room and a path to them: tennis, a ball court, allotments
    for what in ("tennis", "bball", "garden", "bball"):
      site = gr.fit_rect(gr.free_mask(label), F, {"tennis": [(40.0, 30.0), (30.0, 40.0)], "bball": [(34.0, 21.0), (21.0, 34.0)],
                                                  "garden": [(36.0, 24.0), (24.0, 36.0)]}[what], target)
      if site:
        self._amenity(what, *site)
    # what's left: copses
    self._copses(label, 9.0, 0.55, cap=80)

  def _amenity(self, what: str, F: Frame, w: float, d: float) -> bool:
    """a fenced pair of tennis courts, a basketball court or allotments in a lobe's leftover land, with a path to it"""
    g = self._begin(what)
    if d > w:
      F, w, d = F.sub(0.0, 0.0, math.pi / 2), d, w
    site = F.rect(-w / 2, w / 2, -d / 2, d / 2)
    self.grid.reserve(site, 0.0)
    if what == "tennis":
      for dc in (-6.2, 6.2):     # two courts side by side, their length along a
        self._flat(F, -11.9, 11.9, dc - 5.5, dc + 5.5, L1, _m((0.25, 0.45, 0.42), 0.8), "court", tag="tennis")
        for a0, a1, d0, d1 in ((-11.9, 11.9, dc - 5.5, dc - 5.4), (-11.9, 11.9, dc + 5.4, dc + 5.5), (-11.9, -11.8, dc - 5.5, dc + 5.5),
                               (11.8, 11.9, dc - 5.5, dc + 5.5)):
          self._flat(F, a0, a1, d0, d1, L3, LINE, "line", tag="line")
        self._prop_box("net", F.p(0.0, dc), F.face, 0.03, 0.5, 5.9, DARK, avoid=("water", "lot"))
      for p0, p1 in ((F.p(-w / 2 + 0.5, -d / 2 + 0.5), F.p(w / 2 - 0.5, -d / 2 + 0.5)), (F.p(-w / 2 + 0.5, d / 2 - 0.5), F.p(w / 2 - 0.5, d / 2 - 0.5))):
        self._fence(p0, p1, 3.0, FENCE_IRON, avoid=("water", "lot"))
      start = F.p(0.0, -d / 2 + 0.6)
    elif what == "bball":
      self._court_bball(F)
      start = F.p(0.0, -d / 2 + 0.3)
    else:
      self._flat(F, -w / 2 + 0.5, w / 2 - 0.5, -d / 2 + 0.5, d / 2 - 0.5, L2, GRAVEL, "path", tag="garden")
      for a in np.arange(-w / 2 + 3.0, w / 2 - 2.0, 3.5):
        for dd in (-d / 2 + 4.0, d / 2 - 4.0):
          self._prop_box("bed", F.p(float(a), dd), F.face, 1.0, 0.25, 1.6, FENCE_WOOD, avoid=("water", "court", "lot"),
                         kids=[("soil", "cube", [0, 0.26, 0], _m((0.25, 0.18, 0.12), 0.95), {"scale": [0.9, 0.02, 1.5]})])
      self._prop_box("shed", F.p(w / 2 - 3.0, 0.0), F.face, 1.4, 1.1, 1.2, FENCE_WOOD, avoid=("water", "court", "lot"))
      start = F.p(0.0, -d / 2 + 0.6)
    if not self._link(start, 2.5, 80.0, kinds=("corridor", "park"), site=site):
      self._abort(g)
      return False
    return True

  def _site_works(self, F: Frame, w: float, d: float) -> None:
    """warehouses with dock doors on a truck apron, trucks at the docks, a fence along the back"""
    rng = self.r
    g = self._begin("works")
    ad = 24.0
    if not self._flat(F, -w / 2 + 0.5, w / 2 - 0.5, -d / 2, -d / 2 + ad, L2, CONCRETE, "path", tag="apron"):
      self._abort(g)
      return
    n = max(1, int((w - 2.0) // 62.0)) if w >= 62.0 else 1
    bw = min(60.0, (w - 2.0) / n - 2.0)
    bd = min(30.0, d - ad - 1.5)
    for k in range(n):
      a = -w / 2 + 1.0 + (w - 2.0) * (k + 0.5) / n
      wid = self._id("warehouse")
      cmds = [self._box(wid, F.p(a, -d / 2 + ad + 0.05 + bd / 2), 4.5, bw / 2, 4.5, bd / 2, F.face, _m((0.72, 0.72, 0.70), 0.7, 0.2)),
              self._kid(wid, "band", "cube", [0, 3.6, bd / 2 + 0.03], SHOPS[int(rng.integers(len(SHOPS)))], scale=[bw / 2, 0.4, 0.03])]
      docks = [-bw / 2 + 5.0 + 7.0 * j for j in range(int((bw - 6.0) // 7.0))]
      for j, x in enumerate(docks):
        cmds.append(self._kid(wid, f"dock{j}", "cube", [x, -2.5, bd / 2 + 0.03], DARK, scale=[1.5, 1.8, 0.03]))
      if not self._add(F.rect(a - bw / 2, a + bw / 2, -d / 2 + ad + 0.05, -d / 2 + ad + 0.05 + bd), "bldg", cmds, top=9.0,
                       door=F.p(a + docks[0], -d / 2 + ad - 0.3)):
        continue
      for x in docks[::2][:3]:
        if rng.random() < 0.6:
          tid = self._id("truck")
          p = F.p(a + x, -d / 2 + ad - 5.2)
          cmds = car_lite_commands(tid, "truck", [0.92, 0.92, 0.9], round(float(p[0]), 3), round(float(p[1]), 3), F.face)
          self._add(_rect(F.p(a + x, -d / 2 + ad - 6.1), F.face, 1.3, 4.95), "prop", cmds, top=3.6, avoid=("water", "court"))
    self._fence(F.p(-w / 2 + 0.5, d / 2 - 0.5), F.p(w / 2 - 0.5, d / 2 - 0.5), 2.2, METAL, avoid=("water", "lot", "path"))

  def _school(self, F: Frame, w: float, d: float) -> None:
    """an elementary school: a brick bar, its drop-off lot, a pitch with goals, two floodlights"""
    self._begin("school")
    self.grid.reserve(F.rect(-w / 2, w / 2, -d / 2, d / 2), 0.0)
    sid = self._id("school")
    bw = min(60.0, w - 4.0)
    cmds = [self._box(sid, F.p(0.0, -d / 2 + 9.0), 4.0, bw / 2, 4.0, 9.0, F.face, BRICKS[0]),
            self._kid(sid, "cap", "cube", [0, 4.2, 0], ROOFTOP, scale=[bw / 2, 0.2, 9.0]),
            self._kid(sid, "canopy", "cube", [0, -0.8, 9.0 + 1.5], METAL, scale=[4.0, 0.1, 1.5])]
    for k in range(2):
      cmds.append(self._kid(sid, f"g{k}", "cube", [0, -2.4 + k * 3.4, 9.03], GLASSES[0], scale=[bw / 2 - 1.0, 0.8, 0.03]))
      cmds.append(self._kid(sid, f"b{k}", "cube", [0, -2.4 + k * 3.4, -9.03], GLASSES[0], scale=[bw / 2 - 1.0, 0.8, 0.03]))
    if not self._add(F.rect(-bw / 2, bw / 2, -d / 2 + 0.5, -d / 2 + 18.0), "bldg", cmds, top=8.4, door=F.p(0.0, -d / 2 + 18.3)):
      return
    self._flat(F, -bw / 2, bw / 2, -d / 2 + 18.05, -d / 2 + 22.0, L2, CONCRETE, "path", tag="yard")
    site = F.rect(-w / 2, w / 2, -d / 2, d / 2)
    if not any(self._link(F.p(a, dd), 4.0, 90.0, site=site) for a, dd in ((-bw / 2 - 0.3, -d / 2 + 20.0), (bw / 2 + 0.3, -d / 2 + 20.0))):
      self.notes.append("school: no path to it")
    pw, pd = min(70.0, w - 2.0), min(45.0, d - 26.0)
    pc = -d / 2 + 23.0 + pd / 2
    if pd > 25.0:
      self._flat(F, -pw / 2, pw / 2, pc - pd / 2, pc + pd / 2, L1, PITCH, "court", tag="pitch")
      for a0, a1, d0, d1 in ((-pw / 2 + 2, pw / 2 - 2, pc - pd / 2 + 2, pc - pd / 2 + 2.1), (-pw / 2 + 2, pw / 2 - 2, pc + pd / 2 - 2.1, pc + pd / 2 - 2),
                             (-0.05, 0.05, pc - pd / 2 + 2, pc + pd / 2 - 2)):
        self._flat(F, a0, a1, d0, d1, L3, LINE, "line", tag="line")
      for k, a in enumerate((-pw / 2 + 2.5, pw / 2 - 2.5)):
        self._prop_box("goal", F.p(a, pc), F.face + math.pi / 2, 3.6, 0.06, 0.06, TRIM, y0=2.3, avoid=("water", "lot", "path"),
                       kids=[(f"p{i}", "cube", [x, -1.17, 0], TRIM, {"scale": [0.06, 1.17, 0.06]}) for i, x in enumerate((-3.6, 3.6))])
        lid = self._id("floodlight")
        lp = F.p(a * 1.03, pc + pd / 2 + 1.0 if k else pc - pd / 2 - 1.0)
        self._add(_rect(lp, 0.0, 0.3, 0.3), "prop", [self._cyl(lid, lp, 7.5, 0.18, 15.0, METAL, seg=8),
                                                    self._kid(lid, "lamps", "cube", [0, 7.6, 0], TRIM, scale=[1.0, 0.5, 0.2])], top=15.6)

  def _park_and_ride(self, F: Frame, w: float, d: float) -> bool:
    """a lot under the deck, rows along it, its apron out to the nearest street"""
    g = self._begin("parkride")
    along = d >= w
    Fl = F if along else F.sub(0.0, 0.0, math.pi / 2)
    lw, ld = (w, d) if along else (d, w)
    # rows along the deck: modules across the narrow way
    Fr = Fl.sub(0.0, 0.0, -math.pi / 2)        # v across the deck, a along it
    if self._surface_lot(Fr, -ld / 2, ld / 2, -lw / 2, lw / 2, 0.5, entry=False) == 0:
      self._abort(g)
      return False
    self.grid.reserve(F.rect(-w / 2, w / 2, -d / 2, d / 2), 0.0)
    ends = [Fr.p(-ld / 2 + 3.5, -lw / 2), Fr.p(-ld / 2 + 3.5, lw / 2), Fr.p(ld / 2 - 3.5, -lw / 2), Fr.p(ld / 2 - 3.5, lw / 2)]
    if not any(self._apron(p) for p in ends):
      self._abort(g)
      return False
    return True

  def _apron(self, p: np.ndarray, maxlen: float = 45.0) -> bool:
    """a 7 m drive from p out to the nearest at-grade street's footway"""
    c = self.centre_hash.near(p, p, maxlen + self.foot)
    if len(c) == 0:
      return False
    q = c[int(np.argmin(np.linalg.norm(c - p, axis=1)))]
    u = (q - p) / max(np.linalg.norm(q - p), 1e-6)
    end = q - u * (self.foot + 0.05)
    if float(np.linalg.norm(end - p)) > maxlen:
      return False
    pts = p + np.outer(np.linspace(0.0, 1.0, 20), end - p)
    if np.any(self.grid.occ_at(pts[3:-3])):
      return False
    return self._strip(p - u * 0.5, end, 7.0, L1, ASPHALT, "path", tag="apron")

  def _link(self, p: np.ndarray, width: float, maxlen: float = 60.0, kinds=("corridor",), site=None) -> bool:
    """a path from p straight to the nearest corridor flat (court, park link, alley) it can reach over unreserved land"""
    cands = []
    for sh in self.index.query(p - maxlen, p + maxlen):
      if sh.cat == "flat" and sh.purpose in WALK and sh.g != self.g and self.groups[sh.g].kind in kinds:
        q = _nearest_on(p, sh.poly)
        cands.append((float(np.linalg.norm(q - p)), tuple(q)))
    for dq, q in sorted(cands)[:6]:
      q = np.array(q)
      if dq > maxlen:
        break
      if dq < 0.3:
        return True
      u = (q - p) / dq
      mid = p + np.outer(np.linspace(0.0, 1.0, int(dq) + 2), q - p)
      if site is not None:        # the site's own reserved land
        mid = mid[~_inside(mid, site, -1.5)]
      if len(mid) > 2 and np.any(self.grid.occ_at(mid[:-2])):
        continue
      if self._strip(p - u * 0.5, q + u * 0.5, width, L2, CONCRETE, "path", tag="link"):
        return True
    return False

  def _neighbourhood_park(self, F: Frame, w: float, d: float, label: int) -> None:
    """2.6: lawn, a gravel loop, entries from the park links, a pond or a fountain, a playground, a court, tree clumps"""
    rng = self.r
    self._begin("park")
    self.grid.reserve(F.rect(-w / 2, w / 2, -d / 2, d / 2), 0.0)
    self._flat(F, -w / 2, w / 2, -d / 2, d / 2, L0, LAWN, "lawn", tag="lawn")
    li = 5.0
    for a0, a1, d0, d1 in ((-w / 2 + li, w / 2 - li, -d / 2 + li, -d / 2 + li + 2.5), (-w / 2 + li, w / 2 - li, d / 2 - li - 2.5, d / 2 - li),
                           (-w / 2 + li, -w / 2 + li + 2.5, -d / 2 + li, d / 2 - li), (w / 2 - li - 2.5, w / 2 - li, -d / 2 + li, d / 2 - li)):
      self._flat(F, a0, a1, d0, d1, L2, GRAVEL, "path", tag="path")
    # entries: from the loop straight to every park link (or the nearest walk) with gate piers
    ents = [e for e in self.links.get(label, []) if np.linalg.norm(e - F.o) < max(w, d) * 0.5 + 60.0]
    for e in ents:
      rel = e - F.o
      a, dd = float(rel @ F.ex), float(rel @ F.v)
      a = min(max(a, -w / 2 + li + 1.25), w / 2 - li - 1.25)
      dd = min(max(dd, -d / 2 + li + 1.25), d / 2 - li - 1.25)
      start = F.p(a, dd)
      self._strip(start, e, 2.5, L2, GRAVEL, "path", tag="entry")
    self._link(F.p(0.0, -d / 2 + li), 2.5, site=F.rect(-w / 2, w / 2, -d / 2, d / 2))
    cx = 0.0
    big = w >= 60.0 and d >= 60.0
    if big:
      self._pond(F.p(cx, d * 0.12), min(w, d) * 0.16, min(w, d) * 0.11, F.face)
    else:
      self._fountain(F.p(cx, 0.0), 3.0)
    self._playground(Frame(F.p(-w / 2 + li + 13.0, -d / 2 + li + 12.0), F.v))
    if w >= 52.0 and d >= 88.0:
      self._court_bball(Frame(F.p(w / 2 - li - 18.0, d / 2 - li - 13.0), F.v))
    # trees in two or three clumps, benches along the loop, lamps every 30 m
    area = w * d
    n_tree = int(min(area / 150.0, 60))
    clumps = [F.p(float(rng.uniform(-w / 3, w / 3)), float(rng.uniform(-d / 3, d / 3))) for _ in range(int(rng.integers(2, 4)))]
    placed = 0
    for _ in range(n_tree * 3):
      if placed >= n_tree:
        break
      c = clumps[int(rng.integers(len(clumps)))]
      p = c + rng.normal(0.0, 10.0, 2)
      rel = p - F.o
      if abs(rel @ F.ex) > w / 2 - 2.5 or abs(rel @ F.v) > d / 2 - 2.5:
        continue
      if any(sh.cat == "tree" and np.linalg.norm(sh.poly.mean(axis=0) - p) < 9.0 for sh in self.index.query(p - 9.0, p + 9.0)):
        continue
      placed += self._tree(p, conifer=rng.random() < 0.2)
    for t in np.arange(-d / 2 + li + 12.0, d / 2 - li - 6.0, 25.0):
      self._bench(F.p(-w / 2 + li + 3.6, float(t)), F.face - math.pi / 2)
      self._bench(F.p(w / 2 - li - 3.6, float(t)), F.face + math.pi / 2)
    for t in np.arange(-d / 2 + li + 6.0, d / 2 - li, 30.0):
      self._lamp(F.p(-w / 2 + li - 0.6, float(t)), F.face, h=4.0, park=True)
      self._lamp(F.p(w / 2 - li + 0.6, float(t)), F.face, h=4.0, park=True)
    for e in ents[:4]:
      self._bin(e + (F.o - e) / max(np.linalg.norm(F.o - e), 1e-6) * 3.0 + F.ex * 2.0)

  def _copses(self, label: int, spacing: float, thr: float, cap: int = 400, far: float = 0.0) -> int:
    """woodland on what is left of a label: a jittered lattice of trees where 80 m value noise is above thr, the densest
    first up to cap; >= 15 m from any road, >= far from a street"""
    gr = self.grid
    rng = self.r
    mask = gr.free_mask(label, erode=2)
    pts = gr.cells(mask)
    if len(pts) == 0:
      return 0
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    noise = rng.random((int((hi[1] - lo[1]) // 80.0) + 3, int((hi[0] - lo[0]) // 80.0) + 3))
    xs, zs = np.meshgrid(np.arange(lo[0], hi[0], spacing), np.arange(lo[1], hi[1], spacing))
    p = np.stack([xs.ravel(), zs.ravel()], axis=1) + rng.uniform(-spacing * 0.35, spacing * 0.35, (xs.size, 2))
    u, v = (p[:, 0] - lo[0]) / 80.0, (p[:, 1] - lo[1]) / 80.0
    i0, j0 = np.clip(u.astype(int), 0, noise.shape[1] - 2), np.clip(v.astype(int), 0, noise.shape[0] - 2)
    fu, fv = u - i0, v - j0
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)      # smoothstep: round clumps, not diamonds
    val = (noise[j0, i0] * (1 - fu) * (1 - fv) + noise[j0, i0 + 1] * fu * (1 - fv) + noise[j0 + 1, i0] * (1 - fu) * fv +
           noise[j0 + 1, i0 + 1] * fu * fv)
    keep = (val > thr) & gr.mask_at(mask, p)
    p, val = p[keep], val[keep]
    order = np.argsort(-val, kind="stable")
    n = 0
    self._begin("wood")
    for k in order:
      q = p[k]
      if far > 0.0:
        c = self.centre_hash.near(q, q, far)
        if len(c) and float(np.min(np.linalg.norm(c - q, axis=1))) < far:
          continue
      if self._road_d(_rect(q, 0.0, 0.5, 0.5)) < 15.0:
        continue
      n += self._tree(q, conifer=rng.random() < 0.35)
      if n >= cap:
        break
    return n

  def _loop_park(self, label: int) -> None:
    """2.8: the loop infield: an 8 m gravel ring path 18 m in from the loop, a pond with an island, four gated entries,
    a 24-bay lot by J22, picnic tables, a playground, woodland clumps"""
    gr = self.grid
    sd = next((s for s in self.sides if s.label == label and s.kind == "loop"), None)
    if sd is None:
      return
    self._begin("looppark")
    # the lot off the straight by the junction first: the ring path stops at it
    F = sd.frame(CORNER + 22.0)
    self._surface_lot(F, -21.0, 21.0, 1.0, 19.0, 0.5, gap=sd.gap + 1.0, max_bays=24)
    gr.reserve(F.rect(-21.0, 21.0, -sd.gap, 20.0), 1.0)
    ring = sd.front + 18.0
    for s in np.arange(CORNER, sd.len - CORNER, 9.0):
      p0, _ = sd.at(s, ring)
      p1, _ = sd.at(s + 9.0, ring)
      if gr.lab_at(np.array([p0, p1])).min() == label and not gr.occ_at(np.array([p0, p1, (p0 + p1) / 2])).any():
        self._strip(p0, p1 + (p1 - p0) * 0.08, 8.0, L2, GRAVEL, "path", tag="ring")
    for f in (0.15, 0.4, 0.65, 0.88):
      F = sd.frame(sd.len * f)
      self._flat(F, -2.0, 2.0, -sd.gap, 14.5, L2, GRAVEL, "path", tag="entry")
      for a in (-2.6, 2.6):
        self._prop_box("gate", F.p(a, 1.0), F.face, 0.35, 0.6, 0.35, STONE, avoid=("water", "lot", "path"))
    c = self.lcent[label]
    self._pond(c, 20.0, 14.0, 0.0)
    gr.reserve(_rect(c, 0.0, 22.0, 16.0), 1.0)
    self._playground(Frame(c + np.array([0.0, -32.0]), np.array([0.0, 1.0])))
    gr.reserve(_rect(c + np.array([0.0, -32.0]), 0.0, 9.0, 7.0), 1.0)
    for k in range(10):
      ang = k * math.pi / 5 + 0.3
      p = c + np.array([math.cos(ang), math.sin(ang)]) * 34.0
      self._prop_box("picnic", p, ang, 0.9, 0.04, 0.4, FENCE_WOOD, y0=0.72, avoid=("water", "path", "lot"),
                     kids=[(f"seat{i}", "cube", [0, -0.3, z], FENCE_WOOD, {"scale": [0.9, 0.03, 0.15]}) for i, z in enumerate((-0.7, 0.7))])
    for s in np.arange(CORNER + 50.0, sd.len - CORNER, 30.0):
      p, _ = sd.at(s, ring + 5.5)
      self._bench(p, sd.frame(s).face + math.pi)
    self._copses(label, 8.0, 0.5, cap=140)

  def _expressway_belts(self) -> None:
    """three rows of trees each side of the at-grade expressway and its ramps, where the land is free"""
    e = self.express_hash.pts
    if len(e) < 3:
      return
    self.r = self.vrng
    gr = self.grid
    t = np.gradient(e, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    self._begin("belt")
    last = -1e9
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(e, axis=0), axis=1))])
    for i in range(len(e)):
      if s[i] - last < 9.0:
        continue
      last = s[i]
      for sgn in (1.0, -1.0):
        for row, off in enumerate((12.0, 18.0, 24.0)):
          p = e[i] + n[i] * sgn * off + t[i] * (6.0 if row == 1 else 0.0)
          lab = int(gr.lab_at(p[None])[0])
          if lab == 0 or gr.occ_at(p[None])[0] or self.zone.get(lab) == "downtown":
            continue
          if np.isfinite(gr.cap_at(p[None])[0]):
            continue
          self._tree(p, conifer=self.r.random() < 0.5)

  def _outer_interior(self, label: int) -> None:
    gr = self.grid
    F = Frame(self.lcent[label], np.array([0.0, 1.0]))
    fields = 0
    for target in ([-650.0, -200.0], [150.0, 650.0], [-650.0, 700.0], [180.0, -200.0], [-400.0, -230.0], [250.0, 300.0], [-700.0, 300.0]):
      mask = gr.free_mask(label, erode=3)
      site = gr.fit_rect(mask, F, [(120.0, 60.0), (60.0, 120.0)], np.array(target))
      if site is None:
        break
      Fs, w, d = site
      p = self._centre_far(Fs.o)
      if p < 45.0:
        continue
      self._begin("field")
      self._flat(Fs, -w / 2, w / 2, -d / 2, d / 2, L0, CROPS[fields % 3], "field", tag="field")
      gr.reserve(Fs.rect(-w / 2, w / 2, -d / 2, d / 2), 4.0)
      for sgn in (-1.0, 1.0):
        if w > d:
          self._prop_box("hedgerow", Fs.p(0.0, sgn * (d / 2 + 1.5)), Fs.face, w / 2, 0.8, 0.6, HEDGE, avoid=("water",))
        else:
          self._prop_box("hedgerow", Fs.p(sgn * (w / 2 + 1.5), 0.0), Fs.face, 0.6, 0.8, d / 2, HEDGE, avoid=("water",))
      fields += 1
    self._copses(label, 7.0, 0.6, cap=600, far=45.0)

  def _centre_far(self, p: np.ndarray) -> float:
    c = self.centre_hash.near(p, p, 60.0)
    return float(np.min(np.linalg.norm(c - p, axis=1))) if len(c) else math.inf

  # ---- outer frontage programmes ----

  def _commercial_side(self, sd: Side) -> None:
    dp = {"retail": 80.0, "bigbox": 110.0, "apartments": 80.0, "highschool": 250.0}[sd.prog]
    for run in self._runs(sd, dp, 60.0):
      s0, s1, depth, homes = run
      d = float(np.floor(depth.min() / 2.0) * 2.0)
      if s1 - s0 < 90.0:          # too short for the programme: workshops, else a pocket park
        for k in range(max(1, int((s1 - s0) // 40.0))):
          n = max(1, int((s1 - s0) // 40.0))
          a, b = s0 + (s1 - s0) * k / n, s0 + (s1 - s0) * (k + 1) / n
          P = self._parcel(sd, a, b, min(d, 30.0), "works")
          self.grid.reserve(P.quad, 0.0)
          if b - a < 26.0 or not self._works(P):
            self._pocket_park(P)
        continue
      P = self._parcel(sd, s0, s1, d, sd.prog)
      self.grid.reserve(P.quad, 0.0)
      getattr(self, "_" + sd.prog)(P)

  def _retail(self, P: Parcel) -> None:
    """gateway retail: a strip mall behind its lot, a pylon sign, a gas station on the corner"""
    F, rng, W = P.frame, P.rng, P.width
    self.r = rng
    gas = W > 120.0
    mw = min(100.0, W - (40.0 if gas else 4.0))
    ma = -W / 2 + 2.0 + mw / 2
    self._begin("retail", P)
    self._surface_lot(F, ma - mw / 2, ma + mw / 2, 1.0, 37.0, 0.45, gap=P.gap + 1.0)
    self._flat(F, ma - mw / 2, ma + mw / 2, 37.0, 41.0, L2, CONCRETE, "path", tag="walk")
    mid = self._id("mall")
    md = min(20.0, P.depth - 44.0)
    cmds = [self._box(mid, F.p(ma, 41.05 + md / 2), 3.0, mw / 2, 3.0, md / 2, F.face, BRICKS[1]),
            self._kid(mid, "cap", "cube", [0, 3.3, md / 2 - 0.5], ROOFTOP, scale=[mw / 2, 0.3, 0.5])]
    n = max(4, min(6, int(mw // 16)))
    for k in range(n):
      x = -mw / 2 + mw * (k + 0.5) / n
      sc = SHOPS[int(rng.integers(len(SHOPS)))]
      cmds.append(self._kid(mid, f"glass{k}", "cube", [x, -1.4, md / 2 + 0.03], GLASSES[0], scale=[mw / n / 2 - 1.0, 1.3, 0.03]))
      cmds.append(self._kid(mid, f"sign{k}", "cube", [x, 1.8, md / 2 + 0.05], sc, scale=[mw / n / 2 - 2.0, 0.5, 0.04]))
    if not self._add(F.rect(ma - mw / 2, ma + mw / 2, 41.05, 41.05 + md), "bldg", cmds, top=6.6, within=P.quad, door=F.p(ma, 40.7)):
      self._abort(self.g)
      return
    pid = self._id("pylon")
    pp = F.p(ma + mw / 2 - 3.0, 1.5)
    self._add(_rect(pp, F.face, 1.6, 0.3), "prop", [self._box(pid, pp, 4.0, 0.3, 4.0, 0.3, F.face, METAL),
                                                    self._kid(pid, "panel", "cube", [0, 2.5, 0], SHOPS[1], scale=[1.6, 1.2, 0.35])], top=8.0)
    if gas:
      ga = W / 2 - 18.0
      self._begin("gas", P)
      self._flat(F, ga - 16.0, ga + 16.0, -P.gap, 30.0, L2, CONCRETE, "path", tag="forecourt")
      cid = self._id("canopy")
      cc = F.p(ga, 12.0)
      cmds = [self._box(cid, cc, 5.4, 12.0, 0.4, 6.0, F.face, TRIM),
              self._kid(cid, "band", "cube", [0, 0, 6.02], SHOPS[1], scale=[12.0, 0.35, 0.03])]
      for k, (x, z) in enumerate(((-8.0, -3.0), (8.0, -3.0), (-8.0, 3.0), (8.0, 3.0))):
        cmds.append(self._kid(cid, f"post{k}", "cube", [x, -2.7, z], TRIM, scale=[0.2, 2.7, 0.2]))
      for k, x in enumerate((-5.4, -1.8, 1.8, 5.4)):     # pump islands under the canopy
        cmds.append(self._kid(cid, f"pump{k}", "cube", [x, -4.6, 0], SHOPS[1], scale=[0.4, 0.8, 0.9]))
      self._add(_rect(cc, F.face, 12.0, 6.0), "prop", cmds, top=5.8, avoid=("water", "lot"))
      kid = self._id("kiosk")
      self._add(F.rect(ga - 5.0, ga + 5.0, 21.0, 29.0), "bldg",
                [self._box(kid, F.p(ga, 25.0), 2.0, 5.0, 2.0, 4.0, F.face, TRIM),
                 self._kid(kid, "glass", "cube", [0, -0.5, 4.03], GLASSES[0], scale=[4.0, 1.1, 0.03])], top=4.0, door=F.p(ga, 20.7),
                allow=("lawn", "path"))

  def _bigbox(self, P: Parcel) -> None:
    """a power centre: a 90 x 60 big box set back behind a big lot, a fenced garden centre"""
    F, rng, W = P.frame, P.rng, P.width
    self.r = rng
    self._begin("bigbox", P)
    bw = min(90.0, W - 30.0)
    lot_d = min(45.0, P.depth - 66.0)
    self._surface_lot(F, -bw / 2 - 10.0, bw / 2 + 10.0, 1.0, 1.0 + lot_d, 0.4, gap=P.gap + 1.0)
    self._flat(F, -bw / 2, bw / 2, 1.0 + lot_d, 5.0 + lot_d, L2, CONCRETE, "path", tag="walk")
    bd = min(60.0, P.depth - lot_d - 7.0)
    bid = self._id("bigbox")
    d0 = 5.05 + lot_d
    cmds = [self._box(bid, F.p(0.0, d0 + bd / 2), 4.5, bw / 2, 4.5, bd / 2, F.face, _m((0.78, 0.74, 0.66))),
            self._kid(bid, "band", "cube", [0, 3.0, bd / 2 + 0.03], SHOPS[0], scale=[bw / 2, 0.6, 0.03]),
            self._kid(bid, "entry", "cube", [0, -2.5, bd / 2 + 0.03], GLASSES[0], scale=[6.0, 2.0, 0.03]),
            self._kid(bid, "sign", "cube", [0, 1.2, bd / 2 + 0.06], SHOPS[3], scale=[8.0, 1.0, 0.04])]
    if not self._add(F.rect(-bw / 2, bw / 2, d0, d0 + bd), "bldg", cmds, top=9.0, within=P.quad, door=F.p(0.0, d0 - 0.3)):
      self._abort(self.g)
      return
    ga = bw / 2 + 2.0
    if ga + 14.0 < W / 2:
      for p0, p1 in ((F.p(ga, d0), F.p(ga + 12.0, d0)), (F.p(ga + 12.0, d0), F.p(ga + 12.0, d0 + 30.0)), (F.p(ga, d0 + 30.0), F.p(ga + 12.0, d0 + 30.0))):
        self._fence(p0, p1, 2.4, FENCE_IRON)

  def _apartments(self, P: Parcel) -> None:
    """2.3 / garden apartments: walk-ups (40 x 14 x 10.5) round a parking court, a side drive into it, lawns"""
    F, rng, W = P.frame, P.rng, P.width
    self.r = rng
    n = max(1, int((W - 10.0) // 46.0))
    for k in range(n):
      a = -W / 2 + 5.0 + 46.0 * k + 23.0
      self._begin("apartments", P)
      self._flat(F, a - 22.0, a + 22.0, 0.0, 8.0, L0, LAWN, "lawn", tag="lawn")
      self._flat(F, a - 1.0, a + 1.0, -P.gap, 8.0, L2, CONCRETE, "path", tag="walk")
      h = float(rng.choice((10.5, 13.5)))
      bid = self._id("walkup")
      col = BRICKS[k % 2]
      cmds = [self._box(bid, F.p(a, 15.0), h / 2, 20.0, h / 2, 7.0, F.face, col),
              self._kid(bid, "cap", "cube", [0, h / 2 + 0.2, 0], ROOFTOP, scale=[20.0, 0.2, 7.0]),
              self._kid(bid, "canopy", "cube", [0, -h / 2 + 2.8, 7.8], METAL, scale=[2.0, 0.08, 0.8])]
      for f in range(int(h // 3)):
        for sgn in (1.0, -1.0):
          cmds.append(self._kid(bid, f"g{f}{int(sgn > 0)}", "cube", [0, -h / 2 + 1.8 + f * 3.0, sgn * 7.03], GLASSES[0], scale=[18.5, 0.7, 0.03]))
      if not self._add(F.rect(a - 20.0, a + 20.0, 8.0, 22.0), "bldg", cmds, top=h + 0.4, within=P.quad, door=F.p(a, 7.7)):
        self._abort(self.g)
        continue
      for x in (-14.0, 14.0):
        self._planter(F.p(a + x, 4.0))
      if P.depth >= 64.0:
        self._flat(F, a + 20.5, a + 22.95, -P.gap, 26.0, L1, ASPHALT, "path", tag="drive")
        self._surface_lot(Frame(F.p(a, 26.0), F.v), -22.95, 22.95, 0.0, 36.0, 0.35, entry=False, max_bays=24)

  def _highschool(self, P: Parcel) -> None:
    """a high school: 100 x 40 x 9 behind a lawn, its lot beside it, a 400 m track round a field behind"""
    F, rng, W = P.frame, P.rng, P.width
    self.r = rng
    self._begin("highschool", P)
    bw = min(100.0, W - 50.0)
    ba = -W / 2 + 2.0 + bw / 2
    self._flat(F, ba - bw / 2, ba + bw / 2, 0.0, 19.9, L0, LAWN, "lawn", tag="lawn")
    self._flat(F, ba - 2.0, ba + 2.0, -P.gap, 20.0, L2, CONCRETE, "path", tag="walk")
    hid = self._id("highschool")
    cmds = [self._box(hid, F.p(ba, 40.0), 4.5, bw / 2, 4.5, 20.0, F.face, BRICKS[1]),
            self._kid(hid, "cap", "cube", [0, 4.7, 0], ROOFTOP, scale=[bw / 2, 0.2, 20.0]),
            self._kid(hid, "entry", "cube", [0, -2.0, 20.03], GLASSES[1], scale=[8.0, 2.5, 0.03])]
    for f in range(2):
      cmds.append(self._kid(hid, f"g{f}", "cube", [0, -2.4 + 3.6 * f, 20.03], GLASSES[0], scale=[bw / 2 - 2.0, 0.8, 0.03]))
    if not self._add(F.rect(ba - bw / 2, ba + bw / 2, 20.0, 60.0), "bldg", cmds, top=9.2, within=P.quad, door=F.p(ba, 19.7)):
      self._abort(self.g)
      return
    la0 = ba + bw / 2 + 3.0
    if W / 2 - la0 > 25.0:
      self._surface_lot(F, la0, W / 2 - 1.0, 1.0, min(P.depth - 2.0, 74.0), 0.6, gap=P.gap + 1.0)
    # track and field behind: long axis along the depth
    tw, td = min(92.0, W - 10.0), min(176.0, P.depth - 70.0)
    if td > 100.0:
      tc = F.p(ba, 66.0 + td / 2)
      r = tw / 2
      tid = self._id("track")
      cmds = [self._box(tid, tc, L1 - 0.01, r, 0.01, td / 2 - r, F.face, RUST),
              self._kid(tid, "end0", "cylinder", [0, 0.0, td / 2 - r], RUST, radius=1.0, height=0.02, segments=24, scale=[r, 1.0, r]),
              self._kid(tid, "end1", "cylinder", [0, 0.0, -(td / 2 - r)], RUST, radius=1.0, height=0.02, segments=24, scale=[r, 1.0, r]),
              self._kid(tid, "field", "cube", [0, 0.01, 0], PITCH, scale=[r - 9.0, 0.01, td / 2 - r])]
      self._add(_rect(tc, F.face, r, td / 2), "flat", cmds, purpose="court", y=L1, key="track")
      for sgn in (-1.0, 1.0):
        self._prop_box("bleacher", F.p(ba + sgn * (r + 4.0), 66.0 + td / 2), F.face + math.pi / 2, 15.0, 1.5, 2.5, METAL, avoid=("water", "lot"))
        for dd in (-td / 4, td / 4):
          lid = self._id("mast")
          lp = F.p(ba + sgn * (r + 9.0), 66.0 + td / 2 + dd)
          self._add(_rect(lp, 0.0, 0.3, 0.3), "prop", [self._cyl(lid, lp, 9.0, 0.2, 18.0, METAL, seg=8),
                                                      self._kid(lid, "lamps", "cube", [0, 9.2, 0], TRIM, scale=[1.2, 0.6, 0.2])], top=19.0)

  # ---- 7. verges ----

  def _verges(self) -> None:
    self.r = self.vrng
    for sd in self.sides:
      if sd.kind in ("loop", "court") or not sd.kind:
        continue
      self._begin("verge")
      down = sd.zone == "downtown"
      lamp_dx = 30.0 if down else 40.0
      off = 0.0 if sd.n[0] @ np.array([1.0, 0.37]) > 0 else lamp_dx / 2      # staggered between the two sides
      for s in np.arange(CORNER + off, sd.len - CORNER, lamp_dx):
        F = sd.frame(float(s))
        self._lamp(F.p(0.0, -1.2 - 0.0), F.face + math.pi / 2)
      for s in np.arange(CORNER + 20.0 + off, sd.len - CORNER, 90.0):
        F = sd.frame(float(s))
        self._add(_rect(F.p(1.5, -1.2), 0.0, 0.2, 0.2), "prop", [self._cyl(self._id("hydrant"), F.p(1.5, -1.2), 0.4, 0.15, 0.8, SHOPS[1], seg=8)],
                  top=0.8)
      if down:
        for s in np.arange(CORNER + 5.0, sd.len - CORNER, 10.0):
          F = sd.frame(float(s))
          if self._tree(F.p(0.0, -0.9), size=1, avoid=("path", "lot", "water", "court")):
            self._flat(F, -0.6, 0.6, -1.5, -0.3, L3, DARK, "grate", tag="grate")
        mid = sd.len / 2
        F = sd.frame(mid + 6.0)
        sid = self._id("shelter")
        self._add(_rect(F.p(0.0, -0.7), F.face, 1.8, 0.7), "prop",
                  [self._box(sid, F.p(0.0, -0.7), 2.4, 1.8, 0.06, 0.7, F.face, METAL),
                   self._kid(sid, "back", "cube", [0, -1.2, -0.65], _m((0.6, 0.7, 0.75), 0.1, 0.3), scale=[1.8, 1.15, 0.03])], top=2.5,
                  avoid=("lot", "water"))
        for s in np.arange(CORNER + 12.0, sd.len - CORNER, 30.0):
          F = sd.frame(float(s))
          self._bin(F.p(0.0, -0.7))
          self._bench(F.p(3.0, 0.6), F.face + math.pi)
        continue
      # residential: one street tree per lot, a mailbox by each driveway; elsewhere a tree every ~12 m
      covered = []
      for P in sd.lots:
        covered.append((P.s0, P.s1))
        F = P.frame
        if P.drive_a is not None:
          self._add(_rect(F.p(P.drive_a + 2.4, -0.6), 0.0, 0.25, 0.25), "prop",
                    [self._cyl(mb := self._id("mailbox"), F.p(P.drive_a + 2.4, -0.6), 0.5, 0.05, 1.0, METAL, seg=6),
                     self._kid(mb, "box", "cube", [0, 0.6, 0], DARK, scale=[0.12, 0.12, 0.25])], top=1.2)
        for shift in (0.0, 3.0, -3.0):
          if self._tree(F.p(shift, -0.9), size=int(sd.theme["crown"] + 1)):
            break
      spacing = 15.0 if sd.kind == "commercial" else 12.0
      for s in np.arange(CORNER + 6.0, sd.len - CORNER, spacing):
        if any(a - 1.0 <= s <= b + 1.0 for a, b in covered):
          continue
        F = sd.frame(float(s))
        for shift in (0.0, 3.0, -3.0):
          if self._tree(F.p(shift, -0.9), size=1):
            break

  # ---- 8. validation ----

  def _validate(self) -> None:
    removed = 0
    for _ in range(3):
      bad = self._unreachable()
      if not bad:
        break
      for g in bad:
        P = self.groups[g].parcel
        if self.verbose:
          print("  unreachable:", self.groups[g].kind, np.round(self.groups[g].doors[0], 1))
        self._abort(g)
        removed += 1
        if isinstance(P, Parcel):
          self.r = P.rng
          self._pocket_park(P)
    self.removed = removed
    # exact rules on every solid that is left
    viol = defaultdict(int)
    solids = [sh for gr in self.groups if gr.alive for sh in gr.shapes if sh.cat != "flat"]
    for sh in solids:
      why = self._hard(sh.poly, sh.cat, sh.top, sh.crown)
      if why:
        viol[why] += 1
    overlaps = 0
    for sh in solids:
      for o in self.index.query(sh.lo, sh.hi):
        if o is not sh and o.cat != "flat" and id(o) > id(sh) and not _sep(sh.poly, o.poly):
          overlaps += 1
    self.violations, self.overlaps = dict(viol), overlaps

  def _unreachable(self) -> list[int]:
    flats = [sh for gr in self.groups if gr.alive for sh in gr.shapes if sh.cat == "flat" and sh.purpose in WALK]
    seen = set()
    q = deque()
    for sh in flats:
      c = self.centre_hash.near(sh.lo, sh.hi, self.foot + 1.0)
      if len(c) and float(_dist(c, sh.poly).min()) <= self.foot + 0.8:
        seen.add(id(sh))
        q.append(sh)
    while q:
      sh = q.popleft()
      for o in self.index.query(sh.lo - 0.25, sh.hi + 0.25):
        if o.cat == "flat" and o.purpose in WALK and id(o) not in seen and not _sep(sh.poly, o.poly, 0.2):
          seen.add(id(o))
          q.append(o)
    bad = []
    for g, gr in enumerate(self.groups):
      if not gr.alive or not gr.doors:
        continue
      for door in gr.doors:
        ok = any(o.cat == "flat" and id(o) in seen and float(_dist(door[None], o.poly)[0]) <= 0.5
                 for o in self.index.query(door - 0.6, door + 0.6))
        if not ok:
          bad.append(g)
          break
    return bad
