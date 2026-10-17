import collections
import json
import os

import numpy as np

from openpilot.tools.sim.bridge.zoomengine.track import Track, catmull_rom

# German Autobahn markings: 0.15 m lines (0.3 m on the right edge), 6 m dashes with 12 m gaps
LINE_W = 0.15
EDGE_W = 0.3
DASH = [6.0, 12.0]
TREE_RADIUS = 260.0   # only forest instances this close to the Autobahn are planted
MAX_TREES = 3500


def chain_loop(segments) -> np.ndarray:
  """Ordered closed centerline through BeamNG AI road segments: each segment's spline is appended where the previous
  one ends, dropping points that overlap the end so the joins don't fold back"""
  segs = [catmull_rom(s) for s in segments]
  path = [segs.pop(0)]
  while segs:
    end = path[-1][-1]
    j = min(range(len(segs)), key=lambda i: np.linalg.norm(segs[i][0] - end))
    nxt = segs.pop(j)
    fwd = path[-1][-1] - path[-1][-5]
    fwd[1] = 0.0
    fwd /= np.linalg.norm(fwd) + 1e-9
    ahead = (nxt - end) @ fwd > 1.0
    first = int(np.argmax(ahead)) if ahead.any() else len(nxt)
    path.append(nxt[first:])
  pts = np.vstack(path)
  # drop the tail that overlaps the start
  fwd = pts[5] - pts[0]
  fwd[1] = 0.0
  fwd /= np.linalg.norm(fwd) + 1e-9
  while len(pts) > 10 and (pts[-1] - pts[0]) @ fwd > -1.0:
    pts = pts[:-1]
  return pts


class RoadSurface:
  """Height of the level's road meshes (Road_* materials) under a point, by point-in-triangle tests on a grid"""
  CELL = 40.0

  def __init__(self, map_dir: str, statics: list[dict]):
    tris = []
    for path in sorted({st["path"] for st in statics if not st.get("dynamic") and st["position"] == [0, 0, 0]}):
      tris += self._load(os.path.join(map_dir, path))  # world-placed meshes only (instanced props sit at offsets)
    self.T = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
    self.grid = collections.defaultdict(list)
    lo = np.floor(self.T[:, :, [0, 2]].min(1) / self.CELL).astype(int)
    hi = np.floor(self.T[:, :, [0, 2]].max(1) / self.CELL).astype(int)
    for k in range(len(self.T)):
      for cx in range(lo[k, 0], hi[k, 0] + 1):
        for cz in range(lo[k, 1], hi[k, 1] + 1):
          self.grid[(cx, cz)].append(k)
    self.grid = {key: np.array(v) for key, v in self.grid.items()}

  @staticmethod
  def _load(gltf_path: str) -> list[np.ndarray]:
    with open(gltf_path) as f:
      g = json.load(f)
    base = os.path.dirname(gltf_path)
    bins: dict[str, bytes] = {}

    def acc(i, n, dtype):
      a = g["accessors"][i]
      bv = g["bufferViews"][a["bufferView"]]
      uri = g["buffers"][bv["buffer"]]["uri"]
      if uri not in bins:
        with open(os.path.join(base, uri), "rb") as fb:
          bins[uri] = fb.read()
      return np.frombuffer(bins[uri], dtype=dtype, count=a["count"] * n, offset=bv.get("byteOffset", 0) + a.get("byteOffset", 0))

    # root node carries the Z-up -> Y-up matrix, apply it like the engine does
    m = np.array(g["nodes"][g["scenes"][0]["nodes"][0]].get("matrix", np.eye(4).flatten()), dtype=float).reshape(4, 4).T
    out = []
    for mesh in g["meshes"]:
      for prim in mesh["primitives"]:
        if not g["materials"][prim["material"]].get("name", "").startswith("Road_"):
          continue
        v = acc(prim["attributes"]["POSITION"], 3, np.float32).reshape(-1, 3).astype(float)
        v = v @ m[:3, :3].T + m[:3, 3]
        ia = g["accessors"][prim["indices"]]
        idx = acc(prim["indices"], 1, {5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[ia["componentType"]]).reshape(-1, 3)
        out.append(v[idx])
    return out

  def height(self, x: float, z: float) -> float | None:
    ks = self.grid.get((int(np.floor(x / self.CELL)), int(np.floor(z / self.CELL))))
    if ks is None:
      return None
    t = self.T[ks]
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    v0, v1 = b[:, [0, 2]] - a[:, [0, 2]], c[:, [0, 2]] - a[:, [0, 2]]
    v2 = np.array([x, z]) - a[:, [0, 2]]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
      u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
      w = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    inside = (u >= -1e-4) & (w >= -1e-4) & (u + w <= 1 + 1e-4) & (np.abs(den) > 1e-9)
    if not inside.any():
      return None
    y = a[inside, 1] + u[inside] * (b[inside, 1] - a[inside, 1]) + w[inside] * (c[inside, 1] - a[inside, 1])
    return float(y.max())


def drape(track: Track, surface: RoadSurface, lanes) -> Track:
  """track with its heights taken from the road surface under its lanes (the AI path heights are only approximate)"""
  y = track.y.copy()
  for i in range(track.n):
    s = i * track.ds
    hs = [surface.height(*np.array(track.pose(s, d))[[0, 1]]) for d in lanes]
    hs = [h for h in hs if h is not None]
    if hs:
      y[i] = sum(hs) / len(hs)
  y[-1] = y[0]
  return Track.from_points(np.stack([track.x, y, track.z], axis=1), step=track.ds)


def smooth_loop(track: Track, window_m: float = 30.0) -> Track:
  n = max(3, int(window_m / track.ds) | 1)
  k = np.ones(n) / n
  pts = []
  for a in (track.x, track.y, track.z):
    a = a[:-1]
    pts.append(np.convolve(np.concatenate([a[-n:], a, a[:n]]), k, mode="same")[n:-n])
  return Track.from_points(np.stack(pts, axis=1))


class BeamngMap:
  """A BeamNG level converted by ZoomEngine's tools/beamng_import.py: its meshes, its two Autobahn carriageways
  as tracks (from the level's AI roads), generated lane markings and trees at the level's forest positions"""
  def __init__(self, level_json: str, ours: str = "A", theirs: str = "b"):
    self.dir = os.path.dirname(os.path.abspath(level_json))
    with open(level_json) as f:
      self.level = json.load(f)
    roads = self.level["roads"]
    self.track = smooth_loop(Track.from_points(chain_loop(roads[ours])))
    self.opposing_track = smooth_loop(Track.from_points(chain_loop(roads[theirs]))) if theirs in roads else None
    widths = [n[3] for seg in roads[ours] for n in seg]
    self.road_width = float(np.median(widths))
    lanes = 2
    lw = self.road_width / lanes
    self.lane_offsets = tuple(-self.road_width / 2 + lw * (i + 0.5) for i in range(lanes))
    # heights from the actual road meshes, so markings sit on the asphalt and cars don't sink into it
    self.surface = RoadSurface(self.dir, self.level["statics"])
    self.track = drape(self.track, self.surface, self.lane_offsets)
    if self.opposing_track is not None:
      self.opposing_track = drape(self.opposing_track, self.surface, self.lane_offsets)
    self.name = self.level.get("level", "beamng")

  def scene_commands(self) -> list[dict]:
    cmds = []
    for i, st in enumerate(self.level["statics"]):
      if st.get("dynamic"):
        continue
      cmds.append({"action": "create", "type": "gltf", "id": f"bng{i}", "path": os.path.join(self.dir, st["path"]),
                   "position": st["position"], "rotation": st["rotation"], "scale": st["scale"]})
    cmds += self._markings(self.track, "a")
    if self.opposing_track is not None:
      cmds += self._markings(self.opposing_track, "b")
    cmds += self._trees()
    return cmds

  def _markings(self, track: Track, tag: str) -> list[dict]:
    half = self.road_width / 2
    cmds = []

    def line(lid, d, width, dash=None):
      pts = []
      for s in np.arange(0.0, track.length + 1.5, 3.0):
        x, z, _ = track.pose(float(s), d)
        h = self.surface.height(x, z)
        pts.append([round(x, 3), round((track.height(float(s)) if h is None else h) + 0.04, 3), round(z, 3)])
      cmd = {"action": "create", "type": "ribbon", "id": f"mark_{tag}_{lid}", "points": pts,
             "width": width, "color": [0.92, 0.92, 0.9], "roughness": 0.6}
      if dash:
        cmd["dash"] = dash
      cmds.append(cmd)

    line("left", -half + 0.25, LINE_W)
    line("right", half - 0.25, EDGE_W)
    for i in range(1, len(self.lane_offsets)):
      line(f"lane{i}", (self.lane_offsets[i - 1] + self.lane_offsets[i]) / 2, LINE_W, DASH)
    return cmds

  def _trees(self) -> list[dict]:
    trees = self.level.get("trees", [])
    if not trees:
      return []
    pos = np.array([t[:3] for t in trees], dtype=float)
    road = np.concatenate([np.stack([t.x, t.z], 1)[::10] for t in (self.track, self.opposing_track) if t is not None])
    near = np.zeros(len(pos), dtype=bool)
    for chunk in range(0, len(road), 512):
      d2 = ((pos[:, None, [0, 2]] - road[None, chunk:chunk + 512]) ** 2).sum(-1).min(1)
      near |= d2 < TREE_RADIUS ** 2
    # keep trees off the carriageways themselves
    clear = np.ones(len(pos), dtype=bool)
    for chunk in range(0, len(road), 512):
      d2 = ((pos[:, None, [0, 2]] - road[None, chunk:chunk + 512]) ** 2).sum(-1).min(1)
      clear &= d2 > (self.road_width + 8.0) ** 2
    idx = np.nonzero(near & clear)[0]
    if len(idx) > MAX_TREES:
      idx = np.random.default_rng(0).choice(idx, MAX_TREES, replace=False)
      print(f"[zoomengine] planting {MAX_TREES} of {int((near & clear).sum())} trees near the Autobahn")
    cmds = []
    for i in idx:
      x, y, z, scale, kind = trees[i]
      h = 14.0 * scale
      conifer = any(k in kind for k in ("fir", "pine", "spruce"))
      green = [0.09, 0.2 + 0.08 * ((i * 7) % 5) / 4, 0.08]
      cmds.append({"action": "create", "type": "cylinder", "id": f"trunk{i}", "position": [x, y + h * 0.2, z], "radius": 0.3,
                   "height": h * 0.4, "segments": 6, "color": [0.28, 0.2, 0.13], "roughness": 0.95})
      if conifer:
        cmds.append({"action": "create", "type": "cone", "id": f"crown{i}", "position": [x, y + h * 0.55, z], "radius": h * 0.22,
                     "height": h * 0.85, "segments": 8, "color": green, "roughness": 0.95})
      else:
        r = h * 0.3
        cmds.append({"action": "create", "type": "sphere", "id": f"crown{i}", "position": [x, y + h * 0.62, z],
                     "scale": [r, r * 0.85, r], "segments": 10, "color": green, "roughness": 0.95})
    return cmds

  def spawn_s(self) -> float:
    return 100.0

