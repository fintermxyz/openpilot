import math
import numpy as np

from openpilot.tools.sim.bridge.zoomengine.scene import EDGE, car_commands, deg
from openpilot.tools.sim.bridge.zoomengine.track import Track, right_of

# A suburb on the outside of the first straight (s 0..1200, right of our carriageway), laid out in track (s, d):
# an exit ramp leaves the right lane, becomes a two-way collector road parallel to the highway, which feeds a
# street grid of houses; an on-ramp at the far end merges back. Everything is static scenery (no NPC traffic).

RAMP_LANE_D = 7.6                       # exit / acceleration lane center: its left edge is the highway edge line
COLLECTOR_D = 48.0                      # collector road center
COLLECTOR_HALF = 4.0
STREET_HALF = 3.75                      # residential streets: two 3.5 m lanes + gutters
BACK_STREETS_D = (198.0, 348.0)
CROSS_STREETS_S = (500.0, 633.0, 767.0, 900.0)
EXIT_S = (200.0, 260.0, 460.0)          # deceleration lane start, ramp diverges, joins the collector
ENTRY_S = (940.0, 1140.0, 1200.0)       # leaves the collector, joins the acceleration lane, lane ends
RAMP_HALF = 2.05
VERGE = 1.5                             # grass strip between curb and sidewalk
SIDEWALK = 1.8

ZONE_S = (180.0, 1220.0)                # keep the highway's random trees / buildings out of here
ZONE_D = (10.0, 430.0)

ASPHALT = [0.13, 0.13, 0.135]
CONCRETE = [0.62, 0.61, 0.58]
WALLS = [[0.92, 0.89, 0.8], [0.78, 0.84, 0.88], [0.76, 0.82, 0.72], [0.95, 0.95, 0.93], [0.86, 0.78, 0.66],
         [0.93, 0.88, 0.62], [0.7, 0.66, 0.62], [0.88, 0.8, 0.78]]
ROOFS = [[0.22, 0.22, 0.24], [0.35, 0.24, 0.18], [0.45, 0.2, 0.16], [0.3, 0.32, 0.34], [0.18, 0.2, 0.26]]
DOORS = [[0.45, 0.08, 0.08], [0.1, 0.18, 0.35], [0.3, 0.2, 0.12], [0.12, 0.25, 0.15], [0.95, 0.95, 0.93]]


def _smooth(t: float) -> float:
  t = min(1.0, max(0.0, t))
  return t * t * (3.0 - 2.0 * t)


def ramp_centerlines() -> dict[str, list[tuple[float, float]]]:
  """(s, d) centerlines of the exit and entry ramps, including their parallel lanes beside the highway"""
  a, b, c = EXIT_S
  end_d = COLLECTOR_D + 0.5 * COLLECTOR_HALF  # the collector's +s lane (driving on the right)
  exit_pts = [(s, RAMP_LANE_D) for s in np.arange(a, b, 10.0)]
  exit_pts += [(s, RAMP_LANE_D + (end_d - RAMP_LANE_D) * _smooth((s - b) / (c - b))) for s in np.arange(b, c + 1e-6, 4.0)]
  a, b, c = ENTRY_S
  entry_pts = [(s, end_d + (RAMP_LANE_D - end_d) * _smooth((s - a) / (b - a))) for s in np.arange(a, b, 4.0)]
  entry_pts += [(s, RAMP_LANE_D) for s in np.arange(b, c + 1e-6, 10.0)]
  return {"exit": exit_pts, "entry": entry_pts}


_RAMPS = ramp_centerlines()


def in_zone(s: float, d: float) -> bool:
  return ZONE_S[0] <= s <= ZONE_S[1] and ZONE_D[0] <= d <= ZONE_D[1]


def near_ramp(s: float, d: float, clearance: float = 3.0) -> bool:
  return any(abs(s - rs) < 6.0 and abs(d - rd) < RAMP_HALF + clearance for pts in _RAMPS.values() for rs, rd in pts)


def keep_clear(s: float, d: float) -> bool:
  """True where the highway's own roadside props would land in the suburb or on a ramp"""
  return in_zone(s, d) or near_ramp(s, d)


def suburb_commands(track: Track, rng: np.random.Generator) -> list[dict]:
  cmds: list[dict] = []

  def world(sd, y):
    return [[round(x, 3), y, round(z, 3)] for x, z, _ in (track.pose(s, d) for s, d in sd)]

  def ribbon(rid, sd, width, color, lift, **kw):
    cmds.append({"action": "create", "type": "ribbon", "id": rid, "points": world(sd, 0.0), "width": width, "color": color,
                 "lift": lift, "roughness": kw.pop("roughness", 0.85), **kw})

  def line(rid, sd, offset, color=(0.9, 0.9, 0.9), dash=None, width=0.14):
    kw = {"offset": offset, "roughness": 0.6}
    if dash:
      kw["dash"] = dash
    ribbon(rid, sd, width, list(color), 0.045, **kw)

  def along_s(d, s0, s1, step=10.0):
    return [(s, d) for s in np.arange(s0, s1 + 1e-6, step)]

  def along_d(s, d0, d1, step=10.0):
    return [(s, d) for d in np.arange(d0, d1 + 1e-6, step)]

  # ---- ramps ----
  for name, pts in ramp_centerlines().items():
    ribbon(f"sub_{name}", pts, 2 * RAMP_HALF, ASPHALT, 0.03)
    line(f"sub_{name}_edge", pts, RAMP_HALF - 0.25)
    # left boundary: dashed where the lane runs beside the highway, solid once it has split off
    beside = [p for p in pts if p[1] <= RAMP_LANE_D + 0.05]
    split = [p for p in pts if p[1] > RAMP_LANE_D + 0.05]
    if len(beside) > 1:
      line(f"sub_{name}_lane", beside, -(RAMP_LANE_D - EDGE), dash=[3.0, 9.0])
    if len(split) > 1:
      line(f"sub_{name}_gore", split, -(RAMP_HALF - 0.25))

  # ---- streets ----
  s_lo, s_hi = EXIT_S[2], ENTRY_S[0]
  collector = along_s(COLLECTOR_D, s_lo, s_hi)
  ribbon("sub_collector", collector, 2 * COLLECTOR_HALF, ASPHALT, 0.03)
  for k, off in enumerate((-0.12, 0.12)):
    line(f"sub_collector_yellow{k}", collector, off, color=(0.85, 0.65, 0.1), width=0.11)
  for k, off in enumerate((-(COLLECTOR_HALF - 0.3), COLLECTOR_HALF - 0.3)):
    line(f"sub_collector_edge{k}", collector, off, width=0.12)

  g_lo, g_hi = CROSS_STREETS_S[0], CROSS_STREETS_S[-1]
  for i, d in enumerate(BACK_STREETS_D):
    ribbon(f"sub_back{i}", along_s(d, g_lo - STREET_HALF, g_hi + STREET_HALF), 2 * STREET_HALF, ASPHALT, 0.031)
  for i, s in enumerate(CROSS_STREETS_S):
    ribbon(f"sub_cross{i}", along_d(s, COLLECTOR_D, BACK_STREETS_D[-1] + STREET_HALF), 2 * STREET_HALF, ASPHALT, 0.032)

  # ---- sidewalks: one strip per street side, broken at intersections ----
  walk_off = VERGE + SIDEWALK / 2
  n_walk = 0

  def sidewalk(sd):
    nonlocal n_walk
    if len(sd) > 1:
      ribbon(f"sub_walk{n_walk}", sd, SIDEWALK, CONCRETE, 0.06, roughness=0.9)
      n_walk += 1

  cut = STREET_HALF + VERGE + SIDEWALK
  walk_ds = [(COLLECTOR_D, COLLECTOR_HALF, (1,), s_lo, s_hi)] + [(d, STREET_HALF, (-1, 1), g_lo, g_hi) for d in BACK_STREETS_D]
  for d, half, sides, a, b in walk_ds:
    stops = [a] + [s for s in CROSS_STREETS_S if a < s < b] + [b]
    if a == g_lo:
      stops = list(CROSS_STREETS_S)
    for side in sides:
      if d == BACK_STREETS_D[-1] and side > 0:
        sidewalk(along_s(d + side * (half + walk_off), g_lo - cut, g_hi + cut))
        continue
      for p, q in zip(stops[:-1], stops[1:], strict=True):
        p0 = p + (cut if p in CROSS_STREETS_S else 0.0)
        q0 = q - (cut if q in CROSS_STREETS_S else 0.0)
        sidewalk(along_s(d + side * (half + walk_off), p0, q0, step=q0 - p0))
  street_ds = [COLLECTOR_D + COLLECTOR_HALF - STREET_HALF] + list(BACK_STREETS_D)
  for s in CROSS_STREETS_S:
    for side in (-1, 1):
      for p, q in zip(street_ds[:-1], street_ds[1:], strict=True):
        sidewalk(along_d(s + side * (STREET_HALF + walk_off), p + cut, q - cut, step=q - p - 2 * cut))

  # ---- street lights and stop signs ----
  n_light = 0
  for d, half, a, b in [(COLLECTOR_D, COLLECTOR_HALF, s_lo, s_hi)] + [(d, STREET_HALF, g_lo, g_hi) for d in BACK_STREETS_D]:
    for s in np.arange(a + 20.0, b, 45.0):
      if any(abs(s - c) < 10.0 for c in CROSS_STREETS_S):
        continue
      x, z, psi = track.pose(s, d + half + 0.6)
      lx, lz, _ = track.pose(s, d + half - 0.6)
      cmds.append({"action": "create", "type": "cylinder", "id": f"sub_pole{n_light}", "position": [x, 3.0, z], "radius": 0.08,
                   "height": 6.0, "segments": 8, "color": [0.32, 0.33, 0.34], "metallic": 0.7, "roughness": 0.5})
      cmds.append({"action": "create", "type": "cube", "id": f"sub_lamp{n_light}", "position": [lx, 6.0, lz],
                   "rotation": [0, deg(psi + math.pi / 2), 0], "scale": [0.12, 0.06, 0.7], "color": [0.32, 0.33, 0.34],
                   "metallic": 0.7, "roughness": 0.5})
      n_light += 1

  n_sign = 0

  def stop_sign(s, d, toward_plus_d):
    # faces traffic coming from the other side: approaching along -d it faces +d, and vice versa
    nonlocal n_sign
    x, z, psi = track.pose(s, d)
    rx, rz = right_of(psi)
    sgn = 1.0 if toward_plus_d else -1.0
    yaw = math.atan2(sgn * rx, sgn * rz)
    sid = f"sub_stop{n_sign}"
    cmds.append({"action": "create", "type": "cylinder", "id": sid, "position": [x, 1.2, z], "rotation": [0, deg(yaw), 0],
                 "radius": 0.04, "height": 2.4, "segments": 6, "color": [0.6, 0.6, 0.62], "metallic": 0.8})
    cmds.append({"action": "create", "type": "cylinder", "id": f"{sid}_face", "parent": sid, "position": [0, 1.0, 0.06],
                 "rotation": [90, 22.5, 0], "radius": 0.38, "height": 0.03, "segments": 8, "color": [0.75, 0.04, 0.04],
                 "emissive": [0.25, 0.0, 0.0], "roughness": 0.5})
    n_sign += 1

  sign_s = STREET_HALF + 1.0
  for s in CROSS_STREETS_S:
    stop_sign(s + sign_s, COLLECTOR_D + COLLECTOR_HALF + 2.5, True)                   # T into the collector
    stop_sign(s + sign_s, BACK_STREETS_D[0] + STREET_HALF + 2.5, True)                # four-way: both approaches
    stop_sign(s - sign_s, BACK_STREETS_D[0] - STREET_HALF - 2.5, False)
    stop_sign(s - sign_s, BACK_STREETS_D[1] - STREET_HALF - 2.5, False)               # T into the last street

  # ---- houses ----
  rows = [(COLLECTOR_D, COLLECTOR_HALF, 1, s_lo + 10.0, s_hi - 10.0)]
  for d in BACK_STREETS_D:
    rows += [(d, STREET_HALF, -1, g_lo, g_hi), (d, STREET_HALF, 1, g_lo, g_hi)]
  n_house = n_tree = n_parked = 0
  for d_street, half, side, a, b in rows:
    stops = sorted({a, b, *[c for c in CROSS_STREETS_S if a <= c <= b]})
    for p, q in zip(stops[:-1], stops[1:], strict=True):
      lo, hi = p + (STREET_HALF + 12.0 if p in CROSS_STREETS_S else 0.0), q - (STREET_HALF + 12.0 if q in CROSS_STREETS_S else 0.0)
      n = int((hi - lo) // 24.0)
      for k in range(n):
        s = lo + (hi - lo) * (k + 0.5) / n + rng.uniform(-1.5, 1.5)
        made = _house(cmds, track, rng, f"sub_house{n_house}", s, d_street, half, side, n_tree, n_parked)
        n_house += 1
        n_tree, n_parked = made
  return cmds


def _house(cmds, track, rng, hid, s, d_street, half, side, n_tree, n_parked):
  """One lot facing the street at d_street from `side` (+1: the lot is at larger d). Returns updated tree / car counters"""
  w, dep = rng.uniform(10.0, 14.0), rng.uniform(8.0, 10.5)
  h = rng.choice([3.2, 3.2, 5.8])
  setback = rng.uniform(7.0, 11.0)
  d = d_street + side * (half + VERGE + SIDEWALK + setback + dep / 2)
  x, z, psi = track.pose(s, d)
  rx, rz = right_of(psi)
  yaw = math.atan2(-side * rx, -side * rz)   # front (+z local) faces the street
  cy, sy = math.cos(yaw), math.sin(yaw)

  def to_world(lx, lz):
    # object local +x is left: local (lx, lz) -> world (x, z)
    return x + lx * cy + lz * sy, z - lx * sy + lz * cy

  wall = WALLS[rng.integers(len(WALLS))]
  cmds.append({"action": "create", "type": "cube", "id": hid, "position": [round(x, 3), h / 2, round(z, 3)], "rotation": [0, deg(yaw), 0],
               "scale": [w / 2, h / 2, dep / 2], "color": wall, "roughness": 0.85})

  def child(name, pos, scale, color, typ="cube", **kw):
    cmds.append({"action": "create", "type": typ, "id": f"{hid}_{name}", "parent": hid, "position": pos, "scale": scale,
                 "color": color, **kw})

  # gable roof: a square prism turned 45 degrees about the width axis, its lower half hidden in the walls. It overhangs
  # the eaves but stops just inside the gable walls, or its lower half would show past them as a tilted cube
  r = (dep / 2 + 0.45) / math.sqrt(2.0)
  child("roof", [0, h / 2, 0], [w / 2 - 0.03, r, r], ROOFS[rng.integers(len(ROOFS))], rotation=[45, 0, 0], roughness=0.8)
  if rng.random() < 0.4:
    child("chimney", [rng.choice([-1, 1]) * w * 0.3, h / 2 + dep * 0.3, -dep * 0.18], [0.4, 1.1, 0.4], [0.5, 0.3, 0.25], roughness=0.9)

  front = dep / 2 + 0.02
  gside = rng.choice([-1, 1])
  door_x = -gside * rng.uniform(0.5, w / 2 - 2.5)
  child("door", [door_x, -h / 2 + 1.05, front], [0.5, 1.05, 0.03], DOORS[rng.integers(len(DOORS))], roughness=0.6)
  glass = {"metallic": 0.6, "roughness": 0.12}
  floors = [-h / 2 + 1.6] + ([-h / 2 + 4.3] if h > 4 else [])
  for f, wy in enumerate(floors):
    for k, wx in enumerate((-w * 0.3, w * 0.3)):
      if f == 0 and abs(wx - door_x) < 1.6:
        continue
      child(f"win{f}{k}", [wx, wy, front], [0.75, 0.6, 0.03], [0.12, 0.15, 0.2], **glass)

  # attached garage + driveway to the street
  gx = gside * (w / 2 + 3.2)
  child("garage", [gx, -h / 2 + 1.5, dep * 0.05], [3.2, 1.5, dep * 0.45], wall, roughness=0.85)
  child("garage_roof", [gx, -h / 2 + 3.05, dep * 0.05], [3.4, 0.08, dep * 0.45 + 0.3], [0.3, 0.3, 0.32], roughness=0.8)
  gfront = dep * 0.5 + 0.02
  child("garage_door", [gx, -h / 2 + 1.15, gfront], [2.6, 1.1, 0.03], [0.93, 0.93, 0.9], roughness=0.5)
  drive_len = setback + VERGE + SIDEWALK + 0.4
  child("driveway", [gx, -h / 2 + 0.045, gfront + drive_len / 2], [2.8, 0.025, drive_len / 2], [0.58, 0.57, 0.54], roughness=0.95)
  if rng.random() < 0.35:
    px, pz = to_world(gx, gfront + 3.2)
    kind = "suv" if rng.random() < 0.4 else "sedan"
    cmds.extend(car_commands(f"sub_parked{n_parked}", kind, [float(c) for c in rng.choice(_CAR_COLORS)], round(px, 3), round(pz, 3),
                             yaw + (math.pi if rng.random() < 0.5 else 0.0)))
    n_parked += 1

  # yard trees: one or two out back, sometimes one by the street
  spots = [(rng.uniform(-w / 2, w / 2), -dep / 2 - rng.uniform(4.0, 11.0)) for _ in range(rng.integers(1, 3))]
  if rng.random() < 0.45:
    spots.append((-gside * rng.uniform(2.0, w / 2), dep / 2 + setback * rng.uniform(0.35, 0.7)))
  for lx, lz in spots:
    tx, tz = to_world(lx, lz)
    _tree(cmds, rng, f"sub_tree{n_tree}", tx, tz, rng.uniform(4.5, 9.0))
    n_tree += 1
  return n_tree, n_parked


_CAR_COLORS = [[0.75, 0.75, 0.78], [0.08, 0.08, 0.09], [0.55, 0.05, 0.05], [0.9, 0.9, 0.92], [0.1, 0.2, 0.45], [0.35, 0.36, 0.38]]


def _tree(cmds, rng, tid, x, z, h):
  green = [rng.uniform(0.08, 0.16), rng.uniform(0.24, 0.4), rng.uniform(0.06, 0.12)]
  cmds.append({"action": "create", "type": "cylinder", "id": f"{tid}_trunk", "position": [round(x, 3), h * 0.3, round(z, 3)], "radius": 0.2,
               "height": h * 0.6, "segments": 8, "color": [0.3, 0.2, 0.12], "roughness": 0.95})
  r = h * 0.32
  cmds.append({"action": "create", "type": "sphere", "id": f"{tid}_crown", "position": [round(x, 3), h * 0.66, round(z, 3)],
               "scale": [r, r * 0.9, r], "segments": 12, "color": green, "roughness": 0.95})
