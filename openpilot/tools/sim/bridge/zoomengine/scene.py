import math
import numpy as np

from openpilot.tools.sim.bridge.zoomengine.track import LANE_WIDTH, Track

EDGE = LANE_WIDTH * 1.5             # lane edge line offset from carriageway center
OPPOSING_D = -(2 * EDGE + 2 * 1.0 + 6.0)  # opposing carriageway center: shoulders + grass median
CAR_Y = 0.68                         # body center height above the road

CAR_COLORS = [[0.75, 0.75, 0.78], [0.08, 0.08, 0.09], [0.55, 0.05, 0.05], [0.9, 0.9, 0.92], [0.1, 0.2, 0.45],
              [0.35, 0.36, 0.38], [0.5, 0.45, 0.35], [0.1, 0.3, 0.2], [0.85, 0.65, 0.1]]

KINDS = {
  # name: (half width, half height, half length, body center y, cabin (half w, half h, half l, y, z), wheel radius)
  "sedan": (0.92, 0.38, 2.3, 0.68, (0.80, 0.30, 1.2, 0.62, -0.25), 0.34),
  "suv":   (0.97, 0.52, 2.4, 0.86, (0.90, 0.36, 1.5, 0.82, -0.35), 0.40),
  "truck": (1.25, 1.55, 4.0, 1.95, (1.20, 0.80, 0.9, -0.55, 4.9), 0.50),
}


def deg(psi: float) -> float:
  return round(math.degrees(psi), 3)


def car_rotation(psi: float, pitch: float = 0.0) -> list[float]:
  """Euler XYZ degrees (engine order Rx*Ry*Rz) of a car heading psi with its nose pitched up by `pitch` radians"""
  if pitch == 0.0:
    return [0, deg(psi), 0]
  cy, sy = math.cos(psi), math.sin(psi)
  cp, sp = math.cos(-pitch), math.sin(-pitch)
  R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
  a = math.atan2(-R[1, 2], R[2, 2])
  b = math.asin(max(-1.0, min(1.0, R[0, 2])))
  c = math.atan2(-R[0, 1], R[0, 0])
  return [round(math.degrees(a), 3), round(math.degrees(b), 3), round(math.degrees(c), 3)]


def car_commands(cid: str, kind: str, color, x: float, z: float, psi: float, sensor_hidden: bool = False,
                 y0: float | None = None, rotation: list[float] | None = None) -> list[dict]:
  """a car whose body center is at (x, y0, z), y0 defaults to the body height above flat ground at y=0"""
  hw, hh, hl, y, (cw, ch, cl, cy, cz), wr = KINDS[kind]
  cmds = [{"action": "create", "type": "cube", "id": cid, "position": [x, y if y0 is None else y0, z],
           "rotation": rotation or [0, deg(psi), 0],
           "scale": [hw, hh, hl], "color": color, "metallic": 0.6, "roughness": 0.3, "sensor_hidden": sensor_hidden}]

  def child(name, typ, pos, scale, col, **kw):
    cmds.append({"action": "create", "type": typ, "id": f"{cid}_{name}", "parent": cid, "position": pos, "scale": scale,
                 "color": col, "sensor_hidden": sensor_hidden, **kw})

  if kind == "truck":
    # boxy trailer is the body, cab sits in front of it
    child("cab", "cube", [0, cy, cz], [cw, ch * 1.6, cl], color, metallic=0.5, roughness=0.35)
    child("glass", "cube", [0, cy + ch * 0.9, cz + cl + 0.01], [cw * 0.85, ch * 0.5, 0.01], [0.05, 0.06, 0.08], metallic=0.9, roughness=0.05)
    trailer_col = [0.92, 0.92, 0.9]
    cmds[0]["color"] = trailer_col
    axles = [(-hl + 1.0), (-hl + 2.3), cz]
  else:
    child("cabin", "cube", [0, cy, cz], [cw, ch, cl], [0.07, 0.08, 0.1], metallic=0.9, roughness=0.08)
    axles = [-hl * 0.62, hl * 0.62]

  wy = wr - y
  for i, az in enumerate(axles):
    for side in (-1, 1):
      child(f"wheel{i}{'lr'[side > 0]}", "cylinder", [side * (hw - 0.08), wy, az], [1, 1, 1], [0.04, 0.04, 0.04],
            radius=wr, height=0.26, segments=20, rotation=[0, 0, 90], roughness=0.9)

  rear = -hl - 0.01
  front = (cz + cl + 0.02) if kind == "truck" else hl + 0.01
  for side in (-1, 1):
    child(f"tail{'lr'[side > 0]}", "cube", [side * (hw - 0.22), hh * 0.35, rear], [0.18, 0.07, 0.02], [0.8, 0.05, 0.05],
          emissive=[2.5, 0.05, 0.05])
    child(f"head{'lr'[side > 0]}", "cube", [side * (hw - 0.22), (cy - ch * 1.2) if kind == "truck" else hh * 0.3, front],
          [0.17, 0.06, 0.02], [0.95, 0.95, 0.9], emissive=[1.5, 1.5, 1.3])
  child("plate", "cube", [0, -hh * 0.25, rear], [0.26, 0.07, 0.01], [0.95, 0.95, 0.95])
  return cmds


def road_commands(track: Track) -> list[dict]:
  cmds: list[dict] = []

  def ribbon(rid, d, width, color, y=0.0, dash=None, **kw):
    cmd = {"action": "create", "type": "ribbon", "id": rid, "points": track.polyline(d, y), "width": width, "color": color, **kw}
    if dash:
      cmd["dash"] = dash
    cmds.append(cmd)

  asphalt = {"roughness": 0.85, "metallic": 0.0}
  for name, center, sign in (("road", 0.0, 1), ("opp", OPPOSING_D, -1)):
    # shoulders: 1 m on the median side, 2.5 m on the outside
    inner, outer = center - sign * (EDGE + 1.0), center + sign * (EDGE + 2.5)
    ribbon(f"{name}_asphalt", 0.5 * (inner + outer), abs(outer - inner), [0.11, 0.11, 0.115], y=0.01, **asphalt)
    ribbon(f"{name}_edge_yellow", center - sign * EDGE, 0.15, [0.85, 0.65, 0.1], y=0.05, roughness=0.6)
    ribbon(f"{name}_edge_white", center + sign * EDGE, 0.15, [0.9, 0.9, 0.9], y=0.05, roughness=0.6)
    for j, dl in enumerate((-0.5, 0.5)):
      ribbon(f"{name}_dash{j}", center + dl * LANE_WIDTH, 0.12, [0.9, 0.9, 0.9], y=0.05, dash=[3.0, 9.0], roughness=0.6)
  ribbon("median", 0.5 * OPPOSING_D, 6.0, [0.18, 0.3, 0.12], y=0.005, roughness=1.0)
  return cmds


def props_commands(track: Track, rng: np.random.Generator, keep_clear=None) -> list[dict]:
  """Roadside props; keep_clear(s, d) -> True skips a prop there (rng draws are unchanged, so the rest stays put)"""
  cmds: list[dict] = []
  clear = keep_clear or (lambda s, d: False)

  # street lights along the outer shoulder
  s = 0.0
  i = 0
  while s < track.length:
    if clear(s, EDGE + 4.0):
      s += 60.0
      i += 1
      continue
    x, z, psi = track.pose(s, EDGE + 4.0)
    cmds.append({"action": "create", "type": "cylinder", "id": f"pole{i}", "position": [x, 5.0, z], "radius": 0.12,
                 "height": 10.0, "segments": 10, "color": [0.45, 0.46, 0.48], "metallic": 0.8, "roughness": 0.4})
    ax, az, _ = track.pose(s, EDGE + 2.2)
    cmds.append({"action": "create", "type": "cube", "id": f"lamp{i}", "position": [ax, 9.9, az], "rotation": [0, deg(psi + math.pi / 2), 0],
                 "scale": [0.12, 0.08, 1.9], "color": [0.45, 0.46, 0.48], "metallic": 0.8, "roughness": 0.4})
    s += 60.0
    i += 1

  # overhead sign gantries over our carriageway
  for g, s in enumerate((350.0, 2150.0, 3500.0)):
    for side, d in (("l", -EDGE - 2.0), ("r", EDGE + 3.5)):
      x, z, psi = track.pose(s, d)
      cmds.append({"action": "create", "type": "cube", "id": f"gantry{g}{side}", "position": [x, 3.6, z], "scale": [0.25, 3.6, 0.25],
                   "color": [0.5, 0.5, 0.52], "metallic": 0.8})
    x, z, psi = track.pose(s, 0.75)
    cmds.append({"action": "create", "type": "cube", "id": f"gantry{g}beam", "position": [x, 7.0, z], "rotation": [0, deg(psi), 0],
                 "scale": [EDGE + 3.0, 0.25, 0.25], "color": [0.5, 0.5, 0.52], "metallic": 0.8})
    for k, d in enumerate((-2.0, 3.0)):
      x, z, psi = track.pose(s - 0.3, d)
      cmds.append({"action": "create", "type": "cube", "id": f"gantry{g}sign{k}", "position": [x, 8.1, z], "rotation": [0, deg(psi), 0],
                   "scale": [2.2, 1.0, 0.05], "color": [0.0, 0.33, 0.18], "roughness": 0.5})

  # trees, kept clear of the carriageways
  for i in range(260):
    s = rng.uniform(0, track.length)
    d = rng.choice([rng.uniform(EDGE + 12, 90), rng.uniform(OPPOSING_D - 70, OPPOSING_D - EDGE - 10)])
    x, z, _ = track.pose(s, d)
    h = rng.uniform(5, 11)
    green = [rng.uniform(0.08, 0.16), rng.uniform(0.22, 0.38), rng.uniform(0.06, 0.12)]
    round_crown = rng.random() >= 0.5
    if clear(s, d):
      continue
    cmds.append({"action": "create", "type": "cylinder", "id": f"trunk{i}", "position": [x, h * 0.3, z], "radius": 0.25,
                 "height": h * 0.6, "segments": 8, "color": [0.3, 0.2, 0.12], "roughness": 0.95})
    if not round_crown:
      cmds.append({"action": "create", "type": "cone", "id": f"crown{i}", "position": [x, h * 0.45, z], "radius": h * 0.28,
                   "height": h * 0.85, "segments": 10, "color": green, "roughness": 0.95})
    else:
      r = h * 0.3
      cmds.append({"action": "create", "type": "sphere", "id": f"crown{i}", "position": [x, h * 0.62, z], "scale": [r, r * 0.9, r],
                   "segments": 12, "color": green, "roughness": 0.95})

  # buildings further out
  for i in range(70):
    s = rng.uniform(0, track.length)
    outside = rng.random() < 0.6
    d = rng.uniform(70, 220) if outside else rng.uniform(OPPOSING_D - 230, OPPOSING_D - 80)
    x, z, psi = track.pose(s, d)
    w, ln, h = rng.uniform(8, 25), rng.uniform(10, 35), rng.choice([rng.uniform(4, 12), rng.uniform(15, 60)])
    tone = rng.uniform(0.35, 0.8)
    tint = rng.uniform(0.9, 1.0), rng.uniform(0.85, 1.0)
    if clear(s, d):
      continue
    cmds.append({"action": "create", "type": "cube", "id": f"bldg{i}", "position": [x, h / 2, z], "rotation": [0, deg(psi), 0],
                 "scale": [w / 2, h / 2, ln / 2], "color": [tone, tone * tint[0], tone * tint[1]],
                 "roughness": 0.8})
  return cmds
