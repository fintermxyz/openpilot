import json
import math
import os
from dataclasses import dataclass

import numpy as np

from openpilot.tools.sim.bridge.zoomengine.track import Track

# A map built from ZoomEngine's road kit (tools/roadkit/assemble.py -> assets/maps/<name>.built.json): pre-built road
# sections placed by their connectors, signal and stop line positions and the straight-across lane routes.

# one openpilot car per approach, each driving straight across
ROUTES = ("N_S", "S_N", "E_W", "W_E")
CAR_HALF = (0.92, 2.3)  # sedan footprint half width / half length


@dataclass
class Route:
  name: str
  phase: str
  track: Track
  stop_line_s: float
  junction_s: tuple[float, float]
  spawn_s: float


class CrossroadsMap:
  def __init__(self, built_json: str):
    with open(built_json) as f:
      self.data = json.load(f)
    # statics are relative to the ZoomEngine checkout the map was built in: assets/maps/<name>.built.json
    self.root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(built_json))))
    self.name = self.data.get("name", "crossroads")
    self.speed_limit = float(self.data.get("speed_limit_mps", 13.41))
    self.routes: dict[str, Route] = {}
    for name, r in self.data["routes"].items():
      track = Track.from_points(np.asarray(r["points"], dtype=float), step=0.5, closed=False)
      spawn = float(self.data.get("spawns", {}).get(name, {}).get("s", 15.0))
      self.routes[name] = Route(name, r["phase"], track, float(r["stop_line_s"]), tuple(r["junction_s"]), spawn)

  def scene_commands(self) -> list[dict]:
    cmds = []
    g = self.data.get("ground")
    if g:  # a flat box: the API's plane primitive doesn't render in this scene
      w, d = float(g["size"][0]), float(g["size"][1])
      cmds.append({"action": "create", "type": "cube", "id": "ground", "scale": [w / 2, 0.05, d / 2],
                   "position": [0.0, float(g.get("y", -0.02)) - 0.05, 0.0], "color": g.get("color", [0.42, 0.55, 0.32]), "roughness": 0.95})
    for i, st in enumerate(self.data["statics"]):
      cmds.append({"action": "create", "type": "gltf", "id": f"kit{i}", "path": os.path.join(self.root, st["path"]),
                   "position": st["position"], "rotation": st.get("rotation", [0, 0, 0]), "scale": st.get("scale", [1, 1, 1])})
    for sig in self.data["signals"]:
      cmds += signal_commands(sig)
    return cmds


# ---- UK 3-aspect signal heads (red top, amber, green bottom; 300 mm aspects) ----

LAMP_OFF = {"red": [0.18, 0.02, 0.02], "amber": [0.2, 0.12, 0.02], "green": [0.02, 0.16, 0.06]}
LAMP_ON = {"red": [14.0, 0.5, 0.25], "amber": [14.0, 7.0, 0.4], "green": [0.4, 12.0, 5.0]}
LAMP_Y = {"red": 0.36, "amber": 0.0, "green": -0.36}
HEAD_Y = 3.0  # head centre above the road: lowest aspect ~2.5 m


def signal_commands(sig: dict) -> list[dict]:
  x, y, z = sig["position"]
  sid = sig["id"]
  rot = [0, round(float(sig["heading_deg"]), 3), 0]
  cmds = [
    {"action": "create", "type": "cylinder", "id": f"{sid}_pole", "position": [x, y + 1.8, z], "radius": 0.06, "height": 3.6,
     "segments": 10, "color": [0.32, 0.33, 0.34], "metallic": 0.6, "roughness": 0.5},
    # head: black housing, black backboard with a white border
    {"action": "create", "type": "cube", "id": sid, "position": [x, y + HEAD_Y, z], "rotation": rot, "scale": [0.17, 0.55, 0.12],
     "color": [0.02, 0.02, 0.02], "roughness": 0.6},
    {"action": "create", "type": "cube", "id": f"{sid}_border", "parent": sid, "position": [0, 0, -0.13], "scale": [0.36, 0.74, 0.005],
     "color": [0.92, 0.92, 0.92], "roughness": 0.7},
    {"action": "create", "type": "cube", "id": f"{sid}_board", "parent": sid, "position": [0, 0, -0.125], "scale": [0.33, 0.71, 0.006],
     "color": [0.02, 0.02, 0.02], "roughness": 0.7},
  ]
  for lamp, ly in LAMP_Y.items():
    cmds.append({"action": "create", "type": "cylinder", "id": f"{sid}_{lamp}", "parent": sid, "position": [0, ly, 0.125],
                 "rotation": [90, 0, 0], "radius": 0.15, "height": 0.02, "segments": 20, "color": LAMP_OFF[lamp], "roughness": 0.3,
                 "emissive": [0, 0, 0]})
  return cmds


def lamp_commands(sig_id: str, aspect: str) -> list[dict]:
  """aspect: 'red', 'red_amber', 'green', 'amber' or 'dark'"""
  lit = {"red": {"red"}, "red_amber": {"red", "amber"}, "green": {"green"}, "amber": {"amber"}, "dark": set()}[aspect]
  return [{"action": "material", "id": f"{sig_id}_{lamp}", "emissive": LAMP_ON[lamp] if lamp in lit else [0, 0, 0]} for lamp in LAMP_Y]


class SignalController:
  """UK two-phase cycle: green -> amber (3 s) -> all red -> red+amber (2 s) -> green for the other phase.
  fault 'stuck_green': from fault_at seconds every head shows green (conflicting greens)."""
  def __init__(self, green: float = 12.0, amber: float = 3.0, all_red: float = 2.0, red_amber: float = 2.0):
    self.seq = []  # (duration, NS aspect, EW aspect)
    for a, b in (("NS", "EW"), ("EW", "NS")):
      self.seq += [(green, a, "green"), (amber, a, "amber"), (all_red, a, "red"), (red_amber, b, "red_amber")]
    self.cycle = sum(d for d, _, _ in self.seq)
    self.fault = False

  def aspects(self, t: float) -> dict[str, str]:
    if self.fault:
      return {"NS": "green", "EW": "green"}
    t %= self.cycle
    for dur, phase, aspect in self.seq:
      if t < dur:
        other = "EW" if phase == "NS" else "NS"
        return {phase: aspect, other: "red"}
      t -= dur
    return {"NS": "red", "EW": "red"}


def footprint(x: float, z: float, psi: float) -> np.ndarray:
  """corners of a car footprint in (x, z)"""
  hw, hl = CAR_HALF
  f = np.array([math.sin(psi), math.cos(psi)])
  r = np.array([-math.cos(psi), math.sin(psi)])
  c = np.array([x, z])
  return np.array([c + f * hl + r * hw, c + f * hl - r * hw, c - f * hl - r * hw, c - f * hl + r * hw])


def overlap(a: np.ndarray, b: np.ndarray) -> bool:
  """separating axis test of two convex quads"""
  for poly in (a, b):
    for i in range(4):
      e = poly[(i + 1) % 4] - poly[i]
      n = np.array([-e[1], e[0]])
      pa, pb = a @ n, b @ n
      if pa.max() < pb.min() or pb.max() < pa.min():
        return False
  return True
