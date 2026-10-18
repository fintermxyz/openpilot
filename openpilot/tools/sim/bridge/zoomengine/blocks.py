import json
import os

import numpy as np

from openpilot.tools.sim.bridge.zoomengine.crossroads import CrossroadsMap, SignalController, lamp_commands
from openpilot.tools.sim.bridge.zoomengine.track import Track
from openpilot.tools.sim.bridge.zoomengine.traffic import Traffic
from openpilot.tools.sim.bridge.zoomengine.zoomengine_world import ENGINE_HZ, ZoomEngineWorld

# A block map (ZoomEngine tools/roadkit/blocks_map.py: straights, curves, crossroads, T junctions and roundabouts chained
# MetaDrive style) driven by the single openpilot car of the zoomengine sim: road + wide camera, the engine window's keys and
# gamepad, the same HUD. The route is open: at its end the car goes back to the start.

END_MARGIN_M = 20.0   # this far before the route's end the car is put back at its start
OFF_ROUTE_M = 6.0     # this far from the lane centre for OFF_ROUTE_S puts the car back on the route ...
OFF_ROUTE_S = 1.5
RESUME_AHEAD_M = 40.0 # ... this far past where it left it (openpilot can't pick a roundabout exit or turn by itself)
MAX_SPEED = 60.0      # m/s: faster means the physics blew up (e.g. off the ground's edge)


class BlocksMap(CrossroadsMap):
  def __init__(self, built_json: str):   # not super().__init__: that expects the crossroads' signalled routes
    with open(built_json) as f:
      self.data = json.load(f)
    self.root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(built_json))))
    self.name = self.data.get("name", "blocks")
    self.speed_limit = float(self.data.get("speed_limit_mps", 13.41))
    route = self.data["routes"]["main"]
    self.track = Track.from_points(np.asarray(route["points"], dtype=float), step=0.5, closed=False)
    self.spawn_s = float(self.data.get("spawns", {}).get("main", {}).get("s", 15.0))
    self.lane_offsets = (0.0,)        # the route is the lane centre
    self.opposing_track = None
    self.road_width = 7.3             # one lane each way
    # signal heads by junction (approach ids are <junction><arm>)
    self.junction_signals: dict[str, list[dict]] = {}
    for sig in self.data["signals"]:
      self.junction_signals.setdefault(sig["approach"][:-1], []).append(sig)


class BlocksWorld(ZoomEngineWorld):
  def __init__(self, status_q, dual_camera: bool, port: int = 9090, seed: int = 0, traffic: str = "light", map_path: str = ""):
    self._blocks_path = map_path
    bm = BlocksMap(map_path)
    self.START_S = bm.spawn_s
    self.START_SPEED = 0.8 * bm.speed_limit
    self.off_since: float | None = None
    self.recoveries = 0
    self.lit: dict[str, str] = {}
    super().__init__(status_q, dual_camera, port=port, seed=seed, traffic=traffic, chase_camera=True, map_path=map_path)

  def _setup_world(self, map_path, traffic, seed) -> None:
    self.map = BlocksMap(self._blocks_path)
    self.track = self.map.track
    self.ego_start_d = 0.0
    self.opp_traffic = None
    # a few cars ahead in the ego's direction only: the opposite lane isn't an offset of the route through a roundabout
    n = max(2, int(self.track.length / 250.0))
    self.traffic = Traffic(self.track, self.rng, self.START_S, traffic, lanes=(0.0,), closure=False, n_same=n, n_opposing=0, ego_lane=0)
    self.signals = {j: SignalController() for j in self.map.junction_signals}
    self.signal_offset = {j: float(self.rng.uniform(0, c.cycle)) for j, c in self.signals.items()}
    self.t_world = 0.0
    self.suburb_rng = np.random.default_rng(seed + 1)

  def _engine_step(self, ego, dt: float, frame: int) -> list[dict]:
    cmds = super()._engine_step(ego, dt, frame)
    self.t_world += dt
    if frame % (ENGINE_HZ // 5) == 0:
      for j, sigs in self.map.junction_signals.items():
        aspects = self.signals[j].aspects(self.t_world + self.signal_offset[j])
        for sig in sigs:
          a = aspects[sig["phase"]]
          if self.lit.get(sig["id"]) != a:
            self.lit[sig["id"]] = a
            cmds += lamp_commands(sig["id"], a)
    if ego[4] > self.track.length - END_MARGIN_M:
      print(f"---- end of the route ({self.track.length:.0f} m): back to the start ----", flush=True)
      self.reset()
      return cmds
    off = abs(ego[5]) > OFF_ROUTE_M
    self.off_since = (self.off_since or self.t_world) if off else None
    if abs(ego[3]) > MAX_SPEED or (off and self.t_world - self.off_since > OFF_ROUTE_S):
      s_new = ego[4] + RESUME_AHEAD_M
      if s_new > self.track.length - END_MARGIN_M - 10.0:
        s_new = self.map.spawn_s
      self.recoveries += 1
      print(f"---- off the route at s {ego[4]:.0f} m (lane offset {ego[5]:+.1f} m, v {ego[3]:.1f} m/s): back on it at " +
            f"s {s_new:.0f} m (recovery {self.recoveries}) ----", flush=True)
      self.place_at(s_new)
    return cmds

  def place_at(self, s: float) -> None:
    """the car (ego state + the engine's physics car) on the route at s, at the start speed"""
    start = self.START_S
    self.START_S = s
    self.reset()
    self.START_S = start
    self.off_since = None
