import json
import math
import socket
import time
from multiprocessing import Queue

from openpilot.common.params import Params
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.sim.bridge.zoomengine.crossroads import map_is_left_hand
from openpilot.tools.sim.bridge.zoomengine.crossroads_car import CrossroadsCarBridge, CrossroadsCarWorld
from openpilot.tools.sim.bridge.zoomengine.town import TownMap, town_layout
from openpilot.tools.sim.bridge.zoomengine.town_director import CAR_PORT_BASE, DIRECTOR_PORT
from openpilot.tools.sim.bridge.zoomengine.zoomengine_world import CAM_HEIGHT, CAR_Y, ENGINE_HZ, ZoomEngineWorld
from openpilot.tools.sim.lib.camerad import W, H
from openpilot.tools.sim.lib.simulated_sensors import SimulatedSensors

# One openpilot-driven car in the town: it drives one of the four loop routes (straight through every junction) for as long
# as the town runs. Like the crossroads cars it builds only its own body + road camera; the town director builds and runs
# the rest and gets this car's pose and what it hit (NPC cars, pedestrians) over UDP. This openpilot's longitudinal planner
# always takes the most cautious of the lead MPC, cruise and the end-to-end model's action, so the model's stopping for red
# lights and slowing for the bends is always in play (there is no separate experimental mode).


class TownCarWorld(CrossroadsCarWorld):
  def __init__(self, status_q, car: int, map_path: str, port: int = 9090):
    self.car = car
    self.EGO_ID = f"op{car}"
    # road + wide camera, like the single-car sim: with only the 40 deg road camera the model loses the lane in the town's
    # 91 m bends and runs wide (the wide camera is openpilot input only, no picture-in-picture)
    self.CAM_IDS = (f"road{car}", f"wide{car}")
    self.SHM_PATHS = (f"/tmp/zoom_sensor_road{car}.bin", f"/tmp/zoom_sensor_wide{car}.bin")
    self.OVERLAY_PREFIX = f"op{car}_"
    self.cmap = TownMap(map_path)
    self.route_name = self.cmap.car_routes[car]
    route = self.cmap.routes[self.route_name]
    self.START_S = route.spawn_s
    self.START_SPEED = self.cmap.speed_limit * 0.8
    self.director = ("127.0.0.1", DIRECTOR_PORT)
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.bind(("127.0.0.1", CAR_PORT_BASE + car))
    self.sock.setblocking(False)
    self.trial = -1
    self.parked = False
    self.new_hits: dict[str, list[float]] = {}
    ZoomEngineWorld.__init__(self, status_q, dual_camera=True, port=port, chase_camera=False)

  def _build_scene(self) -> None:
    self.api.send({"action": "sensor_camera_remove", "id": self.CAM_IDS[1]})
    super()._build_scene()
    self.api.send({"action": "sensor_camera", "id": self.CAM_IDS[1], "attach": self.EGO_ID,
                   "offset": [0, CAM_HEIGHT - CAR_Y, self.CAM_FORWARD_M], "yaw": 0, "pitch": 0, "width": W, "height": H, "hz": 20,
                   "hfov": 120, "shm": self.SHM_PATHS[1], "format": "nv12", "nv12": list(get_nv12_info(W, H)),
                   "pip": town_layout(self.car)[1], "label": f"car {self.car + 1} wide (openpilot {self.car + 1})"})

  def _road_pip(self) -> list[float]:
    return town_layout(self.car)[0]

  def _obstacles(self) -> list[str]:
    # the other openpilot cars' bodies, the town's NPC cars and pedestrians
    return [f"op{c}" for c in range(len(self.cmap.car_routes)) if c != self.car] + ["npc", "ped"]

  def _apply_hits(self, hits: dict) -> None:
    """what this car did to NPCs / pedestrians goes to the director, who owns them (other openpilot cars handle themselves)"""
    for oid, h in hits.items():
      prev = self.hits_seen.get(oid, [0.0, 0.0, 0.0])
      dv = [h["dv"][i] - prev[i] for i in range(3)]
      self.hits_seen[oid] = h["dv"]
      if oid.startswith("op") or max(abs(x) for x in dv) < 1e-3:
        continue
      acc = self.new_hits.setdefault(oid, [0.0, 0.0, 0.0])
      for i in range(3):
        acc[i] += dv[i]
      print(f"[town car {self.car}] HIT {oid}: closing {h['speed']:.1f} m/s, its dv {math.hypot(dv[0], dv[2]):.1f} m/s", flush=True)

  def _engine_step(self, ego, dt: float, frame: int) -> list[dict]:
    if self._handle_director():   # the director put this car back at its spawn
      with self.lock:
        ego = (self.ego_x, self.ego_z, self.ego_psi, self.ego_v, self.ego_s, self.ego_d)
    cmds = [self._vehicle_input_command()]
    if frame % 2 == 0:
      self.sm.update(0)
      self._track_engagement(ego)
      cmds += self._overlay_commands()
    pose = {"car": self.car, "route": self.route_name, "x": ego[0], "z": ego[1], "psi": ego[2], "v": ego[3], "s": ego[4], "d": ego[5],
            "engaged": self.op_active, "t": time.monotonic()}
    if self.new_hits:
      pose["hits"], self.new_hits = self.new_hits, {}
    try:
      self.sock.sendto(json.dumps(pose).encode(), self.director)
    except OSError:
      pass
    if frame % (ENGINE_HZ // 2) == 0:
      road_k = self.track.curvature(ego[4])
      op_k = self.sm['carControl'].actuators.curvature
      print(f"[town car {self.car}] {self.route_name} s {ego[4]:7.1f} m  v {ego[3]:5.2f} m/s  lane offset {ego[5]:+5.2f} m  " +
            f"steer {self.steer_deg:+6.1f} (cmd {self.steer_cmd:+6.1f}) deg  curvature road {road_k:+.4f} op {op_k:+.4f}  " +
            f"engaged {self.op_active}  disengagements {self.disengagements}", flush=True)
    return cmds


class TownCarBridge(CrossroadsCarBridge):
  def __init__(self, car: int, map_path: str, port: int = 9090):
    super().__init__(car, map_path, port)
    self.dual_camera = True  # road + wide camera into openpilot (see TownCarWorld)

  def spawn_world(self, queue: Queue):
    SimulatedSensors.IMU_REPEAT, SimulatedSensors.GPS_REPEAT, SimulatedSensors.GPS_EVERY = 1, 1, 10
    SimulatedSensors.IS_RHD = map_is_left_hand(self.map_path)   # UK town: a right-hand-drive car
    # a simulated crash in an earlier run can leave openpilot's excessive-actuation lockout set (a real car clears it in the
    # UI), which blocks every later start
    Params().remove("Offroad_ExcessiveActuation")
    return TownCarWorld(queue, self.car, self.map_path, port=self.port)
