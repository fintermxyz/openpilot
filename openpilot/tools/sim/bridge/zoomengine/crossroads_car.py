import json
import os
import socket
import time
from multiprocessing import Queue

from openpilot.tools.sim.bridge.zoomengine.crossroads import ROUTES, CrossroadsMap
from openpilot.tools.sim.bridge.zoomengine.scene import car_commands
from openpilot.tools.sim.bridge.zoomengine.zoomengine_bridge import ZoomEngineBridge
from openpilot.tools.sim.bridge.zoomengine.zoomengine_world import CAM_HEIGHT, CAR_Y, ENGINE_HZ, ZoomEngineWorld
from openpilot.tools.sim.lib.camerad import W, H
from openpilot.tools.sim.lib.simulated_sensors import SimulatedSensors
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

# One openpilot-driven car at the crossroads. Run one bridge per car, each with its own OPENPILOT_PREFIX so every car has
# its own openpilot. Cars report their pose to the junction director (crossroads_director.py) over UDP; the director
# resets them all at once to start a trial.
DIRECTOR_PORT = 9190
CAR_PORT_BASE = 9200
CAR_COLORS = ([0.92, 0.93, 0.95], [0.7, 0.06, 0.06], [0.08, 0.2, 0.55], [0.9, 0.7, 0.08])
# picture-in-picture slot per car: top left, top right, bottom left, bottom right
PIP_W = 0.24
PIP_H = PIP_W * H / W * 16 / 9
PIPS = ([0.01, 0.10, PIP_W, PIP_H], [0.75, 0.10, PIP_W, PIP_H], [0.01, 0.97 - PIP_H, PIP_W, PIP_H], [0.75, 0.97 - PIP_H, PIP_W, PIP_H])


class CrossroadsCarWorld(ZoomEngineWorld):
  # the camera sits just ahead of the bumper: the other cars' cameras must see this car's body, so it can't be hidden
  # from sensor views, and a camera inside it would see its own bonnet from the inside
  CAM_FORWARD_M = 2.4

  def __init__(self, status_q, car: int, map_path: str, port: int = 9090):
    self.car = car
    self.route_name = ROUTES[car]
    self.EGO_ID = f"op{car}"
    self.CAM_IDS = (f"road{car}",)
    self.SHM_PATHS = (f"/tmp/zoom_sensor_road{car}.bin",)
    self.OVERLAY_PREFIX = f"op{car}_"
    self.cmap = CrossroadsMap(map_path)
    route = self.cmap.routes[self.route_name]
    self.START_S = route.spawn_s
    self.START_SPEED = self.cmap.speed_limit
    self.director = ("127.0.0.1", DIRECTOR_PORT)
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.bind(("127.0.0.1", CAR_PORT_BASE + car))
    self.sock.setblocking(False)
    self.trial = -1
    self.parked = False
    # L3 research layer, simulator only: built before the engine thread starts (the import goes through the sim-only gate)
    self.l3 = None
    if os.environ.get("L3") == "1":
      from openpilot.tools.sim.l3 import msgs as l3_msgs
      from openpilot.tools.sim.l3.host import L3Host
      self.l3_msgs = l3_msgs
      self.l3 = L3Host(self.EGO_ID, self.route_name, self.cmap)
      self.l3_mode = ""
    super().__init__(status_q, dual_camera=False, port=port, chase_camera=False)

  @property
  def lanes(self) -> tuple:
    return (0.0,)  # the route is the lane centre

  def _obstacles(self) -> list[str]:
    # car-to-car crashes with the other openpilot cars' bodies (top-level ids, so not their wheels / lights)
    return [f"op{c}" for c in range(len(ROUTES)) if c != self.car]

  def _setup_world(self, map_path, traffic, seed) -> None:
    self.map = None
    self.traffic = None
    self.opp_traffic = None
    self.track = self.cmap.routes[self.route_name].track
    self.ego_start_d = 0.0

  def _build_scene(self) -> None:
    """only this car and its camera, the director builds the junction"""
    api = self.api
    api.send({"action": "sensor_camera_remove", "id": self.CAM_IDS[0]})
    y, rot = self._ego_height_rotation(self.ego_s, self.ego_psi)
    api.batch(car_commands(self.EGO_ID, "sedan", CAR_COLORS[self.car], self.ego_x, self.ego_z, self.ego_psi, y0=y, rotation=rot))
    api.send(self._vehicle_spawn_command("vehicle"))  # the engine's physics Civic drives this body
    api.send({"action": "sensor_camera", "id": self.CAM_IDS[0], "attach": self.EGO_ID, "offset": [0, CAM_HEIGHT - CAR_Y, self.CAM_FORWARD_M],
              "yaw": 0, "pitch": 0, "width": W, "height": H, "hz": 20, "hfov": 40, "shm": self.SHM_PATHS[0],
              "format": "nv12", "nv12": list(get_nv12_info(W, H)), "pip": PIPS[self.car],
              "label": f"car {self.car + 1} {self.route_name.replace('_', ' to ')} (openpilot {self.car + 1})"})
    print(f"---- crossroads car {self.car} ({self.route_name}) ready, route {self.track.length:.0f} m ----", flush=True)

  def _poll_input(self, q: dict) -> None:
    pass  # the engine window's keys belong to the director

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    super().apply_controls(steer_angle, throttle_out, brake_out)
    if self.parked:  # reached the end of the route: wait for the next trial (overrides openpilot's pedals too)
      with self.lock:
        self.throttle_cmd, self.brake_cmd = 0.0, 1.0

  def _engine_step(self, ego, dt: float, frame: int) -> list[dict]:
    if self._handle_director():
      with self.lock:  # reset: the pose sampled before it is stale
        ego = (self.ego_x, self.ego_z, self.ego_psi, self.ego_v, self.ego_s, self.ego_d)
    if ego[4] > self.track.length - 8.0:
      self.parked = True
    cmds = [self._vehicle_input_command()]
    if frame % 2 == 0:
      self.sm.update(0)
      self._track_engagement(ego)
      cmds += self._overlay_commands()
    if frame % 10 == 0 and self.l3 is not None:
      cmds += self._l3_hud_commands()
    pose = {"car": self.car, "route": self.route_name, "trial": self.trial, "x": ego[0], "z": ego[1], "psi": ego[2], "v": ego[3],
            "s": ego[4], "d": ego[5], "engaged": self.op_active, "parked": self.parked, "t": time.monotonic()}
    if self.l3 is not None:
      pose["l3"] = self.l3.status()
      pose["l3_stats"] = self.l3.stats()  # for the director's results, not relayed to the other cars
    try:
      self.sock.sendto(json.dumps(pose).encode(), self.director)
    except OSError:
      pass
    if frame % ENGINE_HZ == 0:
      where = f"trial {self.trial} s {ego[4]:6.1f} m  v {ego[3]:5.2f} m/s  lane offset {ego[5]:+5.2f} m"
      print(f"[crossroads car {self.car}] {where}  engaged {self.op_active}  disengagements {self.disengagements}", flush=True)
    return cmds

  def _l3_tick(self) -> None:
    # control thread, self.lock held (ZoomEngineWorld.apply_controls)
    st = self.l3_msgs.EgoState(t=time.monotonic(), id=self.EGO_ID, route=self.route_name, x=self.ego_x, z=self.ego_z, psi=self.ego_psi,
                               v=self.ego_v, a=self.ego_accel, s=self.ego_s, d=self.ego_d, op_active=self.op_active)
    self.l3.tick(st)  # never raises: a failure latches the arbiter's FAILSAFE
    mode = self.l3.mode
    if mode != self.l3_mode:
      print(f"[l3 car {self.car}] mode {self.l3_mode or '--'} -> {mode} {self.l3.reason}", flush=True)
      self.l3_mode = mode

  def _l3_hud_commands(self) -> list[dict]:
    """this car's L3 state on the shared engine HUD: one line per car (ids prefixed per car), between the picture-in-pictures"""
    lines = self.l3.hud_lines()
    text = f"car {self.car + 1}  " + "   ".join(t for t, _ in lines)
    return [{"action": "hud_text", "id": f"{self.OVERLAY_PREFIX}l3", "text": text, "pos": [0.27, 0.115 + 0.03 * self.car],
             "color": lines[0][1], "scale": 0.9}]

  def _handle_director(self) -> bool:
    """True if the director started a new trial (ego was reset)"""
    reset = False
    while True:
      try:
        msg = json.loads(self.sock.recv(4096))
      except (BlockingIOError, ValueError):
        return reset
      if self.l3 is not None and msg.get("type") == "world":
        try:
          self.l3.feed(self.l3_msgs.WorldMsg.from_json(msg))
        except (KeyError, TypeError, ValueError) as e:
          print(f"[l3 car {self.car}] bad world message: {e!r}", flush=True)
        continue
      if "reset" in msg and msg["reset"] != self.trial:
        self.trial = msg["reset"]
        self.parked = False
        self.reset()  # ego state + the engine's physics car back to the spawn
        reset = True


class CrossroadsCarBridge(ZoomEngineBridge):
  def __init__(self, car: int, map_path: str, port: int = 9090):
    super().__init__(dual_camera=False, high_quality=False, port=port)
    self.car = car
    self.map_path = map_path

  def spawn_world(self, queue: Queue):
    # four openpilots share the machine: one IMU sample per 100 Hz tick and GPS at 10 Hz (a real receiver's rate) instead of
    # the default 5x / 10x repeats, which kept each car's locationd busy
    SimulatedSensors.IMU_REPEAT, SimulatedSensors.GPS_REPEAT, SimulatedSensors.GPS_EVERY = 1, 1, 10
    return CrossroadsCarWorld(queue, self.car, self.map_path, port=self.port)
