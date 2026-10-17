import math
import threading
import time

import numpy as np

from opendbc.car.honda.interface import CarInterface
from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import messaging
from openpilot.tools.sim.bridge.common import ManualInput, QueueMessage, QueueMessageType, control_cmd_gen
from openpilot.tools.sim.bridge.zoomengine.beamng_map import BeamngMap
from openpilot.tools.sim.bridge.zoomengine.scene import CAR_Y, EDGE, car_commands, car_rotation, props_commands, road_commands
from openpilot.tools.sim.bridge.zoomengine.suburb import keep_clear, suburb_commands
from openpilot.tools.sim.bridge.zoomengine.track import Track
from openpilot.tools.sim.bridge.zoomengine.traffic import PRESETS, Traffic
from openpilot.tools.sim.bridge.zoomengine.zoom_api import SensorReader, ZoomAPI
from openpilot.tools.sim.lib.common import SimulatorState, World, vec3
from openpilot.tools.sim.lib.camerad import W, H
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

# the plant is openpilot's own vehicle model of the car launch_openpilot.sh fingerprints as
CAR_FINGERPRINT = "HONDA_CIVIC_2022"
STEER_RATE = 300.  # deg/s of steering wheel
GRAVITY = 9.81

CAM_HEIGHT = 1.22      # comma device height above the road
CAM_FORWARD = 0.6      # mount position ahead of the car center
EGO_START_S = 100.0
EGO_START_SPEED = 20.0

SHM_ROAD = "/tmp/zoom_sensor_road.bin"
SHM_WIDE = "/tmp/zoom_sensor_wide.bin"

ENGINE_HZ = 50

# manual driving from the engine window (keys polled from the agent API, see _poll_input)
MAX_LAT_ACCEL = 4.0      # m/s^2, full A/D lock at speed
MAX_CURVATURE = 0.12     # 1/m, full lock when slow
STEER_EASE = 8.0         # 1/s, steering eases toward the key target (like the drive demo)
MANUAL_KEYS = ("w", "a", "s", "d", "up", "down", "left", "right", "space")  # pressing any while engaged disengages
PAD_TAKEOVER = 0.25      # gamepad stick / trigger past this while engaged disengages (resting noise never does)
TRIGGER_DEADZONE = 0.05
TAP_ALIASES = {"select": "e", "pad_up": "+", "pad_down": "-"}  # gamepad buttons that act like keys


def _trigger(v: float) -> float:
  return float(np.clip((v - TRIGGER_DEADZONE) / (1.0 - TRIGGER_DEADZONE), 0.0, 1.0))


class ZoomEngineWorld(World):
  def __init__(self, status_q, dual_camera: bool, port: int = 9090, seed: int = 0, traffic: str = "heavy", chase_camera: bool = True,
               map_path: str | None = None):
    super().__init__(dual_camera)
    self.status_q = status_q
    self.chase_camera = chase_camera
    self.status_q.put(QueueMessage(QueueMessageType.START_STATUS, "starting"))

    self.api = ZoomAPI(port=port)
    print(f"---- waiting for ZoomEngine on {self.api.url} ----")
    self.api.wait_ready()

    self.VM = VehicleModel(CarInterface.get_non_essential_params(CAR_FINGERPRINT))
    self.lock = threading.Lock()
    self.rng = np.random.default_rng(seed)
    self.map = BeamngMap(map_path) if map_path else None
    self.opp_traffic = None
    if self.map is not None:
      # BeamNG level: drive its Autobahn in the right lane, the other carriageway gets its own traffic
      self.track = self.map.track
      lanes = self.map.lane_offsets
      self.ego_start_d = lanes[-1]
      self.traffic = Traffic(self.track, self.rng, EGO_START_S, traffic, lanes=lanes, n_opposing=0, ego_lane=len(lanes) - 1)
      if self.map.opposing_track is not None:
        self.opp_traffic = Traffic(self.map.opposing_track, self.rng, 0.0, traffic, lanes=lanes, closure=False,
                                   n_same=PRESETS[traffic].n_opposing, n_opposing=0, prefix="opp", ego_lane=None)
    else:
      self.track = Track()
      self.ego_start_d = 0.0
      self.traffic = Traffic(self.track, self.rng, EGO_START_S, traffic)
    self.suburb_rng = np.random.default_rng(seed + 1)  # own stream: the highway layout and traffic stay the same
    self._reset_ego()
    self._build_scene()

    # the engine converts to NV12 on the GPU, straight into VisionIPC's layout: no per-frame RGB->YUV here
    nv12 = get_nv12_info(W, H)
    self.road_yuv = np.zeros(nv12[3], dtype=np.uint8)
    self.road_reader = SensorReader(SHM_ROAD, W, H, nv12_size=nv12[3])
    self.wide_reader = None
    if dual_camera:
      self.wide_road_yuv = np.zeros(nv12[3], dtype=np.uint8)
      self.wide_reader = SensorReader(SHM_WIDE, W, H, nv12_size=nv12[3])

    self.sm = messaging.SubMaster(['modelV2', 'selfdriveState', 'carControl', 'carState', 'radarState', 'onroadEvents'])
    self.cam_pos = None
    self.collisions = 0
    self.was_colliding = False
    self.collision_kinds = {"ego": 0, "npc": 0, "cut-in": 0}
    # disengagement tracking, the bridge re-engages unless the disengagement was user initiated
    self.was_active = False
    self.disengaged_at: float | None = None
    self.user_disengaged = False
    self.driver_took_over = False  # WASD / E cancelled openpilot: no auto re-engage until it is engaged again
    self.disengagements = 0
    self.last_disengage_reason = ""
    # engine window keys
    self.held: dict[str, int] = {}
    self.pad: dict[str, float] = {}
    self.manual_held = False
    self.taps: dict[str, int] | None = None
    self.manual_steer = 0.0
    self.manual_t = time.monotonic()
    self.engine_thread = threading.Thread(target=self._engine_loop, daemon=True)
    self.engine_thread.start()
    self.status_q.put(QueueMessage(QueueMessageType.START_STATUS, "started"))

  # ---- scene ----
  def _build_scene(self):
    api = self.api
    api.send({"action": "clear"})
    if self.map is None:
      # clear also removes the engine scene's own ground, so lay grass again (a little below the road ribbons so they never
      # z-fight with it); it covers the loop and the suburb with kilometres to spare. BeamNG levels bring their terrain.
      api.send({"action": "create", "type": "plane", "id": "ground", "width": 6000.0, "height": 6000.0, "position": [650.0, -0.04, 600.0],
                "color": [0.62, 0.72, 0.52], "texture": "assets/textures/pasture/pasture_col.png",
                "normal_map": "assets/textures/pasture/pasture_nrm.png", "uv_scale": [0.25, 0.25, 0.0], "roughness": 0.95, "metallic": 0.0})
    for cam_id in ("road", "wide"):
      api.send({"action": "sensor_camera_remove", "id": cam_id})
    if self.map is not None:
      scene = self.map.scene_commands()
    else:
      scene = road_commands(self.track) + props_commands(self.track, self.rng, keep_clear) + suburb_commands(self.track, self.suburb_rng)
    cmds = scene + self.traffic.create_commands()
    if self.opp_traffic is not None:
      cmds += self.opp_traffic.create_commands()
    y, rot = self._ego_height_rotation(self.ego_s, self.ego_psi)
    cmds += car_commands("ego", "sedan", [0.92, 0.93, 0.95], self.ego_x, self.ego_z, self.ego_psi, sensor_hidden=True,
                         y0=y, rotation=rot)
    api.batch(cmds, chunk=200)

    pip_w = 0.32
    pip_h = pip_w * H / W * 16 / 9  # keep the camera aspect on a 16:9 window
    cam = {"action": "sensor_camera", "attach": "ego", "offset": [0, CAM_HEIGHT - CAR_Y, CAM_FORWARD], "yaw": 0, "pitch": 0,
           "width": W, "height": H, "hz": 20, "format": "nv12", "nv12": list(get_nv12_info(W, H))}
    api.send({**cam, "id": "road", "hfov": 40, "shm": SHM_ROAD, "pip": [0.67, 0.02, pip_w, pip_h], "label": "road camera (openpilot input)"})
    if self.dual_camera:
      api.send({**cam, "id": "wide", "hfov": 120, "shm": SHM_WIDE, "pip": [0.67, 0.04 + pip_h, pip_w, pip_h],
                "label": "wide camera (openpilot input)"})
    api.send({"action": "sim_input", "enabled": True})  # the engine window's keys drive the car / engage openpilot
    name = self.map.name if self.map is not None else "highway loop"
    print(f"---- ZoomEngine scene built ({name}): {len(cmds)} objects, track {self.track.length:.0f} m ----")

  def _ego_height_rotation(self, s: float, psi: float) -> tuple[float, list[float]]:
    return self.track.height(s) + CAR_Y, car_rotation(psi, self.track.pitch(s))

  def _reset_ego(self):
    with self.lock:
      self.ego_x, self.ego_z, self.ego_psi = self.track.pose(EGO_START_S, self.ego_start_d)
      self.ego_v = EGO_START_SPEED
      self.ego_yaw_rate = 0.0
      self.ego_accel = 0.0
      self.steer_deg = 0.0
      self.steer_cmd = 0.0
      self.accel_cmd = 0.0
      self.ego_s, self.ego_d = EGO_START_S, self.ego_start_d
      self.op_active = False
      self.ego_pitch = self.track.pitch(EGO_START_S)
      self.last_t = time.monotonic()
      self.reset_time = self.last_t

  # ---- vehicle ----
  def _driver_steer(self) -> float:
    """steering wheel angle [deg] a human driver would use to hold the current lane (pure pursuit)"""
    lanes = self.traffic.lanes
    lane_d = min(lanes, key=lambda d: abs(d - self.ego_d))
    lookahead = max(12.0, 1.3 * self.ego_v)
    tx, tz, _ = self.track.pose(self.ego_s + lookahead, lane_d)
    alpha = math.atan2(tx - self.ego_x, tz - self.ego_z) - self.ego_psi
    alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
    curvature = 2.0 * math.sin(alpha) / lookahead
    return math.degrees(self.VM.get_steer_from_curvature(curvature, max(self.ego_v, 1.0), 0.0))

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    with self.lock:
      now = time.monotonic()
      dt = min(0.05, now - self.last_t)
      self.last_t = now
      # until openpilot first engages, and while it's being re-engaged, the driver holds the lane; never once
      # the driver has taken over (engine window keys / gamepad) or is holding a driving key
      if not self.op_active and not self.driver_took_over and not self.manual_held and abs(steer_angle) < 1e-6:
        steer_angle = self._driver_steer()
      self.steer_cmd = float(steer_angle)
      self.accel_cmd = 1.6 * float(throttle_out) - 4.0 * float(brake_out)

      self.steer_deg += float(np.clip(self.steer_cmd - self.steer_deg, -STEER_RATE * dt, STEER_RATE * dt))
      v = self.ego_v
      drag = 0.02 + 0.0004 * v * v if v > 0.1 else 0.0
      self.ego_pitch = self.track.pitch(self.ego_s)
      grade = GRAVITY * math.sin(self.ego_pitch) if v > 0.1 or self.accel_cmd > 0 else 0.0  # parked cars don't roll back
      self.ego_v = max(0.0, v + (self.accel_cmd - drag - grade) * dt)
      self.ego_accel = (self.ego_v - v) / dt if dt > 0 else 0.0
      self.ego_yaw_rate = self.ego_v * self.VM.calc_curvature(math.radians(self.steer_deg), self.ego_v, 0.0)
      self.ego_psi += self.ego_yaw_rate * dt
      self.ego_x += self.ego_v * math.sin(self.ego_psi) * dt
      self.ego_z += self.ego_v * math.cos(self.ego_psi) * dt
      self.ego_s, self.ego_d = self.track.project(self.ego_x, self.ego_z, self.ego_s)

  def manual_input(self) -> ManualInput:
    """Held keys from the engine window -> gas/brake and an eased steering wheel angle (used while disengaged)"""
    held, pad = self.held, self.pad
    with self.lock:
      v, steer_deg = self.ego_v, self.steer_deg
    now = time.monotonic()
    dt = min(0.05, now - self.manual_t)
    self.manual_t = now
    gas = max(1.0 if held.get("w") or held.get("up") else 0.0, _trigger(pad.get("gas", 0.0)))
    brake = max(1.0 if held.get("s") or held.get("down") or held.get("space") else 0.0, _trigger(pad.get("brake", 0.0)))
    steer_in = float(bool(held.get("a") or held.get("left"))) - float(bool(held.get("d") or held.get("right")))  # +1 = left
    steer_in = float(np.clip(steer_in - pad.get("steer", 0.0), -1.0, 1.0))  # stick right = +1: proportional steering
    if self.was_active:
      self.manual_steer = steer_deg  # openpilot is steering: pick up from its wheel angle when it lets go
    else:
      u = max(v, 1.0)
      curvature = min(MAX_CURVATURE, MAX_LAT_ACCEL / (u * u))
      target = steer_in * math.degrees(self.VM.get_steer_from_curvature(curvature, u, 0.0))
      self.manual_steer += (target - self.manual_steer) * min(1.0, dt * STEER_EASE)
    return ManualInput(gas, brake, -0.15 * steer_in, self.manual_steer)

  def _poll_input(self) -> None:
    """Engine window keys and gamepad (sim_input): WASD / left stick + R2 / L2 drive (and take over from openpilot),
    E / select engage / disengage, +/- / d-pad change the set speed while engaged, R resets, Q quits"""
    inp = self.api.query().get("input")
    if not inp:
      return
    self.held = inp.get("held", {})
    self.pad = inp.get("pad", {}) if inp.get("pad", {}).get("connected") else {}
    pad_active = abs(self.pad.get("steer", 0.0)) > PAD_TAKEOVER or max(self.pad.get("gas", 0.0), self.pad.get("brake", 0.0)) > PAD_TAKEOVER
    manual = pad_active or any(self.held.get(k) for k in MANUAL_KEYS)
    if manual and not self.manual_held and self.was_active:
      self._driver_cancel()  # the driver takes over
    self.manual_held = manual

    taps = inp.get("taps", {})
    if self.taps is not None:
      for name in ("e", "+", "-", "r", "q", *TAP_ALIASES):
        key = TAP_ALIASES.get(name, name)
        for _ in range(max(0, taps.get(name, 0) - self.taps.get(name, 0))):
          if key == "e" and self.was_active:
            self._driver_cancel()
            continue
          cmd = {"e": "cruise_down",  # set: engage at the current speed
                 "+": "cruise_up" if self.was_active else None,
                 "-": "cruise_down" if self.was_active else None,
                 "r": "reset", "q": "quit"}[key]
          if cmd is not None:
            self.status_q.put(control_cmd_gen(cmd))
    self.taps = taps

  def _driver_cancel(self) -> None:
    # the cancel button event lasts one frame, too short for _track_engagement to see reliably, so remember it here
    self.driver_took_over = True
    self.status_q.put(control_cmd_gen("cruise_cancel"))

  def read_state(self):
    pass

  def read_sensors(self, state: SimulatorState):
    with self.lock:
      v, psi = self.ego_v, self.ego_psi
      state.velocity = vec3(v * math.sin(psi), v * math.cos(psi), 0.0)
      state.bearing = (-math.degrees(psi)) % 360
      state.imu.bearing = state.bearing
      state.steering_angle = self.steer_deg
      # raw IMU axes as locationd expects them: device frame (x fwd, y right, z down) = [-v[2], -v[1], -v[0]]
      state.imu.gyroscope = vec3(self.ego_yaw_rate, 0.0, 0.0)
      # specific force: gravity tilts into the forward axis on hills
      p = self.ego_pitch
      state.imu.accelerometer = vec3(GRAVITY * math.cos(p), v * self.ego_yaw_rate, -(self.ego_accel + GRAVITY * math.sin(p)))
      state.gps.from_xy((self.ego_z, -self.ego_x))  # north = +z, east = -x
      state.valid = True

  def read_cameras(self):
    if self.wide_reader is not None:
      self.wide_reader.read(self.wide_road_yuv)
    if self.road_reader.read(self.road_yuv):
      self.image_lock.release()  # wakes the camerad thread to send the new frame

  def tick(self):
    pass

  def reset(self):
    self._reset_ego()

  def close(self, reason: str):
    self.status_q.put(QueueMessage(QueueMessageType.CLOSE_STATUS, reason))
    self.exit_event.set()
    self.road_reader.close()
    if self.wide_reader is not None:
      self.wide_reader.close()

  # ---- engine sync: traffic, ego pose, chase camera, overlays ----
  def _engine_loop(self):
    period = 1.0 / ENGINE_HZ
    frame = 0
    last = time.monotonic()
    while not self.exit_event.is_set():
      now = time.monotonic()
      dt, last = now - last, now
      with self.lock:
        ego = (self.ego_x, self.ego_z, self.ego_psi, self.ego_v, self.ego_s, self.ego_d)
      self.traffic.step(dt, (ego[4], ego[5], ego[3]))
      if self.opp_traffic is not None:
        self.opp_traffic.step(dt, None)

      cmds = self.traffic.engine_commands((ego[0], ego[1]), frame)
      if self.opp_traffic is not None:
        cmds += self.opp_traffic.engine_commands((ego[0], ego[1]), frame)
      y, rot = self._ego_height_rotation(ego[4], ego[2])
      # "smooth": the engine glides between these 50 Hz samples, so the view doesn't stutter at 60-120 fps
      cmds.append({"action": "move", "id": "ego", "position": [round(ego[0], 3), round(y, 3), round(ego[1], 3)], "smooth": True})
      cmds.append({"action": "rotate", "id": "ego", "rotation": rot, "smooth": True})
      if self.chase_camera:
        cmds.append(self._chase_camera(ego, dt))
      if frame % 2 == 0:
        self.sm.update(0)
        self._track_engagement(ego)
        cmds += self._overlay_commands()
      if frame % 10 == 0:
        cmds += self._hud_commands(ego)
      if frame % ENGINE_HZ == 0:
        pos = f"s {ego[4]:7.1f} m  lane offset {ego[5]:+6.2f} m  v {ego[3]:5.2f} m/s"
        ctl = f"steer {self.steer_deg:+6.2f} (cmd {self.steer_cmd:+6.2f}) deg  accel cmd {self.accel_cmd:+5.2f}"
        lead = self.sm['radarState'].leadOne
        lead_str = f"{lead.dRel:5.1f} m" if lead.present else "--"
        status = f"lead {lead_str}  collisions {self.collisions}  disengagements {self.disengagements}  {self.traffic.stats}"
        print(f"[zoomengine] {pos}  {ctl}  {status}", flush=True)
      try:
        self.api.batch(cmds)
        self._poll_input()
      except OSError:
        print("ZoomEngine went away, stopping")
        self.exit_event.set()
        return
      frame += 1
      time.sleep(max(0.0, period - (time.monotonic() - now)))

  def _chase_camera(self, ego, dt):
    x, z, psi = ego[0], ego[1], ego[2]
    h = self.track.height(ego[4])
    fx, fz = math.sin(psi), math.cos(psi)
    want = np.array([x - 11.0 * fx, h + 4.2, z - 11.0 * fz])
    a = 1.0 - math.exp(-dt * 4.0)
    self.cam_pos = want if self.cam_pos is None else self.cam_pos + a * (want - self.cam_pos)
    return {"action": "camera", "position": [round(float(c), 3) for c in self.cam_pos], "target": [x + 10.0 * fx, h + 1.2, z + 10.0 * fz],
            "smooth": True}

  @staticmethod
  def _to_ego(x, y, max_x=None):
    """device frame (x fwd, y right) -> ego object local frame (+x left, +y up, +z fwd) at road level"""
    pts = []
    for xi, yi in zip(x, y, strict=True):
      if max_x is not None and xi > max_x:
        break
      pts.append([round(-yi, 3), round(0.03 - CAR_Y, 3), round(xi + CAM_FORWARD, 3)])
    return pts

  def _overlay_commands(self) -> list[dict]:
    if not self.sm.updated['modelV2'] and self.sm.recv_frame['modelV2'] == 0:
      return []
    md = self.sm['modelV2']
    engaged = self.sm['selfdriveState'].active
    cmds = []
    if len(md.position.x):
      color = [0.1, 0.9, 0.35, 0.55] if engaged else [0.9, 0.9, 0.9, 0.45]
      cmds.append({"action": "overlay_path", "id": "plan", "attach": "ego", "points": self._to_ego(md.position.x, md.position.y, 120.0),
                   "color": color, "width": 1.0})
    for i, (line, prob) in enumerate(zip(md.laneLines, md.laneLineProbs, strict=False)):
      if len(line.x):
        cmds.append({"action": "overlay_path", "id": f"lane{i}", "attach": "ego", "points": self._to_ego(line.x, line.y, 100.0),
                     "color": [1.0, 1.0, 1.0, round(0.15 + 0.7 * float(prob), 3)], "width": 0.12})
    for i, edge in enumerate(md.roadEdges):
      if len(edge.x):
        cmds.append({"action": "overlay_path", "id": f"edge{i}", "attach": "ego", "points": self._to_ego(edge.x, edge.y, 100.0),
                     "color": [1.0, 0.15, 0.15, 0.6], "width": 0.12})
    lead = self.sm['radarState'].leadOne
    if lead.present:
      d, y = lead.dRel, lead.yRel
      cmds.append({"action": "overlay_path", "id": "lead", "attach": "ego",
                   "points": self._to_ego([d - 1.2, d], [-y, -y]), "color": [1.0, 0.25, 0.1, 0.8], "width": 2.0})
    else:
      cmds.append({"action": "overlay_path_remove", "id": "lead"})
    return cmds

  def _hud_commands(self, ego) -> list[dict]:
    ss, cs, cc = self.sm['selfdriveState'], self.sm['carState'], self.sm['carControl']
    lead = self.sm['radarState'].leadOne
    state = "ENGAGED" if ss.active else ("off, you're driving (E to engage)" if ss.engageable else "not ready, you're driving")
    set_speed = f"{cs.vCruise:.0f} km/h" if ss.active else "--"
    colliding = self._colliding(ego)
    if colliding and not self.was_colliding:
      self.collisions += 1
      self.collision_kinds[colliding] += 1
    self.was_colliding = colliding is not None
    lead_str = f"{lead.dRel:5.1f} m" if lead.present else "  --  "
    disengaged = f"disengagements {self.disengagements} {self.last_disengage_reason}"
    if self.map is not None:
      on_highway = abs(ego[5]) < self.map.road_width / 2 + 3.0
      where = f"lane offset {ego[5]:+5.2f} m" if on_highway else "off the Autobahn"
    else:
      on_highway = abs(ego[5]) < EDGE + 2.5
      where = f"lane offset {ego[5]:+5.2f} m" if on_highway else ("in the suburb" if keep_clear(ego[4], ego[5]) else "off the highway")
    lines = [
      (f"openpilot {state}   speed {ego[3] * 3.6:5.1f} km/h   set {set_speed}", [0.2, 1.0, 0.4] if ss.active else [1.0, 1.0, 1.0]),
      (f"steer {self.steer_deg:+6.1f} deg (cmd {self.steer_cmd:+6.1f})   accel cmd {cc.actuators.accel:+5.2f} m/s^2   lead {lead_str}",
       [0.85, 0.85, 0.85]),
      (f"{where}   lap {ego[4]:6.0f} / {self.track.length:.0f} m   collisions {self.collisions} {self.collision_kinds}   {disengaged}",
       [1.0, 0.4, 0.3] if (on_highway and abs(ego[5]) > 2.5) or self.collisions else [0.85, 0.85, 0.85]),
      ("traffic {name}: {n} cars   cut-ins {cut_ins}   hard brakes {hard_brakes}   brake checks {brake_checks}   merges {merges}".format(
        name=self.traffic.preset_name, n=len(self.traffic.cars), **self.traffic.stats), [1.0, 0.75, 0.3]),
      ("keys: WASD manual   +/- set speed   E engage/disengage   R reset   Q quit", [0.6, 0.6, 0.6]),
      ("pad: left stick + R2/L2 manual   d-pad set speed   select engage/disengage", [0.6, 0.6, 0.6]),
    ]
    return [{"action": "hud_text", "id": f"hud{i}", "text": t, "pos": [0.015, 0.025 + 0.03 * i], "color": c, "scale": 1.0}
            for i, (t, c) in enumerate(lines)]

  def _track_engagement(self, ego) -> None:
    active = self.sm['selfdriveState'].active
    self.op_active = active
    if self.was_active and not active:
      events = self.sm['onroadEvents']
      self.disengagements += 1
      self.user_disengaged = any(e.userDisable for e in events)
      self.last_disengage_reason = ",".join(str(e.name) for e in events if e.userDisable or e.immediateDisable or e.softDisable)
      self.disengaged_at = time.monotonic()
      print(f"[zoomengine] DISENGAGED at s {ego[4]:.0f} m, v {ego[3]:.1f} m/s, events: {[str(e.name) for e in events]}", flush=True)
    elif active:
      self.disengaged_at = None
      if not self.was_active:
        self.driver_took_over = False  # engaged again (not the moments before our cancel lands)
    self.was_active = active

  def disengaged_for(self) -> float:
    """seconds since openpilot disengaged on its own, 0 while engaged or after a user disengagement"""
    if self.disengaged_at is None or self.user_disengaged or self.driver_took_over:
      return 0.0
    return time.monotonic() - self.disengaged_at

  def _colliding(self, ego) -> str | None:
    """who is at fault if ego is touching a car: 'ego' (ego rear-ended it), 'npc' (it rear-ended ego) or 'cut-in'"""
    for car in self.traffic.cars:
      if car.opposing:
        continue
      ds = (car.s - ego[4] + self.track.length / 2) % self.track.length - self.track.length / 2
      if abs(ds) < car.half_length + 2.3 and abs(car.d - ego[5]) < 1.85:
        kind = "cut-in" if car.lc_t >= 0 else ("ego" if ds > 0 else "npc")
        if not self.was_colliding:
          rel = f"{car.cid} v {car.v:.1f} m/s, gap {ds:+.1f} m, lateral {car.d - ego[5]:+.1f} m"
          print(f"[zoomengine] COLLISION ({kind}) at s {ego[4]:.0f} m: ego v {ego[3]:.1f} m/s, {rel}", flush=True)
        return kind
    return None
