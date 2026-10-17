import math
from dataclasses import dataclass, field

import numpy as np

from openpilot.tools.sim.bridge.zoomengine.scene import CAR_COLORS, EDGE, KINDS, OPPOSING_D, car_commands, deg
from openpilot.tools.sim.bridge.zoomengine.track import LANE_WIDTH, Track

LANE_OFFSETS = (-LANE_WIDTH, 0.0, LANE_WIDTH)
EGO_HALF_LENGTH = 2.3
FAR = 1e9


@dataclass(frozen=True)
class Preset:
  n_same: int                 # cars on the ego's carriageway
  n_opposing: int
  aggressive_frac: float      # share of drivers with short headways that cut in with small gaps
  lane_change_rate: float     # spontaneous lane changes per car per second
  phantom_brake_rate: float   # random hard braking events per car per second
  brake_check_period: float   # seconds between hard brakes by the car directly ahead of ego (0 = off)
  cut_in_rate: float          # per aggressive car per second, when it's alongside ego
  closure: bool               # right lane closed with cones, forcing merges


PRESETS = {
  "light": Preset(16, 14, 0.0, 1 / 12, 0.0, 0.0, 0.0, False),
  "heavy": Preset(48, 30, 0.3, 1 / 7, 1 / 150, 45.0, 1 / 8, True),
  "chaos": Preset(80, 44, 0.55, 1 / 4, 1 / 70, 25.0, 1 / 3, True),
}

# right lane closure on the second long straight, cones taper over the first 100 m
CLOSURE_LANE = 2
CLOSURE_TAPER_S = 3200.0
CLOSURE_START_S = 3300.0
CLOSURE_END_S = 3650.0


@dataclass
class Driver:
  headway: float
  s0: float
  a_max: float
  b: float
  gap_ahead: float    # minimum gaps accepted for a lane change
  gap_behind: float
  lc_time: float      # duration of a lane change


CALM = Driver(1.5, 4.0, 1.4, 2.5, 35.0, 25.0, 4.5)


@dataclass
class Car:
  cid: str
  kind: str
  opposing: bool
  s: float
  lane: int
  v: float
  v_des: float
  driver: Driver = field(default_factory=lambda: CALM)
  aggressive: bool = False
  d: float = 0.0
  d_from: float = 0.0
  lc_t: float = -1.0  # time into a lane change, <0 when not changing lanes
  cooldown: float = 0.0
  brake_until_v: float = -1.0  # hard braking event: brake until below this speed
  brake_hold: float = 0.0

  @property
  def half_length(self) -> float:
    return KINDS[self.kind][2]

  def lane_d(self, lane: int) -> float:
    return OPPOSING_D - LANE_OFFSETS[lane] if self.opposing else LANE_OFFSETS[lane]


def idm(v: float, v0: float, gap: float, dv: float, drv: Driver) -> float:
  s_star = drv.s0 + max(0.0, v * drv.headway + v * dv / (2 * math.sqrt(drv.a_max * drv.b)))
  return drv.a_max * (1 - (v / max(v0, 0.1)) ** 4 - (s_star / max(gap, 0.1)) ** 2)


class Traffic:
  """NPC traffic on both carriageways: IDM car following, lane changes, cut-ins, hard braking and a lane closure.
  Ego is part of the traffic, NPCs follow it and cut in front of it."""
  def __init__(self, track: Track, rng: np.random.Generator, ego_s: float, preset: str = "heavy"):
    self.track = track
    self.rng = rng
    self.preset = PRESETS[preset]
    self.preset_name = preset
    self.cars: list[Car] = []
    self.t = 0.0
    self.next_brake_check = self.preset.brake_check_period or FAR
    self.stats = {"cut_ins": 0, "hard_brakes": 0, "brake_checks": 0, "merges": 0}
    kinds = ["sedan"] * 6 + ["suv"] * 3 + ["truck"]
    p = self.preset

    # a lead car right in front of ego so openpilot has something to follow immediately
    self._add("npc0", "sedan", False, ego_s + 45.0, 1, 20.0, 23.0, aggressive=False)
    placed = [(ego_s, 1), (ego_s + 45.0, 1)]
    for i in range(1, p.n_same):
      for _ in range(50):
        s = float(rng.uniform(0, track.length))
        lane = int(rng.integers(0, 3))
        if self._closed(s, lane) or (self._ds(s, ego_s) > -40 and self._ds(s, ego_s) < 60):
          continue
        if all(lane != pl or abs(self._ds(s, ps)) > 25 for ps, pl in placed):
          break
      placed.append((s, lane))
      kind = str(rng.choice(kinds))
      aggressive = kind != "truck" and rng.random() < p.aggressive_frac
      v_des = 21.0 if kind == "truck" else float(rng.uniform(22.0, 30.0) - lane * 1.5 + (4.0 if aggressive else 0.0))
      self._add(f"npc{i}", kind, False, s, lane, v_des * 0.85, v_des, aggressive)
    spacing = track.length / p.n_opposing
    for i in range(p.n_opposing):
      kind = str(rng.choice(kinds))
      self._add(f"opp{i}", kind, True, i * spacing + rng.uniform(-0.3, 0.3) * spacing, int(rng.integers(0, 3)), 25.0,
                float(rng.uniform(23.0, 31.0)), rng.random() < p.aggressive_frac)

  def _driver(self, aggressive: bool) -> Driver:
    if not aggressive:
      return Driver(float(self.rng.uniform(1.3, 1.8)), 4.0, float(self.rng.uniform(1.2, 1.6)), 2.5, 35.0, 25.0, 4.5)
    return Driver(float(self.rng.uniform(0.6, 0.95)), 2.0, float(self.rng.uniform(2.0, 2.8)), 3.5,
                  float(self.rng.uniform(7.0, 12.0)), float(self.rng.uniform(5.0, 9.0)), float(self.rng.uniform(2.0, 2.8)))

  def _add(self, cid, kind, opposing, s, lane, v, v_des, aggressive):
    car = Car(cid, kind, opposing, s % self.track.length, lane, v, v_des, self._driver(aggressive), aggressive)
    car.d = car.d_from = car.lane_d(lane)
    self.cars.append(car)

  def _ds(self, a: float, b: float) -> float:
    """signed shortest arc length from b to a"""
    return (a - b + self.track.length / 2) % self.track.length - self.track.length / 2

  def _closed(self, s: float, lane: int) -> bool:
    return self.preset.closure and lane == CLOSURE_LANE and CLOSURE_TAPER_S - 20 <= s % self.track.length <= CLOSURE_END_S

  # ---- engine objects ----
  def create_commands(self) -> list[dict]:
    cmds = []
    for i, car in enumerate(self.cars):
      x, z, psi = self.pose(car)
      color = [0.92, 0.92, 0.9] if car.kind == "truck" else CAR_COLORS[i % len(CAR_COLORS)]
      cmds += car_commands(car.cid, car.kind, color, x, z, psi)
    if self.preset.closure:
      cmds += self._closure_commands()
    return cmds

  def _closure_commands(self) -> list[dict]:
    cmds = []
    s, i = CLOSURE_TAPER_S, 0
    while s <= CLOSURE_END_S:
      u = min(1.0, (s - CLOSURE_TAPER_S) / (CLOSURE_START_S - CLOSURE_TAPER_S))
      d = EDGE + 0.6 + u * (LANE_WIDTH * 0.5 + 0.3 - EDGE - 0.6)  # ends just right of the middle lane's line
      x, z, _ = self.track.pose(s, d)
      cmds.append({"action": "create", "type": "cone", "id": f"cone{i}", "position": [x, 0.38, z], "radius": 0.22, "height": 0.75,
                   "segments": 12, "color": [1.0, 0.35, 0.0], "emissive": [0.6, 0.15, 0.0], "roughness": 0.6})
      s += 6.0 if u < 1.0 else 12.0
      i += 1
    for k, s in enumerate((CLOSURE_TAPER_S - 250.0, CLOSURE_TAPER_S - 20.0)):
      x, z, psi = self.track.pose(s, EDGE + 1.6)
      cmds.append({"action": "create", "type": "cube", "id": f"closure_sign{k}", "position": [x, 1.2, z],
                   "rotation": [0, deg(psi + math.pi / 4), 45], "scale": [0.6, 0.6, 0.03], "color": [1.0, 0.5, 0.0],
                   "emissive": [0.5, 0.25, 0.0]})
      cmds.append({"action": "create", "type": "cylinder", "id": f"closure_post{k}", "position": [x, 0.4, z], "radius": 0.04,
                   "height": 0.8, "color": [0.4, 0.4, 0.4]})
    x, z, psi = self.track.pose(CLOSURE_START_S + 30.0, LANE_WIDTH + 0.2)
    cmds.append({"action": "create", "type": "cube", "id": "closure_truck", "position": [x, 1.3, z], "rotation": [0, deg(psi), 0],
                 "scale": [1.2, 1.3, 3.5], "color": [1.0, 0.55, 0.05], "emissive": [0.3, 0.12, 0.0], "roughness": 0.5})
    return cmds

  def pose(self, car: Car) -> tuple[float, float, float]:
    x, z, psi = self.track.pose(car.s, car.d)
    if car.opposing:
      psi += math.pi
    # yaw into the new lane while changing lanes
    if car.lc_t >= 0:
      T = car.driver.lc_time
      lat_v = (car.lane_d(car.lane) - car.d_from) * math.pi / (2 * T) * math.sin(math.pi * min(car.lc_t, T) / T)
      psi -= math.atan2(lat_v, max(car.v, 1.0)) * (-1 if car.opposing else 1)
    return x, z, psi

  def engine_commands(self, ego_s: float, frame: int) -> list[dict]:
    """poses for all cars, far away cars only every 5th frame to keep the API traffic down"""
    cmds = []
    for car in self.cars:
      if frame % 5 and abs(self._ds(car.s, ego_s)) > 400.0:
        continue
      x, z, psi = self.pose(car)
      cmds.append({"action": "move", "id": car.cid, "position": [round(x, 3), KINDS[car.kind][3], round(z, 3)]})
      cmds.append({"action": "rotate", "id": car.cid, "rotation": [0, deg(psi), 0]})
    return cmds

  # ---- simulation ----
  def _snapshot(self) -> None:
    """vectorized car state, refreshed once per step for the leader searches"""
    self._idx = {id(c): i for i, c in enumerate(self.cars)}
    self._s = np.array([c.s for c in self.cars])
    self._d = np.array([c.d for c in self.cars])
    self._target_d = np.array([c.lane_d(c.lane) for c in self.cars])  # a lane changing car already occupies its target lane
    self._v = np.array([c.v for c in self.cars])
    self._hl = np.array([c.half_length for c in self.cars])
    self._opp = np.array([c.opposing for c in self.cars])

  def _leader(self, car: Car, lane_d: float, ego, behind: bool = False) -> tuple[float, float]:
    """(bumper gap, speed) of the nearest car ahead (or behind) in the lane at offset lane_d"""
    best_gap, best_v = FAR, 0.0
    mask = (self._opp == car.opposing) & ((np.abs(self._d - lane_d) <= 2.4) | (np.abs(self._target_d - lane_d) < 1.0))
    mask[self._idx[id(car)]] = False
    if mask.any():
      ds = (car.s - self._s[mask]) if behind != car.opposing else (self._s[mask] - car.s)
      gaps = ds % self.track.length - car.half_length - self._hl[mask]
      j = int(np.argmin(gaps))
      best_gap, best_v = float(gaps[j]), float(self._v[mask][j])
    if not car.opposing:
      if abs(ego[1] - lane_d) < 2.4:
        gap = (self._ds(car.s, ego[0]) if behind else self._ds(ego[0], car.s)) % self.track.length
        if gap - car.half_length - EGO_HALF_LENGTH < best_gap:
          best_gap, best_v = gap - car.half_length - EGO_HALF_LENGTH, ego[2]
      # the closure acts like a stopped car at the start of the taper
      if not behind and self.preset.closure and abs(car.lane_d(CLOSURE_LANE) - lane_d) < 1.0:
        gap = self._ds(CLOSURE_TAPER_S, car.s)
        if 0 < gap - car.half_length < best_gap:
          best_gap, best_v = gap - car.half_length, 0.0
    return best_gap, best_v

  def _start_lane_change(self, car: Car, target: int) -> None:
    car.lane, car.d_from, car.lc_t, car.cooldown = target, car.d, 0.0, 6.0 if car.aggressive else 15.0

  def _try_lane_change(self, car: Car, target: int, ego, scale: float = 1.0) -> bool:
    if not 0 <= target < 3 or (not car.opposing and self._closed(car.s + 60.0, target)):
      return False
    td = car.lane_d(target)
    ahead, v_ahead = self._leader(car, td, ego)
    behind, v_behind = self._leader(car, td, ego, behind=True)
    # leave room to match the new leader's speed, and don't make the new follower brake hard
    if ahead > car.driver.gap_ahead * scale + max(0.0, car.v - v_ahead) * 2.0 and \
       behind > car.driver.gap_behind * scale + max(0.0, v_behind - car.v) * 1.5:
      self._start_lane_change(car, target)
      return True
    return False

  def step(self, dt: float, ego: tuple[float, float, float]) -> None:
    """ego: (s, d, speed) on the track"""
    p = self.preset
    self.t += dt
    ego_lane = min(range(3), key=lambda i: abs(LANE_OFFSETS[i] - ego[1]))

    # brake check: the car directly ahead of ego slams on the brakes
    if self.t >= self.next_brake_check:
      self.next_brake_check = self.t + p.brake_check_period * float(self.rng.uniform(0.7, 1.3))
      lead = min((c for c in self.cars if not c.opposing and abs(c.d - ego[1]) < 1.8 and 5.0 < self._ds(c.s, ego[0]) < 70.0),
                 key=lambda c: self._ds(c.s, ego[0]), default=None)
      if lead is not None and lead.brake_until_v < 0:
        lead.brake_until_v = lead.v * float(self.rng.uniform(0.25, 0.55))
        self.stats["brake_checks"] += 1

    self._snapshot()
    for car in self.cars:
      gap, v_lead = self._leader(car, car.d, ego)
      a = idm(car.v, car.v_des, gap, car.v - v_lead, car.driver)
      if car.lc_t >= 0:  # mid lane change: also follow the leader in the target lane
        gap2, v_lead2 = self._leader(car, car.lane_d(car.lane), ego)
        a = min(a, idm(car.v, car.v_des, gap2, car.v - v_lead2, car.driver))

      if car.brake_until_v >= 0:
        if car.v > car.brake_until_v:
          a = min(a, -float(self.rng.uniform(5.0, 7.0)))
        else:
          car.brake_hold += dt
          a = min(a, 0.0)
          if car.brake_hold > 2.0:
            car.brake_until_v, car.brake_hold = -1.0, 0.0
      elif p.phantom_brake_rate and self.rng.random() < p.phantom_brake_rate * dt and car.v > 15.0:
        car.brake_until_v = car.v * float(self.rng.uniform(0.3, 0.7))
        self.stats["hard_brakes"] += 1

      car.v = max(0.0, car.v + max(-9.0, a) * dt)
      ds = car.v * dt / (1.0 + self.track.curvature(car.s) * car.d)
      car.s = (car.s + (-ds if car.opposing else ds)) % self.track.length

      car.cooldown -= dt
      if car.lc_t >= 0:
        car.lc_t += dt
        u = min(1.0, car.lc_t / car.driver.lc_time)
        car.d = car.d_from + (car.lane_d(car.lane) - car.d_from) * (1 - math.cos(math.pi * u)) / 2
        if u >= 1.0:
          car.lc_t = -1.0
        continue

      # forced merge out of the closed lane: zipper, accepting ever smaller gaps closer to the cones
      if not car.opposing and p.closure and car.lane == CLOSURE_LANE:
        to_taper = self._ds(CLOSURE_TAPER_S, car.s)
        if 0 < to_taper < 350.0 and self._try_lane_change(car, CLOSURE_LANE - 1, ego, scale=max(0.25, to_taper / 350.0)):
          self.stats["merges"] += 1
          continue

      if car.cooldown > 0 or car.kind == "truck":
        continue
      # cut-in: an aggressive car just ahead of ego in the next lane dives into ego's lane
      if not car.opposing and car.aggressive and p.cut_in_rate and abs(car.lane - ego_lane) == 1 \
         and 3.0 < self._ds(car.s, ego[0]) - car.half_length - EGO_HALF_LENGTH < 25.0 and self.rng.random() < p.cut_in_rate * dt:
        if self._try_lane_change(car, ego_lane, ego, scale=0.5):
          self.stats["cut_ins"] += 1
          continue
      # overtaking a slower leader, otherwise the occasional random lane change
      wants = gap < 60.0 and v_lead < car.v_des - 3.0
      if wants or self.rng.random() < p.lane_change_rate * dt:
        self._try_lane_change(car, car.lane + int(self.rng.choice([-1, 1])), ego)
