#!/usr/bin/env python3
"""Town director: builds the road-kit town in ZoomEngine (roads, signals, buildings, street furniture, car parks) and runs
everything that isn't an openpilot car: the four junctions' UK signals (with a pedestrian stage), NPC traffic on the lane
graph (following, signals, slowing for turns, stopping for pedestrians and openpilot cars), and pedestrians (walking the
footways, crossing at the junctions in the pedestrian stage and now and then mid-block when it's clear). The openpilot
cars (town_car.py, one bridge + one openpilot each) report their pose and what they hit over UDP; the director knocks
NPCs / pedestrians about accordingly, counts collisions and laps, and respawns an openpilot car that has been stuck."""
import argparse
import itertools
import json
import math
import socket
import time

import numpy as np

from openpilot.tools.sim.bridge.zoomengine.crossroads import footprint, lamp_commands, overlap
from openpilot.tools.sim.bridge.zoomengine.scene import KINDS, car_commands, car_rotation, deg
from openpilot.tools.sim.bridge.zoomengine.town import (CROSSING_D, FOOTWAY_D, PIP_H, PIP_ROW_Y, JunctionSignals, Npc, Ped, Scenery,
                                                       TownMap, idm, pedestrian_commands, town_layout)
from openpilot.tools.sim.bridge.zoomengine.zoom_api import ZoomAPI

HZ = 20
DIRECTOR_PORT = 9290
CAR_PORT_BASE = 9300
STALE = 1.0
STUCK_RESET = 45.0       # s an openpilot car may stand still (not at a red light) before it's put back at its spawn
CAR_LEN = 4.6
LOOKAHEAD = 60.0
WRECK_CLEAR = 25.0       # s before a wrecked NPC is towed (respawned elsewhere)
DOWN_TIME = 20.0         # s a knocked-down pedestrian stays down
NPC_COLORS = ([0.75, 0.75, 0.78], [0.08, 0.08, 0.09], [0.55, 0.05, 0.05], [0.9, 0.9, 0.92], [0.1, 0.2, 0.45], [0.35, 0.36, 0.38],
              [0.5, 0.45, 0.35], [0.1, 0.3, 0.2], [0.85, 0.65, 0.1])


class Director:
  def __init__(self, map_path: str, cars: int, npcs: int, peds: int, seed: int = 1, port: int = 9090):
    self.tmap = TownMap(map_path)
    self.g = self.tmap.graph
    self.cars = cars
    self.rng = np.random.default_rng(seed)
    self.api = ZoomAPI(port=port)
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.bind(("127.0.0.1", DIRECTOR_PORT))
    self.sock.setblocking(False)
    self.poses: dict[int, dict] = {}
    self.t0 = time.monotonic()
    junctions = sorted(self.tmap.junctions)
    # UK signals show red+amber before green; US ones go straight from red to green
    red_amber = 2.0 if self.tmap.left_hand else 0.0
    self.signals = {j: JunctionSignals(offset=8.0 * i, red_amber=red_amber) for i, j in enumerate(junctions)}
    self.sig_junction = {s["id"]: s["approach"][:3] for s in self.tmap.data["signals"]}
    self.shown: dict[str, str] = {}
    self.aspects: dict[str, dict] = {}
    self.ped_stage: dict[str, tuple[bool, float]] = {}
    # op car bookkeeping
    self.stuck = dict.fromkeys(range(cars), 0.0)
    self.last_s = dict.fromkeys(range(cars))
    self.laps = dict.fromkeys(range(cars), 0)
    self.resets = dict.fromkeys(range(cars), 0)
    self.reset_seq = 0
    self.op_contacts: set = set()
    self.stats = {"op_collisions": 0, "npc_hit": 0, "ped_hit": 0, "towed": 0, "crossings": 0, "jaywalks": 0}
    # streets for pedestrians: footway polylines extended into the junctions to the crossing points
    self.streets = []
    for st in self.g.streets():
      if st["elevated"]:   # the expressway: no footway down at ground level
        continue
      c, left = st["centre"], st["left"]
      t0 = c[0] - c[1]
      t0 /= np.linalg.norm(t0)
      t1 = c[-1] - c[-2]
      t1 /= np.linalg.norm(t1)
      ext = 16.0 - CROSSING_D
      cc = np.concatenate([[c[0] + t0 * ext], c, [c[-1] + t1 * ext]])
      ll = np.concatenate([[left[0]], left, [left[-1]]])
      s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cc, axis=0), axis=1))])
      self.streets.append({"c": cc, "left": ll, "s": s, "len": float(s[-1]), "ends": (st["from"], st["to"])})
    self.npcs: list[Npc] = []
    self.peds: list[Ped] = []
    self._spawn_npcs(npcs)
    self._spawn_peds(peds)

  # ---- setup ----
  def _op_spawns(self) -> list[np.ndarray]:
    out = []
    for name in self.tmap.car_routes[:self.cars]:
      r = self.tmap.routes[name]
      x, z, _ = r.track.pose(r.spawn_s, 0.0)
      out.append(np.array([x, z]))
    return out

  def _spawn_npcs(self, n: int) -> None:
    linear = [lid for lid, ln in self.g.lanes.items() if ln.junction is None and ln.length > 20.0]
    keep_off = self._op_spawns()
    tries = 0
    while len(self.npcs) < n and tries < 5000:
      tries += 1
      lid = linear[self.rng.integers(len(linear))]
      s = float(self.rng.uniform(3.0, self.g.lanes[lid].length - 3.0))
      x, z, psi = self.g.pose(lid, s)
      p = np.array([x, z])
      if any(np.linalg.norm(p - q) < 50.0 for q in keep_off) or any(math.hypot(x - o.x, z - o.z) < 22.0 for o in self.npcs):
        continue
      v_des = float(self.rng.uniform(0.8, 1.0)) * self.tmap.speed_limit
      npc = Npc(f"npc{len(self.npcs)}", "sedan" if self.rng.random() < 0.7 else "suv", lid, s, v_des * 0.8, v_des, x=x, z=z, psi=psi)
      self._extend_plan(npc)
      self.npcs.append(npc)

  def _spawn_peds(self, n: int) -> None:
    for i in range(n):
      k = int(self.rng.integers(len(self.streets)))
      st = self.streets[k]
      p = Ped(f"ped{i}", k, float(self.rng.choice([-1.0, 1.0])), float(self.rng.uniform(8.0, st["len"] - 8.0)),
              float(self.rng.choice([-1.0, 1.0])), float(self.rng.uniform(1.1, 1.5)), phase=float(self.rng.uniform(0, 6.28)))
      p.h = float(self.rng.uniform(0.92, 1.06))
      self._ped_pose(p)
      self.peds.append(p)

  def build(self) -> None:
    self.api.wait_ready()
    self.api.send({"action": "clear"})
    scenery = Scenery(self.tmap, self.rng)
    cmds = self.tmap.scene_commands() + scenery.cmds
    for npc in self.npcs:
      color = NPC_COLORS[int(self.rng.integers(len(NPC_COLORS)))]
      y, pitch = self.g.height(npc.lane, npc.s)
      cmds += car_commands(npc.id, npc.kind, color, npc.x, npc.z, npc.psi, y0=KINDS[npc.kind][3] + y, rotation=car_rotation(npc.psi, pitch))
    for p in self.peds:
      cmds += pedestrian_commands(p.id, p.x, p.z, p.psi, self.rng, p.h)
    self.api.batch(cmds, chunk=300)
    # high over the south edge of the town, looking north across it
    cx = float(np.mean([j[0] for j in self.tmap.junctions.values()]))
    cz = float(np.mean([j[1] for j in self.tmap.junctions.values()]))
    k = float(self.tmap.data["ground"]["size"][1]) / 780.0   # the town's ground is ~780 m across: scale the view to the map
    self.api.send({"action": "camera", "position": [cx, 300.0 * k, cz - 390.0 * k], "target": [cx, 0.0, cz + 30.0 * k]})
    print(f"---- town built: {len(self.tmap.data['statics'])} road sections, {len(self.tmap.data['signals'])} signal heads, " +
          f"{len(scenery.cmds)} scenery objects, {len(self.npcs)} NPC cars, {len(self.peds)} pedestrians ----", flush=True)

  # ---- NPC traffic ----
  def _extend_plan(self, npc: Npc) -> None:
    ahead = self.g.lanes[npc.lane].length - npc.s + sum(self.g.lanes[lid].length for lid in npc.plan)
    while ahead < LOOKAHEAD + 30.0:
      last = npc.plan[-1] if npc.plan else npc.lane
      nxt = self.g.lanes[last].next
      # UK: no offside (right) turns across oncoming traffic; mostly straight on
      ok = [n for n in nxt if self.g.lanes[n].movement != "offside"] or nxt
      w = np.array([0.65 if self.g.lanes[n].movement in (None, "through") else 0.35 for n in ok])
      choice = ok[int(self.rng.choice(len(ok), p=w / w.sum()))]
      npc.plan.append(choice)
      ahead += self.g.lanes[choice].length

  def _path_ahead(self, npc: Npc) -> tuple[np.ndarray, np.ndarray, list]:
    """points every ~1 m along the NPC's lanes ahead, their distance from it, and (lane, distance to its start)"""
    ln = self.g.lanes[npc.lane]
    i = int(np.searchsorted(ln.s, npc.s))
    pts = [np.array([npc.x, npc.z])[None], ln.pts[i:]]
    starts = [(npc.lane, -npc.s)]
    d0 = ln.length - npc.s
    for lid in npc.plan:
      if d0 > LOOKAHEAD:
        break
      starts.append((lid, d0))
      pts.append(self.g.lanes[lid].pts[1:])
      d0 += self.g.lanes[lid].length
    P = np.concatenate(pts)
    D = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    m = D <= LOOKAHEAD
    return P[m], D[m], starts

  def _step_npcs(self, dt: float) -> list[dict]:
    cmds = []
    vehicles = [(o.x, o.z, o.v, o.psi, o.id) for o in self.npcs]
    now = time.monotonic()
    vehicles += [(p["x"], p["z"], p["v"], p["psi"], f"op{c}") for c, p in self.poses.items() if now - p["t"] < STALE]
    walkers = [(p.x, p.z) for p in self.peds if p.state in ("cross", "down")]
    for npc in self.npcs:
      if npc.wrecked > 0.0:
        npc.wrecked += dt
        npc.v = max(0.0, npc.v - 6.0 * dt)
        if npc.wrecked > WRECK_CLEAR:
          self._tow(npc)
          continue
      else:
        P, D, starts = self._path_ahead(npc)
        gap, v_lead = 1e3, 0.0
        if len(P) > 1:
          for (x, z, v, psi, oid) in vehicles:
            if oid == npc.id or abs(x - npc.x) > LOOKAHEAD + 5 or abs(z - npc.z) > LOOKAHEAD + 5:
              continue
            d2 = (P[:, 0] - x) ** 2 + (P[:, 1] - z) ** 2
            j = int(np.argmin(d2))
            if d2[j] < 1.7 ** 2 and D[j] > 0.5:
              g = D[j] - CAR_LEN
              if g < gap:
                gap, v_lead = g, max(0.0, v * math.cos(psi - npc.psi))
          for (x, z) in walkers:
            d2 = (P[:, 0] - x) ** 2 + (P[:, 1] - z) ** 2
            j = int(np.argmin(d2))
            if d2[j] < 2.2 ** 2 and D[j] - 3.5 < gap:
              gap, v_lead = D[j] - 3.5, 0.0
          # signals: stop at the line of a junction movement whose phase isn't green
          for lid, d_start in starts:
            ln = self.g.lanes[lid]
            if ln.junction is None or ln.stop_s is None:
              continue
            d_stop = d_start + ln.stop_s - CAR_LEN / 2
            if d_stop < -0.5:
              continue
            aspect = self.aspects.get(ln.junction, {}).get(ln.phase, "red")
            if aspect != "green" and not (aspect == "amber" and d_stop < npc.v ** 2 / 8.0 + 1.0) and d_stop < gap:
              gap, v_lead = max(d_stop, 0.05), 0.0
        # slow for tight lanes ahead (junction turns), comfortable 2 m/s^2
        v0 = min(npc.v_des, self.g.lanes[npc.lane].vmax)
        d0 = self.g.lanes[npc.lane].length - npc.s
        for lid in npc.plan[:3]:
          v0 = min(v0, math.sqrt(self.g.lanes[lid].vmax ** 2 + 4.0 * max(d0, 0.0)))
          d0 += self.g.lanes[lid].length
        a = idm(npc.v, v0, gap, npc.v - v_lead)
        npc.v = max(0.0, npc.v + max(-8.0, a) * dt)
      npc.s += npc.v * dt
      self._extend_plan(npc)   # keep the lanes chosen ahead (a long lane can use up the whole plan)
      while npc.s > self.g.lanes[npc.lane].length:
        npc.s -= self.g.lanes[npc.lane].length
        npc.lane = npc.plan.pop(0)
        self._extend_plan(npc)
      if npc.lat_v:
        npc.lat += npc.lat_v * dt
        npc.lat_v *= math.exp(-dt * 1.5)
        if abs(npc.lat_v) < 0.05:
          npc.lat_v = 0.0
      x, z, psi = self.g.pose(npc.lane, npc.s)
      rx, rz = -math.cos(psi), math.sin(psi)
      npc.x, npc.z, npc.psi = x + rx * npc.lat, z + rz * npc.lat, psi
      y, pitch = self.g.height(npc.lane, npc.s)   # bridges and ramps
      cmds.append({"action": "move", "id": npc.id, "position": [round(npc.x, 3), round(KINDS[npc.kind][3] + y, 3), round(npc.z, 3)], "smooth": True})
      cmds.append({"action": "rotate", "id": npc.id, "rotation": car_rotation(npc.psi, pitch), "smooth": True})
    return cmds

  def _tow(self, npc: Npc) -> None:
    """a wrecked NPC is cleared away and re-enters the town somewhere quiet"""
    linear = [lid for lid, ln in self.g.lanes.items() if ln.junction is None and ln.length > 20.0]
    now = time.monotonic()
    others = [(o.x, o.z) for o in self.npcs if o is not npc] + [(p["x"], p["z"]) for p in self.poses.values() if now - p["t"] < STALE]
    for _ in range(200):
      lid = linear[self.rng.integers(len(linear))]
      s = float(self.rng.uniform(3.0, self.g.lanes[lid].length - 3.0))
      x, z, psi = self.g.pose(lid, s)
      if all(math.hypot(x - ox, z - oz) > 40.0 for ox, oz in others):
        npc.lane, npc.s, npc.plan, npc.v, npc.wrecked, npc.lat, npc.lat_v = lid, s, [], npc.v_des * 0.6, 0.0, 0.0, 0.0
        npc.x, npc.z, npc.psi = x, z, psi
        self._extend_plan(npc)
        self.stats["towed"] += 1
        self.api.send({"action": "move", "id": npc.id, "position": [x, KINDS[npc.kind][3] + self.g.height(lid, s)[0], z]})
        return

  # ---- pedestrians ----
  def _ped_pose(self, p: Ped) -> None:
    st = self.streets[p.street]
    s = min(max(p.s, 0.0), st["len"])
    i = int(np.clip(np.searchsorted(st["s"], s) - 1, 0, len(st["s"]) - 2))
    t = (s - st["s"][i]) / max(st["s"][i + 1] - st["s"][i], 1e-6)
    c = st["c"][i] + t * (st["c"][i + 1] - st["c"][i])
    q = c + st["left"][i] * p.side * FOOTWAY_D
    d = (st["c"][i + 1] - st["c"][i]) * p.dir
    p.x, p.z, p.psi = float(q[0]), float(q[1]), math.atan2(d[0], d[1])

  def _clear_to_cross(self, x: float, z: float) -> bool:
    now = time.monotonic()
    cars = [(o.x, o.z) for o in self.npcs] + [(c["x"], c["z"]) for c in self.poses.values() if now - c["t"] < STALE]
    return all(math.hypot(x - cx, z - cz) > 45.0 for cx, cz in cars)

  def _start_crossing(self, p: Ped) -> None:
    st = self.streets[p.street]
    i = int(np.clip(np.searchsorted(st["s"], p.s), 0, len(st["s"]) - 1))
    c, left = st["c"][i], st["left"][i]
    p.cross_from = (float(c[0] + left[0] * p.side * FOOTWAY_D), float(c[1] + left[1] * p.side * FOOTWAY_D))
    p.cross_to = (float(c[0] - left[0] * p.side * FOOTWAY_D), float(c[1] - left[1] * p.side * FOOTWAY_D))
    p.state, p.cross_t = "cross", 0.0
    p.psi = math.atan2(p.cross_to[0] - p.cross_from[0], p.cross_to[1] - p.cross_from[1])

  def _step_peds(self, dt: float, frame: int) -> list[dict]:
    cmds = []
    for p in self.peds:
      st = self.streets[p.street]
      if p.state == "down":
        p.timer += dt
        if p.timer > DOWN_TIME:   # gets up and walks on
          p.state, p.timer = "walk", 0.0
          self._ped_pose(p)
        else:
          continue
      elif p.state == "wait":
        p.timer += dt
        end = 0 if p.s <= 0.5 else 1
        jn = st["ends"][end]
        ped, left = self.ped_stage.get(jn, (False, 0.0))
        if ped and left > 5.0:
          self._start_crossing(p)
          self.stats["crossings"] += 1
        elif p.timer > 40.0:
          p.dir, p.state, p.timer = -p.dir, "walk", 0.0
      elif p.state == "cross":
        span = math.dist(p.cross_from, p.cross_to)
        p.cross_t += dt * p.speed * 1.15 / max(span, 1.0)
        if p.cross_t >= 1.0:
          p.side, p.state = -p.side, "walk"
          p.dir = float(self.rng.choice([-1.0, 1.0]))
          self._ped_pose(p)
        else:
          p.x = p.cross_from[0] + (p.cross_to[0] - p.cross_from[0]) * p.cross_t
          p.z = p.cross_from[1] + (p.cross_to[1] - p.cross_from[1]) * p.cross_t
      if p.state == "walk":
        p.s += p.dir * p.speed * dt
        if p.s <= 0.0 or p.s >= st["len"]:
          p.s = min(max(p.s, 0.0), st["len"])
          if self.rng.random() < 0.6:
            p.state, p.timer = "wait", 0.0   # at a junction crossing: wait for the pedestrian stage
          else:
            p.dir = -p.dir
        elif 15.0 < p.s < st["len"] - 15.0 and self.rng.random() < 0.004 * dt * HZ / 20.0:
          self._ped_pose(p)
          if self._clear_to_cross(p.x, p.z):   # mid-block, only with nothing coming
            self._start_crossing(p)
            self.stats["jaywalks"] += 1
        if p.state == "walk":
          self._ped_pose(p)
      moving = p.state in ("walk", "cross")
      if moving:
        p.phase += dt * p.speed * 2 * math.pi / 1.3
      cmds.append({"action": "move", "id": p.id, "position": [round(p.x, 3), round(1.18 * p.h, 3), round(p.z, 3)], "smooth": True})
      cmds.append({"action": "rotate", "id": p.id, "rotation": [0, deg(p.psi), 0], "smooth": True})
      if frame % 2 == 0:   # gait: legs and arms swing about the hips / shoulders
        a = 26.0 * math.sin(p.phase) if moving else 0.0
        for side, sgn in (("l", 1.0), ("r", -1.0)):
          cmds.append({"action": "rotate", "id": f"{p.id}_hip{side}", "rotation": [round(a * sgn, 2), 0, 0], "smooth": True})
          cmds.append({"action": "rotate", "id": f"{p.id}_sh{side}", "rotation": [round(-0.8 * a * sgn, 2), 0, 0], "smooth": True})
    return cmds

  # ---- openpilot cars ----
  def _recv(self) -> list[dict]:
    cmds = []
    while True:
      try:
        p = json.loads(self.sock.recv(65536))
      except (BlockingIOError, ValueError):
        return cmds
      p["t"] = time.monotonic()
      for oid, dv in p.pop("hits", {}).items():
        cmds += self._hit(oid, dv)
      self.poses[p["car"]] = p

  def _hit(self, oid: str, dv: list[float]) -> list[dict]:
    mag = math.hypot(dv[0], dv[2])
    npc = next((o for o in self.npcs if o.id == oid), None)
    if npc is not None:
      along = dv[0] * math.sin(npc.psi) + dv[2] * math.cos(npc.psi)
      across = dv[0] * -math.cos(npc.psi) + dv[2] * math.sin(npc.psi)
      npc.v = max(0.0, npc.v + along)
      npc.lat_v += across
      if mag > 1.0 and npc.wrecked == 0.0:
        npc.wrecked = 1e-3
        self.stats["npc_hit"] += 1
        print(f"[town] openpilot car hit {oid}: dv {mag:.1f} m/s", flush=True)
      return []
    ped = next((p for p in self.peds if p.id == oid), None)
    if ped is not None and ped.state != "down" and mag > 0.3:
      ped.state, ped.timer = "down", 0.0
      self.stats["ped_hit"] += 1
      print(f"[town] PEDESTRIAN {oid} hit by an openpilot car: dv {mag:.1f} m/s", flush=True)
      ped.x += dv[0] * 0.4
      ped.z += dv[2] * 0.4
      return [{"action": "move", "id": oid, "position": [round(ped.x, 3), 0.22, round(ped.z, 3)]},
              {"action": "rotate", "id": oid, "rotation": [-90, deg(ped.psi), 0]}]
    return []

  def _watch_cars(self, dt: float) -> None:
    now = time.monotonic()
    live = {c: p for c, p in self.poses.items() if now - p["t"] < STALE}
    for c, p in live.items():
      r = self.tmap.routes[p["route"]]
      # laps: s wraps round the loop
      if self.last_s[c] is not None and self.last_s[c] - p["s"] > r.track.length / 2:
        self.laps[c] += 1
      self.last_s[c] = p["s"]
      waiting = False   # standing at a red light isn't being stuck
      for stop in self.tmap.stops.get(p["route"], []):
        d = (stop["stop_line_s"] - p["s"]) % r.track.length
        if d < 30.0 and self.aspects.get(stop["junction"], {}).get(stop["phase"]) != "green":
          waiting = True
      self.stuck[c] = self.stuck[c] + dt if (p["v"] < 0.3 and not waiting) else 0.0
      if self.stuck[c] > STUCK_RESET:
        self.reset_seq += 1
        self.sock.sendto(json.dumps({"reset": self.reset_seq}).encode(), ("127.0.0.1", CAR_PORT_BASE + c))
        self.stuck[c] = 0.0
        self.resets[c] += 1
        print(f"[town] car {c + 1} stuck for {STUCK_RESET:.0f} s: back to its spawn", flush=True)
    for a, b in itertools.combinations(sorted(live), 2):
      pa, pb = live[a], live[b]
      touching = overlap(footprint(pa["x"], pa["z"], pa["psi"]), footprint(pb["x"], pb["z"], pb["psi"]))
      if touching and (a, b) not in self.op_contacts:
        self.stats["op_collisions"] += 1
        print(f"[town] COLLISION car {a + 1} ({pa['v']:.1f} m/s) and car {b + 1} ({pb['v']:.1f} m/s)", flush=True)
        self.op_contacts.add((a, b))
      elif not touching:
        self.op_contacts.discard((a, b))

  # ---- signals / HUD ----
  def _show_signals(self, t: float) -> list[dict]:
    cmds = []
    for j, ctl in self.signals.items():
      aspects, ped, left = ctl.state(t)
      self.aspects[j] = aspects
      self.ped_stage[j] = (ped, left)
    for sig in self.tmap.data["signals"]:
      a = self.aspects[self.sig_junction[sig["id"]]][sig["phase"]]
      if self.shown.get(sig["id"]) != a:
        cmds += lamp_commands(sig["id"], a)
        self.shown[sig["id"]] = a
    return cmds

  def _hud(self, t: float) -> list[dict]:
    now = time.monotonic()
    cmds = []
    for c in range(self.cars):   # each car's status line next to its camera views
      p = self.poses.get(c)
      if p is None or now - p["t"] > STALE:
        s, col = f"car {c + 1}: --", [0.6, 0.6, 0.6]
      else:
        s = (f"car {c + 1}  {p['route'].replace('_', ' ')}  {p['v'] * 2.237:4.1f} mph  {'openpilot' if p['engaged'] else 'manual'}  " +
             f"lane {p['d']:+.1f} m  lap {self.laps[c]}")
        col = [0.5, 1.0, 0.6] if p["engaged"] else [1.0, 0.6, 0.4]
      cmds.append({"action": "hud_text", "id": f"town_car{c}", "text": s, "pos": town_layout(c)[2], "color": col, "scale": 1.15})
    moving = sum(1 for o in self.npcs if o.wrecked == 0.0 and o.v > 0.5)
    walking = sum(1 for p in self.peds if p.state == "walk")
    crossing = sum(1 for p in self.peds if p.state == "cross")
    peds_now = [j for j, (ped, _) in self.ped_stage.items() if ped]
    st = self.stats
    where = "UK, drive on the left" if self.tmap.left_hand else "US, drive on the right"
    lines = [
      (f"{self.tmap.name} ({where}, 30 mph)   t {t:6.1f} s   pedestrian stage at {', '.join(peds_now) or 'none'}   R: restart", [1.0, 1.0, 1.0]),
      (f"NPC cars {len(self.npcs)} ({moving} moving)   pedestrians {len(self.peds)} ({walking} walking, {crossing} crossing)   " +
       f"crossings {st['crossings']}  mid-block {st['jaywalks']}", [0.7, 0.85, 1.0]),
      (f"openpilot collisions: cars {st['op_collisions']}   NPCs hit {st['npc_hit']}   pedestrians hit {st['ped_hit']}   " +
       f"towed {st['towed']}   stuck resets {sum(self.resets.values())}", [1.0, 0.75, 0.3]),
    ]
    mid = PIP_ROW_Y[0] + PIP_H + 0.05     # below the top row's status lines
    return cmds + [{"action": "hud_text", "id": f"town{i}", "text": s, "pos": [0.01, mid + 0.03 * i], "color": c, "scale": 1.0}
                   for i, (s, c) in enumerate(lines)]

  # ---- loop ----
  def run(self) -> None:
    self.build()
    period = 1.0 / HZ
    frame = 0
    last = time.monotonic()
    while True:
      now = time.monotonic()
      dt, last = min(now - last, 0.2), now
      t = now - self.t0
      cmds = self._recv()
      cmds += self._show_signals(t)
      cmds += self._step_npcs(dt)
      cmds += self._step_peds(dt, frame)
      self._watch_cars(dt)
      if frame % 10 == 0:
        cmds += self._hud(t)
      if frame % (HZ * 5) == 0:
        print(f"[town] t {t:6.1f}  " + "  ".join(f"car{c + 1} lap {self.laps[c]} v {self.poses[c]['v']:.1f}" for c in sorted(self.poses))
              + f"  {self.stats}", flush=True)
      try:
        self.api.batch(cmds, chunk=400)
      except OSError:
        print("[town] ZoomEngine went away, stopping", flush=True)
        return
      frame += 1
      time.sleep(max(0.0, period - (time.monotonic() - now)))


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--map", required=True, help="ZoomEngine assets/maps/town.built.json")
  ap.add_argument("--cars", type=int, default=4)
  ap.add_argument("--npcs", type=int, default=18)
  ap.add_argument("--peds", type=int, default=30)
  ap.add_argument("--seed", type=int, default=1)
  args = ap.parse_args()
  Director(args.map, args.cars, args.npcs, args.peds, args.seed).run()


if __name__ == "__main__":
  main()
