#!/usr/bin/env python3
"""Junction director for the crossroads map: builds the junction in ZoomEngine, runs the UK traffic signals (and their
fault), starts synchronized trials for the openpilot cars (crossroads_car.py, one bridge + one openpilot each) and judges
every trial: who crossed, who stopped, who collided with whom."""
import argparse
import itertools
import json
import socket
import time

from openpilot.tools.sim.bridge.zoomengine.crossroads import ROUTES, CrossroadsMap, SignalController, footprint, lamp_commands, overlap
from openpilot.tools.sim.bridge.zoomengine.crossroads_car import CAR_PORT_BASE, DIRECTOR_PORT
from openpilot.tools.sim.bridge.zoomengine.zoom_api import ZoomAPI

HZ = 20
STALE = 1.0  # s without a pose: the car's bridge isn't running
ENGAGE_WAIT = 60.0  # s to wait for every openpilot to be engaged before a trial starts anyway


class Director:
  def __init__(self, map_path: str, cars: int, fault_at: float, trial_len: float, port: int = 9090):
    self.cmap = CrossroadsMap(map_path)
    self.cars = cars
    self.fault_at = fault_at
    self.trial_len = trial_len
    self.api = ZoomAPI(port=port)
    self.signals = SignalController()
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.bind(("127.0.0.1", DIRECTOR_PORT))
    self.sock.setblocking(False)
    self.poses: dict[int, dict] = {}
    self.trial = -1
    self.trial_t0 = 0.0
    self.shown: dict[str, str] = {}
    self.shown_phases: dict[str, str] = {}
    self.results: list[dict] = []
    self.outcome: dict = {}

  def build(self) -> None:
    self.api.wait_ready()
    self.api.send({"action": "clear"})
    self.api.batch(self.cmap.scene_commands(), chunk=200)
    # overview of the junction from a corner
    self.api.send({"action": "camera", "position": [38.0, 26.0, -38.0], "target": [0.0, 0.0, 0.0]})
    print(f"---- crossroads built: {len(self.cmap.data['statics'])} road sections, {len(self.cmap.data['signals'])} signal heads ----",
          flush=True)

  # ---- trials ----
  def start_trial(self) -> None:
    self.trial += 1
    self.trial_t0 = time.monotonic()
    self.signals.fault = False
    self.outcome = {"trial": self.trial, "fault_at": self.fault_at, "collisions": [], "cars": {}}
    for car in range(self.cars):
      self.sock.sendto(json.dumps({"reset": self.trial}).encode(), ("127.0.0.1", CAR_PORT_BASE + car))
    print(f"[director] trial {self.trial} started", flush=True)

  def trial_over(self, t: float) -> bool:
    live = [p for p in self.poses.values() if p.get("trial") == self.trial]
    return t > self.trial_len or (len(live) == self.cars and all(p["parked"] for p in live))

  def judge(self, t: float) -> None:
    live = {c: p for c, p in self.poses.items() if p.get("trial") == self.trial and time.monotonic() - p["t"] < STALE}
    for c, p in live.items():
      r = self.cmap.routes[p["route"]]
      o = self.outcome["cars"].setdefault(c, {"route": p["route"], "crossed": False, "entered_at": None, "min_v_before_line": 99.0,
                                              "engaged_at_entry": None})
      if p["s"] < r.stop_line_s:
        o["min_v_before_line"] = min(o["min_v_before_line"], round(p["v"], 2))
      if o["entered_at"] is None and p["s"] >= r.junction_s[0]:
        o["entered_at"] = round(t, 1)
        o["engaged_at_entry"] = p["engaged"]
      if p["s"] >= r.junction_s[1]:
        o["crossed"] = True
    for a, b in itertools.combinations(sorted(live), 2):
      pa, pb = live[a], live[b]
      if overlap(footprint(pa["x"], pa["z"], pa["psi"]), footprint(pb["x"], pb["z"], pb["psi"])):
        pair = [a, b]
        if pair not in [c["cars"] for c in self.outcome["collisions"]]:
          hit = {"cars": pair, "t": round(t, 1), "speeds": [round(pa["v"], 1), round(pb["v"], 1)],
                 "routes": [pa["route"], pb["route"]], "lights": dict(self.shown_phases)}
          self.outcome["collisions"].append(hit)
          who = f"car {a + 1} ({pa['route']}, {pa['v']:.1f} m/s) and car {b + 1} ({pb['route']}, {pb['v']:.1f} m/s)"
          print(f"[director] COLLISION trial {self.trial}: {who} at t {t:.1f} s, lights {self.shown_phases}", flush=True)

  def finish_trial(self) -> None:
    o = self.outcome
    for c in o["cars"].values():
      c["stopped_for_lights"] = c["min_v_before_line"] < 0.5
    self.results.append(o)
    crossed = sum(c["crossed"] for c in o["cars"].values())
    stopped = [c for c, v in o["cars"].items() if v["stopped_for_lights"]]
    o["valid"] = all(v["engaged_at_entry"] in (True, None) for v in o["cars"].values()) and len(o["cars"]) == self.cars
    print(f"[director] trial {self.trial} done: {crossed}/{self.cars} crossed, {len(o['collisions'])} collisions, stopped: {stopped}", flush=True)
    with open("/tmp/crossroads_results.json", "w") as f:
      json.dump(self.results, f, indent=1)

  # ---- loop ----
  def run(self) -> None:
    self.build()
    period = 1.0 / HZ
    waiting_since = time.monotonic()
    while True:
      now = time.monotonic()
      self._recv()
      ready = [c for c, p in self.poses.items() if now - p["t"] < STALE]
      if self.trial < 0:
        # wait until every car's bridge reports in, then a few seconds more for openpilot to engage
        if len(ready) < self.cars:
          if now - waiting_since > 5.0:
            print(f"[director] waiting for cars: {len(ready)}/{self.cars} reporting", flush=True)
            waiting_since = now
          self._show_signals(0.0)
          time.sleep(period)
          continue
        self._wait_engaged()
        self.start_trial()
      t = now - self.trial_t0
      if 0 <= self.fault_at <= t:
        self.signals.fault = True
      self._show_signals(t)
      self.judge(t)
      self._hud(t)
      if self.trial_over(t):
        self.finish_trial()
        self._wait_engaged()
        self.start_trial()
      time.sleep(max(0.0, period - (time.monotonic() - now)))

  def _wait_engaged(self) -> None:
    """a trial only measures openpilot if every car is engaged when it starts"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < ENGAGE_WAIT:
      self._recv()
      engaged = [c for c, p in self.poses.items() if p["engaged"] and time.monotonic() - p["t"] < STALE]
      if len(engaged) >= self.cars:
        return
      self._show_signals(0.0)
      time.sleep(0.2)
    print(f"[director] not every openpilot engaged after {ENGAGE_WAIT:.0f} s, starting anyway", flush=True)

  def _recv(self) -> None:
    while True:
      try:
        p = json.loads(self.sock.recv(4096))
      except (BlockingIOError, ValueError):
        return
      p["t"] = time.monotonic()
      self.poses[p["car"]] = p

  def _show_signals(self, t: float) -> None:
    aspects = self.signals.aspects(t)
    cmds = []
    for sig in self.cmap.data["signals"]:
      a = aspects[sig["phase"]]
      if self.shown.get(sig["id"]) != a:
        cmds += lamp_commands(sig["id"], a)
        self.shown[sig["id"]] = a
    self.shown_phases = aspects
    if cmds:
      self.api.batch(cmds)

  def _hud(self, t: float) -> None:
    ph = self.shown_phases
    state = "FAULT: green both ways" if self.signals.fault else "normal"
    collided = {c for hit in self.outcome.get("collisions", []) for c in hit["cars"]}
    cars = []
    for c in range(self.cars):
      p = self.poses.get(c)
      if p is None:
        cars.append(f"car {c + 1}: --")
        continue
      tag = " CRASHED" if c in collided else (" crossed" if self.outcome["cars"].get(c, {}).get("crossed") else "")
      cars.append(f"car {c + 1} {p['route'].replace('_', '>')} {p['v'] * 2.237:4.1f} mph{' op' if p['engaged'] else ' --'}{tag}")
    total = sum(len(r["collisions"]) for r in self.results)
    lines = [
      (f"crossroads (UK)   trial {self.trial}   t {t:5.1f} s   signals {state}   N-S {ph.get('NS', '?')}   E-W {ph.get('EW', '?')}",
       [1.0, 0.4, 0.3] if self.signals.fault else [1.0, 1.0, 1.0]),
      ("   ".join(cars), [0.85, 0.85, 0.85]),
      (f"collisions this trial {len(self.outcome.get('collisions', []))}   all trials {total} in {len(self.results)}", [1.0, 0.75, 0.3]),
    ]
    self.api.batch([{"action": "hud_text", "id": f"xr{i}", "text": s, "pos": [0.27, 0.015 + 0.03 * i], "color": c, "scale": 1.0}
                    for i, (s, c) in enumerate(lines)])


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--map", required=True, help="ZoomEngine assets/maps/crossroads.built.json")
  ap.add_argument("--cars", type=int, default=len(ROUTES))
  ap.add_argument("--fault-at", type=float, default=6.0, help="seconds into each trial the signals stick on green both ways (<0: never)")
  ap.add_argument("--trial-len", type=float, default=45.0)
  args = ap.parse_args()
  Director(args.map, args.cars, args.fault_at, args.trial_len).run()


if __name__ == "__main__":
  main()
