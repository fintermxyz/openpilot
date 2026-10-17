#!/usr/bin/env python3
"""Junction director for the crossroads map: builds the junction in ZoomEngine, runs the UK traffic signals (and their
fault), starts synchronized trials for the openpilot cars (crossroads_car.py, one bridge + one openpilot each) and judges
every trial: who crossed, who stopped, who collided with whom. It also broadcasts a "world" message (the signal aspects
shown and every car's pose, no fault flag) to every car at 20 Hz: the cooperative feed the L3 research layer reads."""
import argparse
import itertools
import json
import os
import random
import socket
import time

from openpilot.tools.sim.bridge.zoomengine.crossroads import CAR_HALF, ROUTES, CrossroadsMap, SignalController, footprint, lamp_commands, overlap
from openpilot.tools.sim.bridge.zoomengine.crossroads_car import CAR_PORT_BASE, DIRECTOR_PORT
from openpilot.tools.sim.bridge.zoomengine.crossroads_scenarios import SCENARIOS, verdict
from openpilot.tools.sim.bridge.zoomengine.zoom_api import ZoomAPI

HZ = 20
STALE = 1.0  # s without a pose: the car's bridge isn't running
ENGAGE_WAIT = 60.0  # s to wait for every openpilot to be engaged before a trial starts anyway
WORLD_MAX = 4000  # bytes: a world message must fit the cars' 4096 byte receive buffer
ACCEL_TAU = 0.2  # s, low-pass on the finite-difference accel of each car
BOX_STOP_V = 0.2  # m/s inside the junction box counts as a standstill there
DEADLOCK_V = 0.1  # m/s: every car slower than this ...
DEADLOCK_S = 20.0  # ... for this long before all have crossed is a deadlock
LINE_STOP_V, LINE_STOP_D = 0.5, 6.0  # m/s, m: a standstill this close before its stop line is a stop there (front bumper)
L3_KEYS = ("mode", "jx_state", "t_stop", "token")  # the l3 fields the feed must carry if the whole dict doesn't fit


class Director:
  def __init__(self, map_path: str, cars: int, fault_at: float | None, trial_len: float, port: int = 9090, seed: int | None = None,
               trials: int = 0, out: str = "/tmp/crossroads_results.json", l3_mask: str | None = None, scenario: str = "stuck_green"):
    self.cmap = CrossroadsMap(map_path)
    self.cars = cars
    self.sc = SCENARIOS[scenario]
    self.fault_at_override = fault_at  # None: the scenario draws it per trial
    self.fault_at = -1.0  # this trial's
    self.plan: dict = {}
    self.trial_len = trial_len
    self.seed = seed
    self.trials = trials  # 0: run trials until killed
    self.out = out
    self.l3_mask = l3_mask  # informational: which cars run L3 (the launcher decides), stored in the results
    if seed is not None:
      random.seed(seed)
    self.api = ZoomAPI(port=port)
    self.signals = SignalController()
    self.signals.fault_mode = self.sc.fault or "stuck_green"
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
    self.seq = 0
    self.accel: dict[int, tuple[float, float, float, int]] = {}  # car -> (t, v, filtered a, trial) of its last pose
    self.still_since: float | None = None
    self._world_trimmed = False

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
    self.still_since = None
    self.plan = self.sc.plan(self.trial, self.seed, self.cars)
    if self.fault_at_override is not None:
      self.plan["fault_at"] = self.fault_at_override
    self.fault_at = self.plan["fault_at"] if self.sc.fault is not None else -1.0
    self.outcome = {"trial": self.trial, "scenario": self.sc.name, "plan": self.plan, "fault_at": self.fault_at, "fault_t": None,
                    "collisions": [], "cars": {}, "seed": self.seed, "l3_mask": self.l3_mask, "deadlock": False, "deadlock_at": None}
    for car in range(self.cars):
      spawn_s = self.cmap.routes[ROUTES[car]].spawn_s + self.plan["spawn_ahead"][car]
      self.sock.sendto(json.dumps({"reset": self.trial, "spawn_s": spawn_s}).encode(), ("127.0.0.1", CAR_PORT_BASE + car))
    fault = f", signals fail ({self.sc.fault}) at {self.fault_at:.1f} s" if self.fault_at >= 0 else ""
    print(f"[director] trial {self.trial} started: {self.sc.title}{fault}", flush=True)

  def trial_over(self, t: float) -> bool:
    live = [p for p in self.poses.values() if p.get("trial") == self.trial]
    return t > self.trial_len or (len(live) == self.cars and all(p["parked"] for p in live))

  def judge(self, t: float) -> None:
    live = {c: p for c, p in self.poses.items() if p.get("trial") == self.trial and time.monotonic() - p["t"] < STALE}
    for c, p in live.items():
      r = self.cmap.routes[p["route"]]
      o = self.outcome["cars"].setdefault(c, {"route": p["route"], "crossed": False, "entered_at": None, "min_v_before_line": 99.0,
                                              "engaged_at_entry": None, "entry_aspect": None, "red_entry": False, "l3_mode_at_entry": None,
                                              "box_stop": False, "box_stop_at": None, "box_stop_em": False, "crossed_at": None,
                                              "stopped_at": None})
      l3 = p.get("l3") if isinstance(p.get("l3"), dict) else {}
      if p["s"] < r.stop_line_s:
        o["min_v_before_line"] = min(o["min_v_before_line"], round(p["v"], 2))
      front = p["s"] + CAR_HALF[1]
      if o["stopped_at"] is None and p["v"] < LINE_STOP_V and 0.0 <= r.stop_line_s - front < LINE_STOP_D:
        o["stopped_at"] = round(t, 2)  # first standstill at its line: its arrival for the give-way order
      if o["entered_at"] is None and p["s"] >= r.junction_s[0]:
        o["entered_at"] = round(t, 1)
        o["engaged_at_entry"] = p["engaged"]
        o["entry_aspect"] = self.shown_phases.get(r.phase)
        # UK: red and red+amber both mean stop. During the fault the heads are wrong, so it's not a red-light entry
        o["red_entry"] = o["entry_aspect"] in ("red", "red_amber") and not self.signals.fault
        o["l3_mode_at_entry"] = l3.get("mode")
      if r.junction_s[0] <= p["s"] <= r.junction_s[1] and p["v"] < BOX_STOP_V:
        if not o["box_stop"]:
          o["box_stop_at"] = round(t, 1)
        o["box_stop"] = True
        o["box_stop_em"] = o["box_stop_em"] or bool(l3.get("em")) or l3.get("mode") == "EM"
      if p["s"] >= r.junction_s[1]:
        o["crossed"] = True
      if o["crossed_at"] is None and p["s"] - CAR_HALF[1] >= r.junction_s[1]:
        o["crossed_at"] = round(t, 2)  # the whole car is through the box
      if isinstance(p.get("l3_stats"), dict):
        o["l3_stats"] = p["l3_stats"]  # the arbiter's per-trial counters (invariant violations, stale / FAILSAFE ticks)
    # deadlock: nobody moving for DEADLOCK_S while someone still has to cross
    all_crossed = len(self.outcome["cars"]) == self.cars and all(o["crossed"] for o in self.outcome["cars"].values())
    if live and not all_crossed and all(p["v"] < DEADLOCK_V for p in live.values()):
      if self.still_since is None:
        self.still_since = t
      if t - self.still_since > DEADLOCK_S and not self.outcome["deadlock"]:
        self.outcome["deadlock"] = True
        self.outcome["deadlock_at"] = round(self.still_since, 1)
        print(f"[director] DEADLOCK trial {self.trial}: nobody moving since t {self.still_since:.1f} s", flush=True)
    else:
      self.still_since = None
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
    times = [c["crossed_at"] for c in o["cars"].values()]
    all_through = len(times) == self.cars and None not in times
    o["t_clear"] = max(times) if all_through else None  # M1 exit 3: every car through within 60 s
    o["timeout"] = not all_through  # the trial ended (trial_len) with a car not yet through the box
    self.results.append(o)
    crossed = sum(c["crossed"] for c in o["cars"].values())
    stopped = [c for c, v in o["cars"].items() if v["stopped_for_lights"]]
    o["valid"] = all(v["engaged_at_entry"] in (True, None) for v in o["cars"].values()) and len(o["cars"]) == self.cars
    judged = {f"op{c}": {"route": v["route"], "phase": self.cmap.routes[v["route"]].phase, "stopped_at": v["stopped_at"],
                         "entered_at": v["entered_at"], "crossed": v["crossed"], "box_stop": v["box_stop"] and not v["box_stop_em"]}
              for c, v in o["cars"].items()}
    o["verdict"] = verdict(self.sc, judged, o["fault_t"], len(o["collisions"]), o["deadlock"])
    if len(o["cars"]) < self.cars:
      o["verdict"] = {"passed": False, "failures": [f"only {len(o['cars'])}/{self.cars} cars reported"]}
    extra = [f"{k}: {v}" for k, v in (("red entries", [c for c, v in o["cars"].items() if v["red_entry"]]),
                                       ("box stops", [c for c, v in o["cars"].items() if v["box_stop"]])) if v]
    extra += ["DEADLOCK"] if o["deadlock"] else []
    extra += [f"TIMEOUT (not through after {self.trial_len:.0f} s)"] if o["timeout"] else [f"cleared in {o['t_clear']:.1f} s"]
    print(f"[director] trial {self.trial} done: {crossed}/{self.cars} crossed, {len(o['collisions'])} collisions, stopped: {stopped}"
          + "".join(f", {e}" for e in extra), flush=True)
    v = o["verdict"]
    print(f"[director] trial {self.trial} {self.sc.name}: {'PASS' if v['passed'] else 'FAIL'}" + "".join(f"\n  - {f}" for f in v["failures"])
          + ("" if o["valid"] else "  (not every openpilot was engaged: invalid)"), flush=True)
    if os.path.dirname(self.out):
      os.makedirs(os.path.dirname(self.out), exist_ok=True)
    with open(self.out, "w") as f:
      json.dump(self.results, f, indent=1)

  def done(self) -> bool:
    return self.trials > 0 and len(self.results) >= self.trials

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
          self.broadcast_world()
          time.sleep(period)
          continue
        self._wait_engaged()
        self.start_trial()
      t = now - self.trial_t0
      if 0 <= self.fault_at <= t and not self.signals.fault:
        self.signals.fault = True
        self.outcome["fault_t"] = round(t, 2)
      self._show_signals(t)
      self.broadcast_world()
      self.judge(t)
      self._hud(t)
      if self.trial_over(t):
        self.finish_trial()
        if self.done():
          print(f"[director] {len(self.results)} trials done, results in {self.out}", flush=True)
          return
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
      self.broadcast_world()
      time.sleep(0.2)
    print(f"[director] not every openpilot engaged after {ENGAGE_WAIT:.0f} s, starting anyway", flush=True)

  def _recv(self) -> None:
    while True:
      try:
        p = json.loads(self.sock.recv(4096))
      except (BlockingIOError, ValueError):
        return
      if not isinstance(p, dict) or "car" not in p:
        continue
      now = time.monotonic()
      if "t" in p:
        p["t_sent"] = p["t"]  # the car's own send time (same clock on this machine): feed latency is measurable
      p["t"] = now
      p["a"] = self._accel(p)
      self.poses[p["car"]] = p

  def _accel(self, p: dict) -> float:
    """finite difference of v between the car's poses, low-passed; restarts on a new trial or a time step backwards"""
    t, v, trial = p.get("t_sent", p["t"]), float(p["v"]), p.get("trial")
    prev = self.accel.get(p["car"])
    a = 0.0
    if prev is not None and prev[3] == trial and t > prev[0]:
      dt = t - prev[0]
      a = prev[2] + dt / (ACCEL_TAU + dt) * ((v - prev[1]) / dt - prev[2])
    elif prev is not None and prev[3] == trial and t == prev[0]:
      return prev[2]
    self.accel[p["car"]] = (t, v, a, trial)
    return a

  # ---- cooperative feed ----
  def world_msg(self) -> dict:
    """what every car gets at 20 Hz: the aspects shown (never the fault flag) and this trial's cars"""
    cars = []
    for c in sorted(self.poses):
      p = self.poses[c]
      if p.get("trial") != self.trial:
        continue  # not reset into this trial yet: its pose is from the last one
      cars.append({"car": c, "route": p["route"], "x": round(p["x"], 3), "z": round(p["z"], 3), "psi": round(p["psi"], 4), "v": round(p["v"], 3),
                   "a": round(p.get("a", 0.0), 3), "s": round(p["s"], 3), "d": round(p["d"], 3), "half_w": CAR_HALF[0], "half_l": CAR_HALF[1],
                   "t": p.get("t_sent", p["t"]), "l3": p.get("l3") if isinstance(p.get("l3"), dict) else None})
    return {"type": "world", "trial": self.trial, "seq": self.seq, "t": time.monotonic(), "spat": dict(self.shown_phases), "cars": cars}

  def encode_world(self) -> bytes:
    msg = self.world_msg()
    data = json.dumps(msg, separators=(",", ":")).encode()
    if len(data) > WORLD_MAX:  # big l3 dicts: keep only what the junction protocol needs
      for car in msg["cars"]:
        if car["l3"] is not None:
          car["l3"] = {k: car["l3"].get(k) for k in L3_KEYS}
      data = json.dumps(msg, separators=(",", ":")).encode()
      if not self._world_trimmed:
        print(f"[director] world message over {WORLD_MAX} bytes: relaying only l3 {L3_KEYS}", flush=True)
        self._world_trimmed = True
    return data

  def broadcast_world(self) -> None:
    if not self.shown_phases:
      return
    data = self.encode_world()
    self.seq += 1
    for car in range(self.cars):
      try:
        self.sock.sendto(data, ("127.0.0.1", CAR_PORT_BASE + car))
      except OSError:
        pass  # that car's bridge isn't up yet

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
    state = {"stuck_green": "FAULT: green both ways", "dark": "FAULT: lights out (power cut)"}[self.signals.fault_mode] if self.signals.fault \
      else "normal"
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
      ((f"collisions this trial {len(self.outcome.get('collisions', []))}   all trials {total} in {len(self.results)}   scenario {self.sc.name}: " +
        f"{sum(r['verdict']['passed'] for r in self.results)}/{len(self.results)} passed"), [1.0, 0.75, 0.3]),
    ]
    self.api.batch([{"action": "hud_text", "id": f"xr{i}", "text": s, "pos": [0.27, 0.015 + 0.03 * i], "color": c, "scale": 1.0}
                    for i, (s, c) in enumerate(lines)])


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--map", required=True, help="ZoomEngine assets/maps/crossroads.built.json")
  ap.add_argument("--cars", type=int, default=len(ROUTES))
  ap.add_argument("--scenario", default="stuck_green", choices=sorted(SCENARIOS), help="what fails, how the cars arrive, what's judged")
  ap.add_argument("--fault-at", type=float, default=None, help="seconds into each trial the signals fail (<0: never; default: the scenario's)")
  ap.add_argument("--trial-len", type=float, default=None, help="seconds per trial (default 45, or 90 with --l3-mask: L3 traffic takes " +
                  "~40-60 s to clear a failed junction under the unmarked rule)")
  ap.add_argument("--seed", type=int, default=None, help="seeds python random (the director has no randomness yet), stored in every trial's results")
  ap.add_argument("--trials", type=int, default=0, help="exit after this many finished trials (0: unlimited)")
  ap.add_argument("--out", default="/tmp/crossroads_results.json", help="results JSON, rewritten after every trial")
  ap.add_argument("--l3-mask", default=None, help="informational: comma list of the cars running L3, stored in the results")
  args = ap.parse_args()
  if args.trial_len is None:
    args.trial_len = 90.0 if args.l3_mask else 45.0
  Director(args.map, args.cars, args.fault_at, args.trial_len, seed=args.seed, trials=args.trials, out=args.out, l3_mask=args.l3_mask,
           scenario=args.scenario).run()


if __name__ == "__main__":
  main()
