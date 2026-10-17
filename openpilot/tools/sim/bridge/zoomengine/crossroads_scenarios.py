"""Crossroads trial scenarios: what goes wrong in a trial (signal fault and when), how the cars arrive, and which rules the
trial is judged against. The junction director (engine) and the L3 kinematic harness use the same definitions and verdict."""
import random
from dataclasses import dataclass

# UK signals: conflicting phases. Opposite approaches of a straight-across crossroads never conflict
TIE_S = 0.3  # s: stops closer than this are simultaneous (either may go first)
PAIR_S = 2.0  # s: opposite cars entering this close together crossed together


@dataclass(frozen=True)
class Scenario:
  name: str
  title: str
  rule: str                              # the rule the trial tests, with its source
  fault: str | None                      # SignalController fault mode from fault_at: 'stuck_green' | 'dark' | None
  fault_at: tuple[float, float]          # s into the trial, uniform in [lo, hi] per trial (seeded)
  spawn_ahead: tuple[float, float]       # m each car spawns ahead of its route's spawn, uniform per car per trial: staggered arrivals
  checks: tuple[str, ...]

  def plan(self, trial: int, seed: int | None, cars: int) -> dict:
    """this trial's draw, deterministic in (scenario, seed, trial)"""
    rng = random.Random(f"{self.name}:{seed}:{trial}")
    return {"fault_at": round(rng.uniform(*self.fault_at), 2), "spawn_ahead": [round(rng.uniform(*self.spawn_ahead), 1) for _ in range(cars)]}


CHECKS = {
  "no_collision": "no two cars touch",
  "all_crossed": "every car is through the junction before the trial ends",
  "no_box_stop": "nobody comes to a stop inside the junction (only an emergency stop may)",
  "stop_before_entry": ("every car that reaches the junction after the failure stops at its line before entering (L3 policy: " +
                        "stricter than rule 176's 'proceed with caution')"),
  # arrival order is the L3 give-way protocol's promise; the Highway Code itself only asks for caution
  "arrival_order": "of two cars on conflicting roads that both stopped, the one that stopped first enters first",
  "no_deadlock": "the junction never jams with nobody moving",
}

SCENARIOS = {
  "stuck_green": Scenario(
    "stuck_green", "Signals stuck on green both ways",
    "Highway Code rule 176: proceed with caution; the conflicting greens give nobody priority", "stuck_green", (6.0, 6.0), (0.0, 0.0),
    ("no_collision", "all_crossed", "no_box_stop", "no_deadlock")),
  "uk_lights_out": Scenario(
    "uk_lights_out", "Traffic lights not working (power cut), staggered arrivals",
    ("Highway Code rule 176: if the traffic lights are not working, treat the situation as you would an unmarked junction and " +
     "proceed with caution (at an unmarked crossroads nobody has priority)"),
    "dark", (3.0, 10.0), (0.0, 45.0),
    ("no_collision", "all_crossed", "no_box_stop", "stop_before_entry", "arrival_order", "no_deadlock")),
}


def verdict(sc: Scenario, cars: dict, fault_t: float | None, collisions: int, deadlock: bool) -> dict:
  """cars: id -> {route, phase, stopped_at (first standstill at its line, s), entered_at (front into the junction, s), crossed,
  box_stop (not an emergency stop)}. fault_t: when the signals failed (None: they didn't). Returns {passed, failures: [..]}"""
  failures = []
  for check in sc.checks:
    if check == "no_collision" and collisions:
      failures.append(f"no_collision: {collisions} collision(s)")
    elif check == "all_crossed":
      left = [c for c, v in cars.items() if not v["crossed"]]
      if left:
        failures.append(f"all_crossed: {left} not through")
    elif check == "no_box_stop":
      stopped = [c for c, v in cars.items() if v["box_stop"]]
      if stopped:
        failures.append(f"no_box_stop: {stopped} stopped in the junction")
    elif check == "no_deadlock" and deadlock:
      failures.append("no_deadlock: the junction jammed")
    elif check == "stop_before_entry" and fault_t is not None:
      bad = [c for c, v in cars.items() if v["entered_at"] is not None and v["entered_at"] > fault_t + 1.0 and
             (v["stopped_at"] is None or v["stopped_at"] > v["entered_at"] or v["stopped_at"] < fault_t)]
      if bad:
        failures.append(f"stop_before_entry: {bad} entered the failed junction without stopping")
    elif check == "arrival_order" and fault_t is not None:
      def turn(c: str) -> float:
        # opposite approaches don't conflict and may cross together: a car going with its opposite partner goes on the
        # partner's turn if that came first
        v = cars[c]
        return min(o["stopped_at"] for o in cars.values() if o["phase"] == v["phase"] and o["stopped_at"] is not None and
                   o["entered_at"] is not None and v["entered_at"] is not None and abs(o["entered_at"] - v["entered_at"]) < PAIR_S)
      for a, va in cars.items():
        for b, vb in cars.items():
          if va["phase"] == vb["phase"] or None in (va["stopped_at"], vb["stopped_at"], va["entered_at"]):
            continue
          if va["stopped_at"] < fault_t or vb["stopped_at"] < fault_t:
            continue  # stopped for a red before the failure: not a give-way at an unmarked junction
          if vb["entered_at"] is not None and va["stopped_at"] + TIE_S < turn(b) and vb["entered_at"] < va["entered_at"]:
            failures.append(f"arrival_order: {b} (stopped {vb['stopped_at']:.1f} s) went before {a} (stopped {va['stopped_at']:.1f} s)")
  return {"passed": not failures, "failures": failures}
