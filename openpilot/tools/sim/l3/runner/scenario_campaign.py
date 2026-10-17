#!/usr/bin/env python3
"""A crossroads scenario (bridge/zoomengine/crossroads_scenarios.py) in the kinematic harness, judged with the same verdict
as the engine's junction director:  L3=1 python -m openpilot.tools.sim.l3.runner.scenario_campaign uk_lights_out -k 200"""
import argparse
import json
from multiprocessing import Pool

from openpilot.tools.sim.bridge.zoomengine.crossroads import ROUTES
from openpilot.tools.sim.bridge.zoomengine.crossroads_scenarios import SCENARIOS, verdict
from openpilot.tools.sim.l3.sim_kinematic import crossroads_map, run_episode


def scenario_episode(name: str, trial: int, seed: int = 0, l3: bool = True, n_cars: int = 4, latency: float = 0.05) -> dict:
  sc = SCENARIOS[name]
  plan = sc.plan(trial, seed, n_cars)
  cmap = crossroads_map()

  def sig_fn(t, signals):
    if sc.fault is not None and t >= plan["fault_at"]:
      signals.fault, signals.fault_mode = True, sc.fault
    return signals.aspects(t)

  spawn = {i: (cmap.routes[ROUTES[i % len(ROUTES)]].spawn_s + plan["spawn_ahead"][i], cmap.speed_limit, 0.0) for i in range(n_cars)}
  res = run_episode(seed * 100003 + trial, fault="none", l3=l3, n_cars=n_cars, latency=latency, sig_fn=sig_fn, spawn=spawn)
  cars = {c: {"route": v["route"], "phase": cmap.routes[v["route"]].phase, "stopped_at": v.get("t_stop_line"), "entered_at": v.get("t_enter"),
              "crossed": v["t_cross"] is not None, "box_stop": v["box_stop"] and not v["box_stop_em"]} for c, v in res["cars"].items()}
  v = verdict(sc, cars, plan["fault_at"] if sc.fault else None, res["n_collisions"], res["deadlock"])
  return {"scenario": name, "trial": trial, "seed": seed, "l3": l3, "plan": plan, **v, "n_collisions": res["n_collisions"],
          "min_pet": res["min_pet"], "t_clear": res["t_clear"], "cars": cars}


def _run(kw: dict) -> dict:
  return scenario_episode(**kw)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("scenario", choices=sorted(SCENARIOS))
  ap.add_argument("-k", "--trials", type=int, default=100)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--no-l3", action="store_true")
  ap.add_argument("--workers", type=int, default=8)
  ap.add_argument("--out", default=None)
  args = ap.parse_args()
  jobs = [{"name": args.scenario, "trial": k, "seed": args.seed, "l3": not args.no_l3} for k in range(args.trials)]
  with Pool(args.workers) as pool:
    res = pool.map(_run, jobs)
  passed = sum(r["passed"] for r in res)
  fails: dict[str, int] = {}
  for r in res:
    for f in r["failures"]:
      key = f.split(":")[0]
      fails[key] = fails.get(key, 0) + 1
  clear = sorted(r["t_clear"] for r in res if r["t_clear"] is not None)
  print(f"{args.scenario} ({SCENARIOS[args.scenario].title}), L3 {'off' if args.no_l3 else 'on'}: {passed}/{len(res)} trials passed")
  print(f"  failures by check: {fails or 'none'}")
  print(f"  collisions {sum(r['n_collisions'] for r in res)}   median time to clear {clear[len(clear) // 2] if clear else '--'} s")
  if args.out:
    with open(args.out, "w") as f:
      json.dump(res, f, indent=1)


if __name__ == "__main__":
  main()
