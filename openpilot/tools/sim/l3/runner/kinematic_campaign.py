#!/usr/bin/env python3
"""Kinematic crossroads campaign: K seeded episodes per configuration (L3 on/off x stuck_green/none), on a process pool.
SIMULATION ONLY, run with L3=1:

  L3=1 python -m openpilot.tools.sim.l3.runner.kinematic_campaign -k 200
"""
import argparse
import json
import math
import statistics
import time
from multiprocessing import Pool

from openpilot.tools.sim.l3.sim_kinematic import run_episode_kw


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  if n == 0:
    return 0.0, 1.0
  p = k / n
  den = 1 + z * z / n
  c = (p + z * z / (2 * n)) / den
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return max(0.0, c - h), min(1.0, c + h)


def bound(k: int, n: int) -> str:
  """0 events: the rule-of-three 95 % upper bound 3/N; otherwise a Wilson 95 % interval"""
  if n == 0:
    return "-"
  if k == 0:
    return f"0/{n} (< {3 / n:.1%})"
  lo, hi = wilson(k, n)
  return f"{k}/{n} ({k / n:.0%}, 95% {lo:.0%}-{hi:.0%})"


def summarize(res: list[dict]) -> dict:
  n = len(res)
  pets = [p["pet"] for r in res for p in r["pets"]]
  clears = [r["t_clear"] for r in res if r["t_clear"] is not None]
  return {
    "episodes": n,
    "collision_eps": sum(r["n_collisions"] > 0 for r in res),
    "collisions": sum(r["n_collisions"] for r in res),
    "min_pet": min(pets) if pets else None,
    "median_pet": statistics.median(pets) if pets else None,
    "pet_lt_1_5": sum(p < 1.5 for p in pets),
    "crossings": len(pets),
    "box_stop_eps": sum(bool(r["box_stops"]) for r in res),
    "box_stops": sum(len(r["box_stops"]) for r in res),
    "red_entry_eps": sum(bool(r["red_entries"]) for r in res),
    "red_entries": sum(len(r["red_entries"]) for r in res),
    "deadlocks": sum(r["deadlock"] for r in res),
    "timeouts": sum(r["timeout"] for r in res),
    "em_eps": sum(bool(r["em"]) for r in res),
    "box_stops_em": sum(len(r["box_stops_em"]) for r in res),
    "failsafe_eps": sum(bool(r["failsafe"]) for r in res),
    "mean_t_clear": statistics.mean(clears) if clears else None,
    "cleared_60s": sum(t <= 60.0 for t in clears),
    "arb_violations": sum(r.get("arb_violations", 0) for r in res),
  }


def fmt(v, spec: str = ".2f") -> str:
  return "-" if v is None else format(v, spec)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("-k", "--episodes", type=int, default=100)
  ap.add_argument("--seed0", type=int, default=0)
  ap.add_argument("--workers", type=int, default=None)
  ap.add_argument("--cars", type=int, default=4)
  ap.add_argument("--latency", type=float, default=0.05)
  ap.add_argument("--fault-at", type=float, default=6.0)
  ap.add_argument("--max-t", type=float, default=90.0)
  ap.add_argument("--stagger", action="store_true", help="spawn each car 0-12 s late: arrival orders vary widely")
  ap.add_argument("--l3", choices=("both", "on", "off"), default="both")
  ap.add_argument("--fault", choices=("both", "stuck_green", "none"), default="both")
  ap.add_argument("--json", help="write every episode result here")
  args = ap.parse_args()

  l3s = {"both": (False, True), "on": (True,), "off": (False,)}[args.l3]
  faults = {"both": ("stuck_green", "none"), "stuck_green": ("stuck_green",), "none": ("none",)}[args.fault]
  configs = [(l3, f) for f in faults for l3 in l3s]
  jobs, keys = [], []
  for l3, f in configs:
    kw = {"fault": f, "fault_at": args.fault_at, "l3": l3, "n_cars": args.cars, "latency": args.latency, "max_t": args.max_t, "stagger": args.stagger}
    for seed in range(args.seed0, args.seed0 + args.episodes):
      jobs.append(dict(kw, seed=seed))
      keys.append((l3, f))

  t0 = time.monotonic()
  with Pool(args.workers) as pool:
    results = pool.map(run_episode_kw, jobs, chunksize=max(1, len(jobs) // 64))
  wall = time.monotonic() - t0

  print(f"kinematic crossroads campaign: {args.episodes} seeds from {args.seed0}, {args.cars} cars, latency {args.latency * 1000:.0f} ms, " +
        f"fault_at {args.fault_at} s, stagger {args.stagger}   ({len(jobs)} episodes in {wall:.1f} s)")
  print("SIMULATOR DESIGN VERIFICATION ONLY: oracle feed, point-mass cars, openpilot-like controller. No road-safety claim.")
  hdr = (f"{'L3':>3} {'fault':<11} {'collision episodes':<30} {'collisions':>10} {'minPET':>7} {'medPET':>7} {'PET<1.5':>8} " +
        f"{'box stops':>9} {'(em)':>4} {'red entries':>11} {'deadlocks':>9} {'timeouts':>8} {'EM eps':>6} {'failsafe':>8} {'t_clear':>8} {'<=60s':>6} " +
         f"{'a>op':>5}")
  print(hdr)
  print("-" * len(hdr))
  summary = {}
  for l3, f in configs:
    res = [r for r, k in zip(results, keys, strict=True) if k == (l3, f)]
    s = summarize(res)
    summary[f"{'l3' if l3 else 'op'}_{f}"] = s
    n = s["episodes"]
    print(f"{'on' if l3 else 'off':>3} {f:<11} {bound(s['collision_eps'], n):<30} {s['collisions']:>10} {fmt(s['min_pet']):>7} {fmt(s['median_pet']):>7} " +
          f"{s['pet_lt_1_5']:>8} {s['box_stops']:>9} {s['box_stops_em']:>4} {s['red_entries']:>11} " +
          f"{s['deadlocks']:>9} {s['timeouts']:>8} {s['em_eps']:>6} {s['failsafe_eps']:>8} {fmt(s['mean_t_clear'], '.1f'):>8} {s['cleared_60s']:>6} " +
          f"{s['arb_violations']:>5}")
  if args.json:
    with open(args.json, "w") as fh:
      json.dump({"args": vars(args), "summary": summary, "episodes": results}, fh, indent=1)
    print(f"episodes written to {args.json}")


if __name__ == "__main__":
  main()
