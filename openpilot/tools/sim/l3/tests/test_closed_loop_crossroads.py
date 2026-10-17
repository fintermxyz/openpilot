"""Kinematic closed loop at the crossroads (PLAN §6 M2 sim_kinematic, pulled forward to validate M1 without the engine).
Runs in well under a minute on a process pool. SIMULATOR DESIGN VERIFICATION ONLY."""
import os
from multiprocessing import get_context

import pytest

from openpilot.tools.sim.l3.params import PARAMS
from openpilot.tools.sim.l3.sim_kinematic import run_episode_kw

pytestmark = pytest.mark.slow

SEEDS = range(30)
WORKERS = min(8, os.cpu_count() or 2)


def _run(seeds=SEEDS, **kw) -> list[dict]:
  jobs = [dict(kw, seed=s) for s in seeds]
  with get_context("spawn").Pool(WORKERS) as pool:
    return pool.map(run_episode_kw, jobs)


def _brief(r: dict) -> str:
  return (f"seed {r['seed']}: collisions {r['collisions']} box {r['box_stops']} box(em) {r['box_stops_em']} uncleared {r['uncleared']} " +
          f"red {r['red_entries']} failsafe {r['failsafe']}")


def test_baseline_reproduces_the_crashes():
  """openpilot alone with stuck greens: the harness must crash in a clear majority of seeds, or it proves nothing"""
  res = _run(l3=False, fault="stuck_green")
  crashed = sum(r["n_collisions"] > 0 for r in res)
  assert crashed >= 0.7 * len(res), f"only {crashed}/{len(res)} baseline episodes crashed"


@pytest.mark.parametrize("stagger", [False, True])
def test_l3_stuck_green(stagger):
  res = _run(l3=True, fault="stuck_green", stagger=stagger)
  bad = [r for r in res if r["n_collisions"] or r["box_stops"] or r["box_stops_em"] or r["deadlock"] or r["timeout"] or r["failsafe"]]
  assert not bad, "\n".join(_brief(r) for r in bad)
  # every conflicting crossing kept the post-encroachment time margin (M1 exit: >= pet_min - 0.5 s)
  pets = [p["pet"] for r in res for p in r["pets"]]
  assert pets and min(pets) >= PARAMS.junction.pet_min - 0.5, f"min PET {min(pets) if pets else None}"


def test_l3_normal_lights():
  res = _run(l3=True, fault="none")
  bad = [r for r in res if r["n_collisions"] or r["red_entries"] or r["box_stops"] or r["box_stops_em"] or r["timeout"] or r["failsafe"]]
  assert not bad, "\n".join(_brief(r) for r in bad)


def test_l3_amber_dilemma_zone_no_red_entry():
  """8 staggered cars, normal lights: seeds where a car on GO met amber 36 m out at 12 m/s. Stopping needed ~2.1 m/s^2 (rule 175:
  stop), but the old a_b-only check let it go on and it crossed the line 0.2 s into red"""
  res = _run(seeds=(51, 133, 211, 237), l3=True, fault="none", n_cars=8, stagger=True)
  bad = [r for r in res if r["n_collisions"] or r["red_entries"] or r["box_stops"] or r["box_stops_em"] or r["failsafe"]]
  assert not bad, "\n".join(_brief(r) for r in bad)
