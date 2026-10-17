"""Hand-engineered ("artisan") SAE Level 3 research layer for the ZoomEngine simulator. SIMULATION ONLY.

It wraps openpilot (the Level 2 driver) with rule-based, unit-tested modules: a signal monitor, a junction manager with
right-of-way and failed-signal rules, and a Simplex arbiter at the simulated actuators that can only make the car more
conservative (accel = min(openpilot, cap)). Nothing here may run on a real car: importing it outside the simulator raises.
"""
import os


class NotInSimulator(RuntimeError):
  pass


def assert_sim_only() -> None:
  """the L3 layer actuates the simulated car only: refuse to load anywhere else"""
  from openpilot.common.hardware import PC
  problems = []
  if os.environ.get("L3") != "1":
    problems.append("L3 != 1")
  if not PC:
    problems.append("running on comma hardware")
  if os.path.exists("/data/params"):
    problems.append("/data/params exists (a device)")
  if problems:
    raise NotInSimulator("tools.sim.l3 is simulator-only: " + ", ".join(problems))


assert_sim_only()
