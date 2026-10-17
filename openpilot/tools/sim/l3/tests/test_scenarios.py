from openpilot.tools.sim.bridge.zoomengine.crossroads_scenarios import SCENARIOS, verdict
from openpilot.tools.sim.l3.runner.scenario_campaign import scenario_episode

SC = SCENARIOS["uk_lights_out"]


def car(phase, stopped, entered, route="N_S"):
  return {"route": route, "phase": phase, "stopped_at": stopped, "entered_at": entered, "crossed": True, "box_stop": False}


def test_queue_jump_is_caught():
  cars = {"op0": car("NS", 10.0, 20.0), "op2": car("EW", 12.0, 14.0, "E_W")}
  v = verdict(SC, cars, 5.0, 0, False)
  assert not v["passed"] and v["failures"][0].startswith("arrival_order")


def test_opposite_partner_may_go_with_the_first_car():
  # op0 and op2 tie (op0 goes), op1 opposite op0 stopped just after op2 but crosses together with op0
  cars = {"op0": car("NS", 17.27, 22.8), "op1": car("NS", 17.45, 22.9, "S_N"), "op2": car("EW", 17.14, 29.6, "E_W"),
          "op3": car("EW", 19.8, 29.8, "W_E")}
  assert verdict(SC, cars, 3.1, 0, False)["passed"]


def test_entering_without_stopping_is_caught():
  cars = {"op0": car("NS", None, 20.0)}
  assert verdict(SC, cars, 5.0, 0, False)["failures"][0].startswith("stop_before_entry")


def test_plan_is_deterministic():
  assert SC.plan(3, 1, 4) == SC.plan(3, 1, 4) and SC.plan(3, 1, 4) != SC.plan(4, 1, 4)


def test_lights_out_closed_loop():
  for trial in range(12):
    assert scenario_episode("uk_lights_out", trial, seed=7)["passed"], trial
  assert not any(scenario_episode("uk_lights_out", trial, seed=7, l3=False)["passed"] for trial in range(4))
