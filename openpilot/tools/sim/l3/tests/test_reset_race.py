import os
from openpilot.tools.sim.bridge.zoomengine.crossroads import CrossroadsMap
from openpilot.tools.sim.l3.host import L3Host
from openpilot.tools.sim.l3.msgs import Actor, EgoState, WorldMsg

MAP = os.path.join(os.environ.get("ZOOMENGINE_DIR", os.path.expanduser("~/Documents/zoom/ZoomEngine-main")), "assets/maps/crossroads.built.json")


def test_stale_end_of_route_pose_after_reset_does_not_clear_the_junction():
  """engine trial 1 crashed: right after a director reset the control thread ticked L3 with the previous trial's end-of-route
  pose (the physics car had not teleported back yet), the junction FSM went CLEARED and never constrained the new trial"""
  cm = CrossroadsMap(MAP)
  t = [1000.0]
  h = L3Host("op1", "S_N", cm, clock=lambda: t[0])
  r = cm.routes["S_N"]
  h.reset(1)
  s, v = 475.0, 0.0  # stale: where the car parked at the end of trial 0
  states = set()
  for k in range(25 * 40):
    t[0] += 0.04
    x, z, psi = r.track.pose(s, 0)
    other = Actor("op3", "W_E", *cm.routes["W_E"].track.pose(min(s, 200.0), 0)[:3], v=v, s=min(s, 200.0), t_obs=t[0])
    h.feed(WorldMsg(1, k, t[0], {"NS": "green", "EW": "green"}, [other]))
    h.tick(EgoState(t[0], "op1", "S_N", x, z, psi, v, 0.0, s, 0.0, True))
    states.add(h.last_jx.state)
    if k == 2:
      s, v = 15.0, 12.0  # the reset lands
    out = h.arbiter.apply(1.0, t[0], 0.04, v=v)
    a = -6.0 if out.brake_hold else out.accel
    v = max(0.0, v + a * 0.04)
    s += v * 0.04
  assert "STOPPED" in states
  assert s < r.stop_line_s  # it stopped at the failed (green both ways) junction instead of driving through
