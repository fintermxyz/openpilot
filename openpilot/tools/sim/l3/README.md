# L3: a hand-engineered Level 3 research layer (simulator only)

openpilot is SAE Level 2: it follows lanes and leads, with no notion of cross traffic or right of way. This package wraps it
with small, rule-based, unit-tested modules that can only make the car **more conservative**, in the ZoomEngine simulator:

| module | what it does |
|---|---|
| `signal_monitor.py` | trusts the signals only while they make sense: conflicting greens, a dark head, a stale feed or a crossing car that can no longer stop make them UNTRUSTED (latched for the trial) |
| `junction.py` | conflict zones from the map, an approach speed envelope (until it may go, the car can always stop at the line), a post-encroachment-time gap test, point of no return, never stopping in the box, and the Highway Code rule 176 protocol for failed signals: stop, then go in arrival order |
| `arbiter.py` | Simplex arbiter at the simulated actuators: `accel = min(openpilot, cap)`, brake hold, a watchdog that holds and then latches FAILSAFE |
| `host.py` | per car: the cooperative feed from the junction director (signal phase + every car), then signals -> junction -> arbiter |
| `sim_kinematic.py`, `runner/kinematic_campaign.py` | a fast closed-loop harness (point-mass cars, an openpilot-like driver that ignores lights) for campaigns of thousands of episodes |

Design, requirements and milestones: [DESIGN.md](DESIGN.md). This is milestone M1 (stop the crossroads crashes); takeover
requests, minimal-risk manoeuvres and the highway are later milestones.

**Simulator only.** Importing `tools.sim.l3` raises unless `L3=1` on a PC with no `/data/params`. It actuates only the
simulated car through the bridge (ZoomEngine `vehicle_input`), never `carControl`/CAN, and leaves openpilot, panda and
driver monitoring unchanged (`tools/check_no_core_diff.sh`). The cooperative feed is ground truth from the director (an
oracle, labelled as such): the 40 deg road camera cannot see cross traffic.

## Scenarios

`SCENARIO=` picks what fails, how the cars arrive and what each trial is judged on (`bridge/zoomengine/crossroads_scenarios.py`,
shared by the engine's director and the kinematic harness); every trial ends with PASS / FAIL and the failed checks.

| scenario | what happens | judged on |
|---|---|---|
| `stuck_green` (default) | the signals stick on green both ways 6 s into the trial | no collision, everyone through, no stop in the box, no deadlock |
| `uk_lights_out` | **Highway Code rule 176**, traffic lights not working: a power cut turns every head dark 3-10 s in (seeded per trial), cars spawn up to 45 m apart so they arrive staggered | the above, plus every car arriving after the failure stops at its line, and of two cars on conflicting roads the one that stopped first goes first (opposite cars may go together) |

``` bash
L3=1 openpilot/tools/sim/launch_crossroads.sh                  # 4 openpilots + L3 at the crossroads, signals stuck green
SCENARIO=uk_lights_out L3=1 TRIALS=6 openpilot/tools/sim/launch_crossroads.sh   # rule 176: lights out, staggered arrivals
L3=1 python -m openpilot.tools.sim.l3.runner.scenario_campaign uk_lights_out -k 500   # the same scenario, kinematic harness
L3=1 L3_MASK=0,2 openpilot/tools/sim/launch_crossroads.sh      # L3 on cars 0 and 2 only
L3=1 TRIALS=6 L3_LOG_DIR=/tmp/l3run openpilot/tools/sim/launch_crossroads.sh   # 6 trials, per-car arbiter logs
python -m pytest -q openpilot/tools/sim/l3/tests               # unit, property and closed-loop tests
L3=1 python -m openpilot.tools.sim.l3.runner.kinematic_campaign --help
```
