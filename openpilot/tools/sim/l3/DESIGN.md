# Artisan L3 research stack for the openpilot + ZoomEngine simulator: final implementation plan

**Basis.** The two judges split their votes between *supervisor-first* and *verification-led*, and both of those use the same Simplex architecture. This plan therefore takes:
- from **supervisor-first**: the Simplex `min()` arbiter in the bridge, run in-process, and its junction mechanics (approach envelope, point of no return, gap test using latest exit, exit space);
- from **verification-led**: the discipline: a red baseline first, INVALID runs, three independent collision sources, a core-diff gate, the fake driver monitoring left attentive, a scripted preset, and an R157-strict comparison mode;
- from **modular-l3**: the ODD-X coverage check with phantom actors, the parameter provenance format, and attribution of who was at fault.

M1 is deliberately small. Its only job is to stop the crossroads crashes.

**Paths.** `OP/` = the `openpilot/` Python package of this repo. `ZE/` = a ZoomEngine checkout (default `~/Documents/zoom/ZoomEngine-main/`).

---

## 1. Goal and honest scope

### What "L3" means here
- **Inside a declared ODD, the system owns the whole driving task (DDT).** That includes cross traffic, right of way and failed signals, which openpilot's Level 2 model does not handle. openpilot remains the *Advanced Controller* for lane keeping and nominal speed. A hand-written, rule-based, unit-tested supervisor (the *Safety Controller* plus *Decision Module*) can only make the car **more conservative**:
  - acceleration out = `min(a_openpilot, a_cap)`;
  - it can hold the brake;
  - it can hand steering to a map pure-pursuit fallback when openpilot is unavailable.
- **The system is its own fallback.** It monitors the ODD, issues a takeover request (TOR) with lead time, escalates, and runs a minimal-risk manoeuvre (MRM) to a minimal-risk condition (MRC) when the driver does not respond. It never hands back silently to Level 2 (R157 §6.2.6).
- **"Artisan" means:** every decision is a pure, deterministic Python function with named parameters, and every parameter carries its source and whether it has been verified. No new machine learning.

### Out of reach in the simulator (recorded in `OP/tools/sim/l3/README.md`)
- Real driver-availability sensing, and whether the HMI is actually perceptible or accessible.
- ISO 26262 hardware metrics and process audit.
- A crash-survivable, OBD-readable, tamper-proof DSSAD.
- R10, R155 and R156 (EMC, cybersecurity, software updates).
- Physical and real-world tests. Simulation may not replace them (R157 Annex 4 §4.2).
- Statistical equivalence to a careful and competent GB driver.
- AV Act authorisation and ASDE status.
- Validated sensor optics and physics. ZoomEngine's ABS, tyres and IDM NPCs are unvalidated.

**Every result is simulator design verification only, with no road-safety claim.**

### Regulatory framing
- **ODD-H (highway)** is "R157-like".
- **ODD-X (UK signalised crossroads)** is a **research ODD extension** with no regulatory precedent, because R157 excludes intersections. It is labelled that way in every artefact.
- An **R157-strict mode** treats the junction as a planned ODD exit, for comparison.

### Simulation-only guardrails (hard requirements, checked in every gate)
1. All L3 code lives in `OP/tools/sim/l3/`. The integration edits stay inside `OP/tools/sim/` (bridge, lib, launch scripts).
2. `OP/tools/sim/l3/__init__.py` calls `assert_sim_only()` at import and raises unless **all** of these hold:
   - `SIMULATION=="1"`, `NOBOARD` is set, `L3=="1"`;
   - `openpilot.system.hardware.PC` is true;
   - `/data/params` does not exist.
   
   If the gate fails, the bridge behaves exactly as it does today.
3. `OP/tools/sim/l3/tools/check_no_core_diff.sh` runs `git diff --exit-code` against the merge base for `OP/selfdrive/`, `OP/cereal/`, `OP/system/`, `opendbc_repo/` and `panda/`. It must be empty. The only later exception is an optional, explicitly reviewed M6 patch.
4. `test_isolation.py` checks the import graph: nothing outside `tools/sim` imports `tools.sim.l3`.
5. L3 actuates **only** through ZoomEngine `vehicle_input`, via the world's `_pedals_for_accel` and `steer_cmd`. It never publishes `carControl` or `sendcan`, and never writes `simulator_state.user_*`. The one exception is the VirtualDriver, which *is* the simulated human. The arbiter's brake therefore never becomes `brakePressed`.
6. Nothing changes in panda or opendbc safety, `ACCEL_MIN`/`ACCEL_MAX`, carcontroller limits, `selfdrive/monitoring`, or `process_config.py`. The fake `driverMonitoringState` stays "attentive". Takeover requests use the engine HUD, not driver-monitoring alert levels.

---

## 2. ODD

ODD definitions are machine-readable files, `OP/tools/sim/l3/odd/odd_x.json` and `odd_h.json`. Each field is `{value, unit, source, verified}`, in the same style as `ZE/tools/roadkit/profiles/uk.json`. Every DSSAD record and every report carries the `odd_id` and `params_hash`.

### ODD-X v1 (built first, in M1-M2): research extension
- **Map:** `ZE/assets/maps/crossroads.built.json`, UK profile, left-hand traffic. Four straight-through routes (N_S, S_N, E_W, W_E). Speed limit 13.41 m/s, and the ego cap is 13.4 m/s.
- **Agents:** cars only. No pedestrians or cyclists, declared. Daylight, dry, as scenario constants.
- **Cross-traffic sensing:** a labelled cooperative feed (director `world` message as V2X/SPaT plus actor list). M1 receives it noise-free and labelled `oracle`. From M2 it has 100 ms latency, 2% loss, σ_pos 0.3 m and σ_v 0.2 m/s, labelled `v2x`.
- **Signal faults are inside the ODD.** These are stuck_green (existing), dark, flashing amber, stuck red, sequence violation and partial conflict. They are handled as an unmarked junction under Highway Code rules 176 and 146, not with a TOR.
- **Exits:** actor feed stale > 0.5 s, SPaT stale > 1.0 s, unknown route, or the generic conditions (driver, DSSAD, openpilot active, heartbeat). From M5 the **coverage check** also applies: every conflicting approach must be observed to `d_cov = v_lim·(t_clear + PET_min) + 20 m ≈ 75 m`. Occluded sectors spawn a phantom actor at the occlusion edge moving at 1.2·v_lim. With the 40° camera only, ODD-X is **refused**; this is a negative control.
- **Interior rule:** the car never comes to rest inside `junction_area` (half-size 9.65 m) unless an emergency manoeuvre (EM) is the only way to avoid a collision.
- **R157-strict variant** (`L3_STRICT=1`): the junction is a planned ODD exit. The TOR fires at `D_tor = 10v + v²/8 + 10 m` (about 167 m at 13.4 m/s) before the stop line, and the MRM stops before the line. This needs approach legs of 180 m or more (see risks).

### ODD-H v1 (M4): R157-like motorway
- **Map:** the ZoomEngine highway loop (`track.py`). BeamNG Autobahn comes later. Divided carriageway, opposing traffic on `opp_traffic`, no pedestrians or cyclists. The ego makes **no lane changes**, so the MRM stays in its lane.
- **Speed profiles:**
  - "R157-00 verified": v ≤ 16.7 m/s (60 km/h).
  - "extended": v ≤ 27.8 m/s (100 km/h). This rises to 36.1 m/s (130 km/h) only after the R157 01-series time-gap and range figures are checked against the consolidated text; until then they are `verified:false`.
  - Further caps: `v ≤ sqrt(2.0/|κ|)` over the next 100 m, and `v_sense`, which satisfies `ρv + v²/(2·4) ≤ 0.8·R_declared`.
- **Traffic:** presets `light` and `heavy`, plus a new `scripted` preset (random hazards off, injected events only).
- **Perception health:**
  - modelV2 `laneLineProbs[1,2] ≥ 0.5`, averaged over 1 s;
  - modelV2 at 15 Hz or more;
  - camera `hz_measured ≥ 18` and frame age ≤ 150 ms (shm `frame_time_us`);
  - object feed age < 150 ms;
  - the model lead and the radar-like lead agree within max(3 m, 15%).

### Engagement conditions (both ODDs; all must hold, R157 §6.2.3)

Each is logged as `{ok, value, threshold}`.

1. Map-matched, with |d| < 1.2 m.
2. v ≤ v_odd_max.
3. openpilot `selfdriveState.active`, with no NO_ENTRY, softDisable or immediateDisable event.
4. Feeds fresh, using the supervisor's own timestamps and **never** `sm.alive` or `freq_ok` (SIMULATION=1 makes those always true).
5. DSSAD writable, with its hash chain verified.
6. VirtualDriver seated, belted and available (2 of 4 criteria within 30 s).
7. Self-check passed.
8. No latched MRC or FAILSAFE since the last trial reset, which stands in for an ignition cycle.

Hysteresis: a condition must be false for 0.3 s to raise a TOR, and true for 2 s to allow engagement.

### Later ODD
- Turning movements, M6: add exits to `ZE/assets/maps/crossroads.json` and rebuild with `ZE/tools/roadkit/assemble.py`. Covers opposed right turns (rules 180 and 181).
- 130 km/h, once verified.
- Noisy-sensor ODD-X with the coverage check, M5.
- Still out of scope: lane changes, pedestrians and cyclists, emergency vehicles, night, rain.

---

## 3. Architecture

### Process layout
No new OS processes, no new rendered cameras and no new HTTP requests.

```
                     ┌───────────────────────────── ZoomEngine (C++, :9090) ─────────────────────────────┐
                     │ physics ego per car (vehicle_input / query "vehicles"), 40° road cam per car (shm) │
                     └───────▲───────────────────────────────▲──────────────────────────────▲────────────┘
                             │ batch+query 50 Hz              │                              │ lamps/signals
┌────────────────────────────┴─── bridge process, car i (OPENPILOT_PREFIX=crossroads$i) ─────┴──────────┐
│ engine thread 50 Hz: ZoomEngineWorld/CrossroadsCarWorld._engine_step                                   │
│   ├─ ego state (_read_vehicle), ground truth (traffic.cars | director world msg)                       │
│   └─ every 2nd frame (25 Hz): L3Host.tick()                                                            │
│        Perception adapter ─► SignalMonitor ─► ODD monitor ─► JunctionManager | HighwayGuard            │
│                                   └──────────────► TakeoverManager (mode FSM) ─► MRM/EM ─► L3Command    │
│        DSSAD (hash-chained JSONL)   Trace (25 Hz JSONL)   HMI (HUD lines, hazard lamp cmds)             │
│ main thread 100 Hz (common.py): world.apply_controls ─► Arbiter.apply(a_op, steer_op, L3Command)       │
│        min()/floor/jerk/brake-hold/watchdog ─► _pedals_for_accel ─► vehicle_input                      │
│        VirtualDriver ─► simulator_state.user_* / CANCEL / blinkers                                     │
│ PubMaster: customReservedRawData0 = L3Status JSON @10 Hz (log/plotjuggler only; nothing consumes it)  │
└──────────────▲──────────────────────────────────┬──────────────────────────────────────────────────────┘
   SubMaster:  │ modelV2, carState, radarState,    │ UDP pose+l3 → 127.0.0.1:9190 (50 Hz)
   selfdriveState, carControl, controlsState       ▼
┌──────────────┴───── openpilot per car (UNCHANGED) ┐   ┌──── crossroads_director.py (20 Hz) ───────────┐
│ modeld radard plannerd controlsd selfdrived card  │   │ SignalController (+fault modes), judge (SAT,  │
│ BLOCK=ui,soundd in 4-car mode                     │   │ PET, stop-line, box-stop), results JSON       │
└───────────────────────────────────────────────────┘   │ NEW: world msg → 9200+i @20 Hz (no fault flag)│
                                                        └────────────────────────────────────────────────┘
```

**Why in-process.**
- Four openpilots already saturate the Mac.
- The bridge is the real actuator layer in sim and has the last word.
- An in-process heartbeat sidesteps the SIMULATION=1 "alive forever" problem.

**The known weakness** is common cause: the supervisor and watchdog share one process and one GIL. The mitigations are a GIL-starvation test in M2, and an optional *out-of-process* guard in M5, the engine-side `vehicle_input` TTL (see M5).

### Messages and interfaces
Dataclasses with JSON (de)serialisers live in `OP/tools/sim/l3/msgs.py`. Every message carries `schema:"l3/1"`.

- **Actor**: `{id, kind, source: oracle|v2x|radar|vision|phantom, t_obs, x, z, psi, v, a, length, width, route|lane, s, d, d_dot, age, l3?: {mode, jx_state, t_stop, token}}`. Intent fields (`lc_t`, `brake_until_v`, `aggressive`, `v_des`) and the director's fault truth are **never** copied. Only the evaluator sees them.
- **EgoState**: `{t, x, z, psi, v, a, yaw_rate, steer_deg, route, s, d, op_active, steering_pressed, brake_pressed, gas_pressed, abs_active, body_contact, cam_hz, frame_age_ms}`.
- **SignalObs**: `{t_obs, source: spat|cam, aspects: {NS, EW}}`.
- **SignalHealth**: `{state: TRUSTED_GREEN|TRUSTED_AMBER|TRUSTED_RED|UNTRUSTED, cause: conflict|dark|flash|sequence|stuck|behavioural|stale, since}`.
- **OddStatus**: `{ok, odd_id, conditions: {name: {ok, value, threshold}}, exit_kind: none|planned|unplanned|severe|driver, dist_to_boundary_m}`.
- **L3Command** (supervisor to arbiter, written under `world.lock`):
  ```
  {t_mono, seq, mode: OFF|AVAILABLE|ACTIVE|CONSTRAINED|TOR|TOR_ESC|MRM|MRC|EM|HANDED_OVER|FAILSAFE,
   a_cap (m/s², +inf = none), a_floor (-3.5 nominal, -4.0 MRM, -a_EM in EM), jerk_down (2.5 | 10 EM),
   brake_hold: bool, steer_source: op|map_pp, hazards: bool, request_cancel: bool,
   hud: [str], reasons: [str], constraint_src: [str]}
  ```
- **L3Status** (10 Hz, published on `customReservedRawData0`, also in the trace): `{mode, odd, tor: {t0, level, cause, planned}, constraints: [{name, a, margin}], rss: {lon_margin, lat_margin, actor}, junction: {state, d_line, signal, token, pet_pred}, a_op, a_cap, a_out, em, dssad_head, step_ms, perception_label}`.
- **Director to car `world`** (UDP 9200+i, 20 Hz, under 4 kB):
  ```
  {type:"world", trial, seq, t, spat:{NS, EW},
   cars:[{car, route, x, z, psi, v, a, s, d, half_w:0.92, half_l:2.3, l3:{mode, jx_state, t_stop, token}}]}
  ```
  The existing `{"reset": n}` gains optional `"scenario": {...}`. There is **no fault field**.
- **Car to director pose**: the existing fields `{car, route, trial, x, z, psi, v, s, d, engaged, parked, t}`, plus `l3: {mode, jx_state, t_stop, token, a_cmd, tor_t0, mrm, em, hazards}`.
- **DSSAD record**: `{seq, t_mono, t_utc, sw: {op_sha, ze_sha}, params_hash, odd_id, car, event, reason, data, prev_hash, hash=sha256(prev_hash+canonical_json(payload))}`.
  - Events: ACTIVATE, DEACTIVATE(kind), TOR(cause, planned), TOR_ESC, MRM_START, MRC, EM_START, EM_END, COLLISION, FAILSAFE, ODD_CHANGE, DRIVER_UNAVAILABLE, SIGNAL_UNTRUSTED.

---

## 4. Components, algorithms and parameters

All tunables live in `OP/tools/sim/l3/params.py` (frozen dataclasses) and `OP/tools/sim/l3/params.json` (`{value, unit, source, verified}`). The `params_hash` goes into the DSSAD.

### 4.1 Arbiter: the Simplex decision module at the actuator
File: `l3/arbiter.py`. It is called from `ZoomEngineWorld.apply_controls` (`zoomengine_world.py:227`) at 100 Hz, *before* `crossroads_car`'s parked override, which still wins.

- `a_op` = `carControl.actuators.accel` when `op_active`.
- `a_raw`:
  - in ACTIVE, CONSTRAINED or TOR: `min(a_op, a_cap)`;
  - in MRM, EM or FAILSAFE: `a_cap`.
  - Then clip to `[a_floor, 2.0]`.
- **Rate limits:**
  - Downward (the safe direction): immediate, up to `jerk_down`.
  - Upward (releasing a cap): ≤ 2.5 m/s³. This is the hand-back blend that absorbs openpilot's PID windup.
- **Output:** `_pedals_for_accel(a_out)`. If `brake_hold`, throttle is 0 and brake is 1.
- **Steering:** openpilot's `steeringAngleDeg`, or `_driver_steer()` (map pure pursuit, labelled "map + oracle localisation") when `steer_source=map_pp`. Transitions are rate-limited at 100°/s.
- **Watchdog:** if `now - cmd.t_mono > 0.2 s` or `L3Host` raised, the arbiter enters a latched **FAILSAFE**:
  - −3.0 m/s² with jerk ≤ 2.5;
  - map_pp steering, which does not depend on openpilot;
  - hazards on, then brake hold, and a DSSAD FAILSAFE event.
- **Severe trigger:** if openpilot commands through an active constraint (`a_op > a_cap + 0.5` for more than 1 s), it is logged and counted. In MRM it is irrelevant because the arbiter owns longitudinal control.
- **Logging:** `a_op`, `a_cap`, `a_out` and the source every tick. Intervention attribution is always explicit.
- **Invariant tests:**
  - `a_out ≤ a_op` except where the floor applies in MRM, EM or FAILSAFE;
  - nothing is ever written to `simulator_state.user_*`;
  - `brakePressed` stays False while the arbiter brakes.

### 4.2 RSS kernel
File: `l3/rss.py`. Pure functions. The ego values are re-tuned to what the bridge can actually deliver, and set from the M0 calibration.

| param | value | source / verified |
|---|---|---|
| ρ_ego | 0.5 s, provisional, measured in M0 by step response | modeld + plannerd + controlsd + engine lag / sim |
| a_acc_ego (worst case during ρ) | 1.0 (the supervisor caps ego accel at 1.0 while margin < 1.5·d_min) | design |
| a_bmin_ego (proper response) | 3.0 nominal, 4.0 MRM | R157 MRM ≤ 4 / design |
| a_EM | 0.85 × measured ABS full-brake decel, capped at 8, provisional 6 | M0 calibration |
| a_bmax_other | 8.0 | ad-rss-lib |
| lateral μ, a_lat_acc, a_lat_brake | 0.3 m, 0.2, 0.8 | ad-rss-lib, μ widened |

- **Longitudinal, same direction:**
  `d_min = max(2.0, v_r·ρ + ½a_acc·ρ² + (v_r+ρ·a_acc)²/(2·a_bmin) − v_f²/(2·a_bmax_other))`.
- **Longitudinal, opposite direction:** the Mobileye formula with `a_bmin_correct = 3.0`.
- **Lateral:**
  `v1ρ = v1+ρ·a_lat_acc`, `v2ρ = v2−ρ·a_lat_acc`,
  `d_lat_min = μ + max(0, (v1+v1ρ)/2·ρ + v1ρ²/(2a_lat_brake) − ((v2+v2ρ)/2·ρ − v2ρ²/(2a_lat_brake)))`.
- **Dangerous** = longitudinal unsafe AND lateral unsafe. The blame time `t_b` is the first moment of the current dangerous period. It is used for ego-at-fault attribution.
- **Graded response:** `a_req = Δv²/(2·max(gap − s0, 0.5)) + max(0, −a_lead)`, with s0 = 3 m.
  - a_req ≤ 3.5: CONSTRAINED, with `a_cap = −1.15·a_req`.
  - a_req > 5.0: EM (R157 §5.3.1.1), with `a_cap = −min(a_EM, max(5, 1.3·a_req))`, floor −a_EM and jerk 10.
  - EM exits when a_req < 2.0 and TTC > 4 s for 1.0 s. The system then continues operating.
- **Intersection RSS:** the lower-priority car must be able to stop before the conflict region after ρ at ≥ a_bmin. If neither car has priority (signal UNTRUSTED), both must be able to stop. This is implemented in the junction manager as the approach envelope plus the gap test.
- **R157 time gap** `t_front(v)`: linear interpolation of 7.2 km/h → 1.0 s, 10 → 1.1, 20 → 1.2, 30 → 1.3, 40 → 1.4, 50 → 1.5, 60 → 1.6, then 2.0 s at 110-130 km/h (that last value `verified:false`). Never below 2 m. It is an SPI and the gap-controller target, not a hard brake trigger.

### 4.3 Signal monitor
File: `l3/signal_monitor.py`. Inputs: SPaT aspects for both phases, their history, and tracked agents. The SignalController fault truth is **never** an input.

The state is **UNTRUSTED** if any of these holds:
- (a) conflict-matrix violation: NS and EW both non-red at the same time, per uk.json phase conflicts;
- (b) the ego head is dark or unknown for ≥ 0.3 s;
- (c) flashing amber;
- (d) a sequence violation against uk.json `red → red+amber → green → amber → red` (for example a skipped amber);
- (e) stuck: an aspect has lasted > 120 s, or a green was shorter than green_min − 0.5 s (7 s minus 0.5 s);
- (f) behavioural: a cross-route agent with v > 3 m/s is inside its own stop-feasibility distance (`d_line < v²/(2·3) + 2`) while the ego aspect is green;
- (g) SPaT age > 1.0 s.

Trust is latched. The monitor returns to TRUSTED only after 2 clean full cycles or a trial reset. Detection latency target: ≤ 0.3 s for conflict and dark.

### 4.4 Junction manager (ODD-X)
Files: `l3/junction.py`, which holds `ConflictGeometry`, `GapChecker`, `JunctionFSM` and `UnmarkedQueue`.

- **Geometry**, precomputed from `CrossroadsMap` routes and cached by map hash:
  - Sample centrelines every 0.25 m. Swept rectangles use half-width 0.92 + 0.5 m and length 4.6 m.
  - Each route pair gives a conflict interval `[cin, cout]` on each route.
  - Opposite through routes have no conflict and may proceed together.
  - Each route also carries `stop_line_s`, `junction_s` and `box_exit_s = junction_s + 2·9.65 + L_ego`.
- **FSM:** `FAR → APPROACH → {GO | STOP_AT_LINE → STOPPED → (QUEUE) } → COMMIT → CLEARED`.
- **APPROACH** begins at `d_line < 80 m`. The speed envelope is the core crash fix and the first green test:
  `v_cap(d) = sqrt(2·a_b·max(d − 1.0 − v·ρ, 0))`, with a_b = 2.5 (2.0 if M0 braking authority is low) and ρ = 0.5.
  `a_cap = 1.0·(v_cap − v) + dv_cap/dt`. Until GO is granted, the car can always stop at the line at ≤ 2.5 m/s².
- **STOP profile:** `a_cap = −v²/(2·max(d − 1.0, 0.3)) − 0.5·(v − v_ref(d))`. Brake hold when v < 0.3 and d < 3.
- **GO** is granted only if all three hold:
  1. the signal is TRUSTED_GREEN, or TRUSTED_AMBER where stopping would need > 3.0 m/s² (rule 175), or the UNMARKED token is held;
  2. the gap test passes against every conflicting actor;
  3. exit space: no actor with v < 2 m/s within `L_ego + 3 m` past `box_exit` (rules 174 and 176, "room to clear").

  With GO, the speed envelope is released, and the release is rate-limited.
- **Gap test (PET):**
  - Ego window `[t_in_e, t_out_e]` over `[cin, cout + L_ego]` uses the planned profile: current v, accelerating at a_go = 1.0 from standstill (otherwise current v) up to `v_cross = min(v_lim, 8)`.
  - Other agent's earliest entry `t_in_o_min`: reaction ρ_o = 1.0 s, then a_max_o = 2.0 up to 1.1·v_lim. A stopped, non-cooperative agent may launch at any time (ρ_o 0.5).
  - Other agent's latest exit `t_out_o_max`: from its current v while braking. If it would stop inside the zone, the exit time is ∞.
  - A cooperative agent broadcasting `STOPPED_YIELD` without a token counts as stationary. A decelerating agent counts as stopping only if stopping before *its* line needs ≤ 4 m/s² and it is already decelerating ("right of way is given, not taken").
  - **Pass** iff `[t_in_e − 2.0, t_out_e + 2.0]` does not intersect `[t_in_o_min, t_out_o_max]` (PET_min = 2.0 s).
- **Point of no return:** if GO is lost (an actor appears, or the signal turns UNTRUSTED) after a stop at 2.5 m/s² has become infeasible:
  - if the car can still stop before `cin − 1 m` at ≤ a_EM, it does so (EM);
  - otherwise it COMMITs.
- **COMMIT:** `a_cap ≥ max(a_op, a_clear)` so the car holds at least 3 m/s until `box_exit`. No stopping in the box. The one exception is an RSS EM for an imminent collision.
- **Unmarked protocol** (signal UNTRUSTED, Highway Code rules 176 and 146; nobody has priority):
  - Mandatory stop: v < 0.2 for 1.0 s at `stop_line − 1 m`. Record `t_stop` and broadcast `STOPPED_YIELD`.
  - Queue: conflicting agents stopped within 4 m of their line, ordered by `(t_stop, car_id)`. Stop times within 0.3 s count as a tie and are broken by the lower id.
  - The token is held when no conflicting agent ahead of the ego in the queue is waiting or is in its conflict zone. There is one token per conflicting pair, and opposite through movements may share a grant.
  - Token AND gap test pass → COMMIT.
  - Deadlock breaker: if no car moves for 8 s, the lowest `(t_stop, id)` whose gap test passes takes the token. In non-cooperative mode (from M2) add a seeded back-off of 0-1.5 s and a creep at ≤ 0.5 m/s to `stop_line + 1 m`, never past `cin − 1 m`. Abort the creep if any conflicting actor exceeds 0.5 m/s.
- **MRM at a junction:**
  - If the car can stop before `stop_line − 1 m` at ≤ 4.0, it does.
  - Otherwise, if stopping before `cin` at ≤ 4 is feasible, it stops there.
  - Otherwise it clears the box at ≥ 3 m/s and stops after `box_exit + L_ego`.

### 4.5 Highway guard (ODD-H)
File: `l3/highway.py`.

- **Lane assignment** of tracks comes from d (lane width 3.7 on the track).
- **Lead** (same lane, ahead): the RSS check, then the graded response from §4.2.
- **Gap controller:** `a_gap = 0.25·(gap − d_R157) + 0.7·(v_l − v_e)`, clipped to [−3, 2], used as `a_cap`.
- **Cut-in:** a track in an adjacent lane becomes a *virtual lead* as soon as:
  - its inner edge is within 0.375 m of the marking and moving toward it at > 0.2 m/s (R157 Annex 3 cue); or
  - lateral RSS is unsafe; or
  - its predicted lateral occupancy over the next 1.44 s overlaps the ego lane edge + 0.3 m.

  Lateral speed comes from the filtered d, never from intent flags. The minimum dwell is 1.0 s.
- **Brake check:** the lead's observed deceleration feeds `a_req`. Flag it when `a_s < −3 m/s²` for 2 cycles.
- **Cut-out reveal:** the next object in the lane becomes the lead the moment its lateral overlap is > 0, including a stationary one.
- **Lane closure / blocked lane:** a stationary object or cone line in the ego lane is a stationary lead. The car stops behind it (R157 §5.2.4). Because there are no lane changes in v1:
  - a map-declared closure is a **planned** ODD boundary, with the TOR at `D_tor`;
  - an undeclared closure is an **unplanned** TOR once the car is stopped behind it, or on detection if the stop requires > 3 m/s².
- **Speed cap:** `min(v_odd, v_limit, v_curv, v_sense)`.
- **Anti-chatter:** a constraint releases only when the margin is > 1.2·d_min for 1.5 s, and the release ramps at ≤ 2.5 m/s³.

### 4.6 Tracking and prediction
File: `l3/perception.py`.

- **Oracle and V2X:** per-id state with an α-β filter on (s, d) and (v_s, v_d), α 0.5, β 0.1. Acceleration comes from a 3-tap filtered dv/dt. Agents older than 0.5 s are *unknown*: the gap test then assumes worst-case arrival from the last position, so they still constrain.
- **Radar-like (M4):** the same α-β filter plus Mahalanobis gating (χ² 9.21) and greedy nearest-neighbour association. A track is confirmed after 2 of 3 hits and deleted after 0.8 s of coasting. Measurements are buffered 200 ms and processed in `t_obs` order. A full Kalman filter is upgraded to only if an OSPA SPI fails.
- **Prediction** over a 6 s horizon:
  - nominal: lane-following at constant v;
  - envelope: arrival bounds with a ∈ [−8, +2] and v ≤ 1.2·v_lim;
  - cut-in hypothesis: active when v_d toward the ego lane > 0.3 m/s or the offset toward ego > 0.375 m.
- **Fusion:** `radarState.leadOne` (vision) is used if prob > 0.5 and it is closer than the tracked lead. The worst case wins. Every constraint records which feed produced it.

### 4.7 ODD monitor
File: `l3/odd.py`.

- Each condition is a pure function `(inputs, params) → (ok, value, threshold)`, with debounce. Freshness uses `logMonoTime`/recv time and an EWMA rate.
- **Planned boundary:** `s_boundary` from route or ODD metadata (zone end, ODD-X boundary in strict mode, route end, declared closure). The TOR triggers when `s_boundary − s ≤ D_tor(v) = 10v + v²/8 + 10`.
- **Unplanned** (TOR): lane prob below threshold for > 1 s, perception feed stale for > 0.3 s, openpilot not active, `hz_measured < 15` for 2 s, SPaT lost while the signal is untrusted, driver unavailable for 15 s, ABS active > 1 s outside EM, gas or brake tap while ACTIVE (R157 §6.3.3).
- **Severe** (immediate MRM): supervisor overrun (> 40 ms twice), a sensor stall > 0.5 s, a localisation inconsistency > 1 m, or openpilot disengaging unexpectedly.
- **Collision** (`vehicles.hits` / `body_contact` / `_colliding`): immediate MRM, latched to MRC with hazards.

### 4.8 Takeover manager (mode FSM)
File: `l3/tor.py`. Tested with a fake clock.

- **States:** `OFF → AVAILABLE → ACTIVE` after an explicit L3 activation (a double press of engine key `L` within 0.6 s, or `AUTO_L3=1` / the scenario's `engage_l3` after openpilot engages).
- **TOR(t0):**
  - HUD amber "TAKE OVER 10…0" plus the reason.
  - All guard and junction constraints stay in force. The TOR adds only its own term, `a_tor ≥ −1.0`, and that term **never relaxes** any other cap. For a planned event, the TOR term also slows the car so the MRM can still stop before the boundary.
- **TOR_ESC** at exactly t0 + 4.0 s (R157 §5.4.3.2, "at the latest"): flashing red HUD and a "haptic" flag. A sound plays in single-car mode only.
- **MRM** at t0 + 10.0 s at the earliest. It starts immediately for severe causes, or earlier for a planned boundary only if the stop distance requires it.
- **Takeover is valid when:**
  - `steeringPressed` holds ≥ 0.5 s and the VirtualDriver is "eyes on road"; or
  - brake or gas is held ≥ 0.5 s with hands on.

  It leads to **HANDED_OVER**: `request_cancel` for 3 frames (cruise CANCEL) so openpilot disengages, auto re-engage is suppressed in `_track_engagement` and in common.py's startup engage, and the VirtualDriver drives with `_driver_steer` + IDM. No silent L2.
- **EM** is an overlay valid in any state. Driver deactivation is deferred while EM is active (§6.2.5.3).
- **MRC:** brake hold, hazards, and a latch until trial reset. There is no drive-off.

### 4.9 MRM controller
File: `l3/mrm.py`.

- Longitudinal: `a_cap = −min(4.0, needed)` from a −2.5 baseline, with jerk 2.0 m/s³. Hazards from t = 0 of the MRM.
- The junction stop-location rule from §4.4 applies.
- Lateral: openpilot steering. It switches to `map_pp` if lane prob < 0.3 or openpilot is inactive.
- Standstill (v < 0.3): brake = 1, which reuses the `parked` pattern, and the state becomes MRC.

### 4.10 VirtualDriver and HMI
Files: `l3/driver_model.py`, `l3/hmi.py`.

- **Driver profiles**, each with a seeded lognormal response latency:
  - ATTENTIVE: median 2.5 s, p95 5 s.
  - NDRT: median 7 s, p95 9 s.
  - LATE: 12-15 s, so the MRM is already running.
  - UNRESPONSIVE: never responds.
  - MISUSE: a brief brake tap without holding the wheel.
- The availability criteria (2 of: control input, blink, head movement, belt/seat) are refreshed in a 30 s window.
- **HMI:**
  - Engine HUD lines appended in `_hud_commands`: "L3 ACTIVE" in green, the TOR countdown, MRM in red, "EMERGENCY BRAKE".
  - **The fake driverMonitoringState is not touched.** `simulated_sensors.py` is unchanged.
  - Hazards: `simulator_state.left_blinker = right_blinker = True` (CAN SCM_FEEDBACK), plus a 1.5 Hz `material` emissive toggle of `{ego}_tailL/R` and `{ego}_headL/R`, following the `crossroads.lamp_commands` pattern. Commands are sent only on change.

### 4.11 DSSAD and trace
Files: `l3/dssad.py`, `l3/trace.py`.

- The DSSAD file is `/tmp/l3/<run>/dssad_<prefix>.jsonl`, opened with O_APPEND and fsynced on every mode change.
- If the open or write test fails, the system goes to AVAILABLE=false and activation is refused.
- `python -m tools.sim.l3.dssad verify <file>` checks the hash chain.
- The trace is a 25 Hz JSONL of about 2 kB per tick.

---

## 5. Sensors: ground truth first, then a path to noisy sensors

| stage | crossroads source | highway source | label | milestone |
|---|---|---|---|---|
| S0 oracle | director `world` (poses + SPaT), exact, intent and fault stripped, plus 50 ms ring-buffer latency | `traffic.cars` + `opp_traffic` passed in-process from `_engine_step`, intent stripped | `oracle` | M1 / M4 |
| S1 V2X | `world` with 100 ms latency, 2% loss (5% stress), σ_pos 0.3 m, σ_v 0.2 m/s | n/a | `v2x` | M2 |
| S2 virtual radar | front 160 m ±45°, corners 80 m ±75° | front 150-160 m ±45°, corners 80 m ±75° | `radar` | M4 (highway), M5 (crossroads) |
| S3 occlusion + coverage | ray shadowing vs actor footprints and corner occluders, phantom actors at edges, ODD-X coverage check | same | `radar+occl` | M5 |
| S4 camera-only negative control | 40° road camera via modelV2 leads; ODD-X must be **refused** | modelV2 leads + lanes only | `vision` | M5 |

Shared details for the radar stages:
- Noise: σ_r = 0.1 + 0.005·r, σ_az 0.3°, σ_v 0.15-0.2 m/s.
- Detection: Pd 0.98 (0.9 beyond 120 m), false alarms 0.05 per scan, latency 60 ± 20 ms, 20 Hz.
- An actor is visible if ≥ 30% of its corner rays are unshadowed.

Lanes come from modelV2 `laneLines`/`laneLineProbs` at every stage. The map_pp lateral fallback uses oracle localisation and is labelled.

No extra rendered cameras are added. Four cars at 20 Hz already produce 80 renders/s, which exceeds one render per frame. Every report is split by perception label, and oracle results are never presented as deployable L3.

---

## 6. File-level work list per milestone

`OP/l3/` means `OP/tools/sim/l3/`. Unless marked NEW, a file is MODIFIED. Every modification is gated on `L3=1`, so with L3 unset behaviour is byte-identical.

### M0: baseline and calibration (about half a day; runs in parallel with M1 coding)
- `OP/tools/sim/bridge/zoomengine/crossroads_director.py`: add `--seed`, `--trials N`, `--out /tmp/l3/<run>/`, and a per-run summary (collisions per trial, cars cleared, min v before the line).
- NEW `OP/l3/runner/run_campaign.py`: starts nothing itself. It drives the director trials, collects results JSON, prints rule-of-three bounds and A/B diffs.
- NEW `OP/l3/runner/calib_brake.py`: through the existing single-car bridge with no L3, from 13 and 30 m/s, applies brake = 0.2/0.4/0.7/1.0 and records the decel, stopping distance and accel step-response lag (ρ). Writes `OP/l3/calib.json`.
- NEW `OP/l3/runner/launch_l3_crossroads.sh`: wraps `OP/tools/sim/launch_crossroads.sh` and exports `L3`, `L3_MASK`, `FAULT`, `SEED`.

### M1: stop the crossroads crashes (minimal)
- NEW `OP/l3/__init__.py` (gate), `OP/l3/params.py`, `OP/l3/params.json`, `OP/l3/msgs.py` (L3Command, Actor, WorldMsg, SignalHealth).
- NEW `OP/l3/signal_monitor.py`: rules (a) conflict, (b) dark, (f) behavioural and (g) stale only.
- NEW `OP/l3/junction.py`: geometry, approach envelope, STOP, the GO gate with the gap test, exit space, point of no return, COMMIT, and the minimal unmarked queue (stop, `(t_stop, id)` order, token, 8 s breaker). No creep and no back-off yet.
- NEW `OP/l3/arbiter.py`: `min()`, the floor, the upward rate limit, brake hold, and the watchdog to FAILSAFE (−3.0, map_pp, hazards via blinker bits only).
- NEW `OP/l3/host.py`: `L3Host.tick()` (25 Hz), building the actor list from the world message and running signal monitor → junction → L3Command. The M1 mode set is OFF/ACTIVE/CONSTRAINED/FAILSAFE. ACTIVE = openpilot active and route on ODD-X.
- `OP/tools/sim/bridge/zoomengine/zoomengine_world.py`:
  - `__init__`: build `self.l3 = L3Host(...)` only if `L3=="1"` (the import goes through the gate).
  - `_engine_step` at `frame % 2 == 0`: `self.l3.tick(ego, self.sm, now)`.
  - `apply_controls`: route `a` through `self.l3.arbiter.apply(...)` before `_pedals_for_accel`.
  - `_hud_commands`: append `self.l3.hud`.
- `OP/tools/sim/bridge/zoomengine/crossroads_car.py`:
  - `_handle_director`: parse `type=="world"` into `self.l3_feed`, while keeping `reset` behaviour.
  - Pose payload: add the `l3` dict.
  - `apply_controls`: parked still overrides last.
- `OP/tools/sim/bridge/zoomengine/crossroads_director.py`: 20 Hz `world` broadcast to 9200+i (SPaT aspects + poses + relayed `l3`, **no fault field**), `--l3-mask`.
- `OP/tools/sim/launch_crossroads.sh`: pass `L3` and `L3_MASK` through to the bridges.
- NEW tests in `OP/l3/tests/`:
  - `test_gate.py`;
  - `test_junction_envelope.py`: a grid over v ∈ [0, 13.4] and d ∈ [0, 80]; GO never granted means the stop at the line needs ≤ 2.5 m/s²;
  - `test_gap.py`: a randomised property test, 1e5 samples; no overlapping windows admitted;
  - `test_arbiter.py`: `a_out ≤ a_op`, no writes to `user_*`, watchdog latches;
  - `test_signal_monitor.py`;
  - `test_unmarked_queue.py`: all 4! arrival permutations plus ties; deterministic and deadlock-free.
- NEW `OP/l3/tools/check_no_core_diff.sh`, `OP/l3/tests/test_isolation.py`.

### M2: crossroads hardening and the verification spine
- `OP/tools/sim/bridge/zoomengine/crossroads.py`:
  - `SignalController.fault_mode ∈ {none, stuck_green, dark, flash_amber, stuck_red, seq_violation, partial_conflict}`;
  - timing read from `crossroads.built.json` `signal_timing`, with a CLI override; the default stays 12 s for backward compatibility.
- `crossroads_director.py`:
  - `--fault`, `--fault-at {never, 0, 6, random}`;
  - judge additions: PET per conflict pair, red/stop-line entries, box standstills, deadlock (all v < 0.1 for > 20 s), time to clear;
  - per-car L3 outcomes;
  - an INVALID flag when engine fps < 45, camera hz < 18 or query lag > 60 ms.
- `OP/l3/signal_monitor.py`: the full rule set, with sequence, stuck, flash and latch.
- `OP/l3/junction.py`: non-cooperative mode (creep plus seeded back-off).
- NEW `OP/l3/perception.py`: V2XReceiver (latency, loss, noise) and the α-β tracker.
- NEW `OP/l3/dssad.py` + `test_dssad.py`. Engagement is gated on a writable DSSAD.
- NEW `OP/l3/trace.py`; L3Status published on `customReservedRawData0`.
- NEW `OP/l3/sim_kinematic.py`: point-mass cars on the real CrossroadsMap geometry, first-order 0.3 s lag, an openpilot-like AC that slows to 5 m/s at the junction and ignores cross traffic. Same host and arbiter code, fake clock.
- NEW `OP/l3/tests/test_closed_loop_crossroads.py`, marked `-m l3_closed_loop`.
- NEW `OP/l3/safety/requirements.yaml` (IDs from §8), `OP/l3/safety/trace_check.py` (fails on any requirement without a scenario or scenario without a requirement). These are written alongside the code, not before it.
- NEW `OP/l3/tests/test_gil_starvation.py`: a CPU-hog thread holds the GIL in bursts, and the watchdog must still trip within 0.3 s.

### M3: ODD monitor, takeover, MRM, VirtualDriver, HMI, DSSAD events
- NEW `OP/l3/odd.py`, `OP/l3/odd/odd_x.json`, `OP/l3/tor.py`, `OP/l3/mrm.py`, `OP/l3/driver_model.py`, `OP/l3/hmi.py`.
- NEW tests: `test_odd.py`, `test_tor.py` (fake clock), `test_mrm.py` (junction stop-location grid).
- `OP/tools/sim/bridge/common.py`: one hook per loop, `world.l3_driver_step(simulator_state)`. It lets the world set the blinkers, the CANCEL button and the VirtualDriver `user_torque/brake/gas`, and suppresses the startup auto-engage when L3 is HANDED_OVER.
- `zoomengine_world.py`:
  - `_track_engagement`: no auto re-engage under HANDED_OVER;
  - lamp `material` commands for hazards appended to the batch;
  - collision events from `_colliding`/hits forwarded to L3.
- `OP/tools/sim/run_bridge.py`: `--l3`, `--scenario <file>`, `--driver <profile>`.
- `OP/tools/sim/launch_openpilot.sh`: pass `L3`, `SEED` and `SCENARIO` through.
- R157-strict flag in `odd_x.json`. If the approach legs are < 180 m: `ZE/assets/maps/crossroads.json` gets longer approach sections, rebuilt via `ZE/tools/roadkit/assemble.py` into `ZE/assets/maps/crossroads.built.json`.

### M4: highway guard
- NEW `OP/l3/rss.py` + `test_rss.py` (hand-computed cases, monotonicity, d_min ≥ 2).
- NEW `OP/l3/highway.py` + `test_highway.py`.
- `OP/l3/perception.py`: VirtualRadar (FOV, noise, dropout, latency) and gated association.
- NEW `OP/l3/odd/odd_h.json`.
- `OP/tools/sim/bridge/zoomengine/traffic.py`:
  - `Preset 'scripted'` with all random hazard rates at 0;
  - `Traffic.inject(event, ego)` for `cut_in(v_rel, ttc_lane_intrusion, lat_v)`, `brake_check(decel, until_v, trigger_gap)`, `cut_out_reveal(v_lead)` and `lane_closure(s, lane, kind=stopped_car|cones)`;
  - all seeded. Existing presets stay byte-identical.
- `zoomengine_world.py` `_engine_step`: hand `traffic.cars`/`opp_traffic` to `L3Host` (in-process, no copy of intent fields).
- NEW `OP/l3/scenarios/highway_*.json`, `OP/l3/sweeps.py` (R157 envelope labelling), `OP/l3/annex3_driver.py` (Annex 3 careful-and-competent reference model, `verified:false` constants).
- NEW `OP/l3/runner/launch_l3_highway.sh`.

### M5: noisy sensors, coverage, robustness, out-of-process guard
- `OP/l3/perception.py`: corner radars at the crossroads, occlusion, phantom actors, `vision` mode.
- `OP/l3/odd.py`: the ODD-X coverage check.
- NEW `OP/l3/falsify.py`: Latin hypercube plus cross-entropy search in the kinematic harness; the top 50 are replayed in the engine.
- Fault injectors in `OP/l3/runner/faults.py`: supervisor exception, V2X loss, radar dropout ×5, noise ×2, kill `modeld` (BLOCK mid-run), engine fps drop, signal fault switched mid-approach.
- **ZoomEngine (optional, C++)**:
  - `ZE/src/core/EngineVehicles.cpp` + `ZE/src/physics/Vehicle.h`: an optional `ttl_ms` field on `vehicle_input`. If no input arrives within the TTL, the engine sets throttle 0 and ramps the brake to give −3 m/s², steering held. It is off unless the field is sent, so existing clients are unaffected.
  - Document it in `ZE/docs/OPENPILOT_SIM.md`.
  
  This is the only guard that does not share the bridge's process. The bridge sends `ttl_ms: 300` when L3=1.
- NEW `ZE/docs/L3_SIM.md`: how L3 uses the API, the performance budget, oracle/V2X labelling.

### M6: safety case, optional HMI patch, turns
- NEW `OP/l3/safety/hara.yaml`, `sotif_triggers.yaml`, `spi.yaml`, `claims.yaml` (UL 4600 style, mapped to the 10 draft UK SoSP principles and to R157 clauses), `README_SAFETY_CASE.md` (gaps list).
- Optional, reviewed separately and SIMULATION-gated:
  - append-only `l3TakeoverRequest1/2`, `l3MinimalRiskManeuver` and `l3OddUnavailable` in `OP/cereal/log.capnp`;
  - entries in `OP/selfdrive/selfdrived/events.py`;
  - a `customReservedRawData0` subscription in `selfdrived.py` only when SIMULATION.
  
  The core-diff gate is waived for exactly these files and nothing else.
- Turns: `ZE/assets/maps/crossroads.json` route exits, rebuilt with `ZE/tools/roadkit/assemble.py`; `OP/tools/sim/bridge/zoomengine/crossroads.py` `CrossroadsMap` loads turning routes; `junction.py` geometry for turns plus opposed right-turn gap acceptance.

### Parallel work (for AI agents)
Integration files are **single-owner** per milestone to avoid merge conflicts: `zoomengine_world.py`, `crossroads_car.py`, `crossroads_director.py` and `common.py`. All other items are pure modules with tests and can be built independently against `types.py`, which must land first; it is small and frozen at the start of M1.

| milestone | independent tracks | depends on |
|---|---|---|
| M0 | A: director seeds/summary + campaign runner · B: brake calibration · C: baseline runs (after A) | none |
| M1 | A: `types/params/gate` (first, 1-2 h) · B: `signal_monitor` + tests · C: `junction` + tests · D: `arbiter` + tests · E: director `world` broadcast · F: integration owner (world/car/host wiring), after A, then merges B-E · G: isolation/core-diff scripts | A before B-F |
| M2 | A: fault modes + judge metrics (director owner) · B: full signal monitor · C: kinematic harness · D: DSSAD + trace · E: V2X receiver/tracker · F: requirements + trace check · G: non-cooperative protocol | M1 |
| M3 | A: odd · B: tor · C: mrm · D: driver_model · E: hmi/lamps · F: integration owner (common.py/world) | M1, with D before F; can run in parallel with M2 |
| M4 | A: rss · B: highway · C: traffic.inject (traffic owner) · D: virtual radar · E: annex3 model + sweeps | M1 arbiter; independent of M2/M3 except the integration owner |
| M5 | A: sensors/occlusion/coverage · B: falsify · C: fault injectors · D: ZE TTL (C++, separate repo, fully independent) | M2-M4 |
| M6 | docs tracks are independent; core patch and turns are separate | M5 |

---

## 7. Milestones, exit criteria and scenario suite

Statistics rule:
- **Engine trials always A/B on identical seeds** (L3=0 vs L3=1).
- A count of 0 is reported with the rule-of-three 95% upper bound, **3/N**.
- INVALID runs (engine fps < 45, camera hz < 18, query lag > 60 ms) are excluded and counted.
- A collision reported by **any** of the three sources fails the run: engine `hits`/`body_contact`, bridge `_colliding`, director SAT judge.
- Collisions are attributed as ego-at-fault (by RSS blame time) or other, for example an NPC rear-ending the ego during an MRM.
- Heavy engine campaigns run nightly. Per-change gates use the kinematic harness plus a short engine smoke run.

### M0: baseline (exit)
- Stuck_green crossroads, 4 cars, L3=0: at least 20 valid trials with fixed seeds. The collision rate is recorded; about 1 per trial is expected, consistent with `/tmp/crossroads_director.log`.
- Normal-signal baseline: 20 trials of time to clear.
- Braking authority at brake = 1 from 13 and 30 m/s, with ρ measured, written to `calib.json` (`verified:true`, sim).
  - If full-brake decel is < 6 m/s², set a_b = 2.0 and lower a_EM accordingly.
- **Open-question check:** after an arbiter brake hold of 10 s or more at standstill with openpilot engaged, does the Civic (openpilot long) resume without a resume press? Hold the brake via the parked pattern in a test run. If it does not resume, M1 adds a sim-only RES_ACCEL button pulse from the bridge when the arbiter releases a hold. This goes through `simulator_state`, the same path as the existing auto-engage.

### M1: crossroads crashes stopped (small, immediately runnable)
Configuration: oracle `world` feed (labelled), 4 L3 cars, `fault_at=6` (the existing stuck_green).

Exit:
1. Unit and property tests pass: envelope grid, gap property (1e5 samples), queue permutations, arbiter invariants, gate.
2. **Stuck_green, 50 valid engine trials: 0 collisions** (bound < 6% per trial; it rises to 100+ trials in M2). L3=0 on the same seeds still crashes.
3. 0 standstills inside `junction_area`. Every car clears within 60 s in ≥ 95% of trials, and unresolved deadlocks are 0.
4. Normal signals, 20 trials: 0 red-light entries, 0 collisions.
5. Arbiter checks:
   - `a_out ≤ a_op` holds in every logged tick outside FAILSAFE;
   - `brakePressed` never goes True from arbiter braking;
   - openpilot stays engaged in 100% of trials (0 disengagements caused by L3).
6. Watchdog: an injected exception in `tick()` gives FAILSAFE within 0.2 s in 20 of 20 runs.
7. With L3 unset, logged throttle, brake and steer are byte-identical to M0 on the same seed. `check_no_core_diff.sh` and `test_isolation.py` pass.
8. Supervisor tick p99 ≤ 2 ms, and engine fps within 3 fps of baseline with 4 cars.

### M2: crossroads hardened (v2x noise, all faults, verification spine)
- Kinematic: 10k seeded episodes across all 7 fault modes, giving 0 conflict-zone overlaps and 0 box stops other than EM.
- Engine (v2x), stuck_green, ≥ 100 trials:
  - 0 collisions (bound < 3%);
  - min PET ≥ 1.5 s in 100% of conflicting crossings, median ≥ 2.5 s;
  - 0 box standstills;
  - ≥ 95% of trials with all cars cleared within 60 s.
- Normal signals, 50 trials:
  - 0 red entries, 0 collisions;
  - median time to clear ≤ 1.3× the M0 baseline;
  - false-intervention rate < 5% of approaches. A false intervention is a constraint active while ground-truth TTC > 6 s and the box is clear.
- Each of dark, flash_amber, stuck_red, seq_violation and partial_conflict, 30 trials: 0 collisions, 100% stop at the line before entry, and detection ≤ 0.3 s after onset for conflict and dark.
- Mixed fleet (2 L3 + 2 plain openpilot), stuck_green, 50 trials: 0 collisions where an L3 car is at fault. L2-caused conflicts are logged with L3 mitigation (impact speed).
- Non-cooperative mode (no intent sharing), 30 trials: 0 collisions and ≥ 90% cleared within 90 s.
- The DSSAD chain verifies, and tampering with one line is detected. The GIL-starvation test passes. `trace_check.py` passes.

### M3: takeover, MRM, VirtualDriver
Across at least 200 TOR events:
- escalation at t0 + 4.0 ± 0.05 s;
- MRM start ≥ t0 + 10.0 s except for severe causes;
- planned TOR distance ≥ 10v + v²/8 + 10.

By driver profile:
- **ATTENTIVE and NDRT:** handover before the MRM in 100%, openpilot disengaged via CANCEL, 0 L2 continuations.
- **UNRESPONSIVE** (100% reach MRC):
  - mean decel ≤ 4.0 m/s², peak ≤ 4.2 excluding EM;
  - 0 lane-marking crossings (`|d| + 0.92 < 1.825`);
  - hazards on within 1 tick of MRM start, and within 5 s of standstill during a TOR;
  - 0 MRCs in `junction_area`;
  - 0 drive-offs before reset.
- **MISUSE:** gives a TOR with cause `override`, not a deactivation.

Other checks:
- Engagement refused in 100% of cases across 9 conditions × injected failure.
- R157-strict crossroads: TOR ≥ 10 s before the line, and the MRM stops before the line in 100% of 30 runs.
- Every event appears in the DSSAD with the correct cause, in order.

### M4: highway guard

Brake check:
- Sweep: lead decel 6/8/10 m/s² × ego 40/60/80/100 km/h × straight/curve × start gap {d_R157, 1.5·d_R157} × lateral offset {0, ±0.5 m}.
- Result: 0 collisions in the kinematic harness (≥ 2000) and the engine (≥ 60).

Cut-in, inside the R157 envelope (visible ≥ 0.72 s, `TTC_LI > v_rel/12 + 0.35`, v_rel 0-10 m/s, v_lat 0.5-2 m/s):
- 0 collisions;
- no EM beyond what a_req requires.

Cut-in, outside the envelope:
- mean impact speed ≤ 50% of L3=0 on the same seeds;
- ≤ 1 mode change per 2 s;
- 0 disengagements.

Cut-out reveal of a stationary car, 60-100 km/h at THW 2 s: 0 collisions.

Lane closure:
- **declared:** planned TOR at ≥ D_tor, then a stop behind it or a handover; 0 collisions;
- **undeclared:** a stop behind it at ≤ 4 m/s² where feasible (EM otherwise), then a TOR; 0 collisions.

Heavy-preset soak, 2 h of sim time:
- 0 ego-at-fault collisions;
- ego-caused time-gap violations < 1 per hour;
- phantom EM < 1 per 30 min;
- release jerk p99 ≤ 2.5 m/s³;
- lateral tyre-edge margin > 0 throughout.

Comparison with the Annex 3 model: in every grid cell, L3 collision count ≤ the model's.

### M5: robustness and SOTIF
- Every injected fault reaches TOR, MRM or FAILSAFE within 1 cycle (≤ 40 ms) plus filter time (≤ 0.3 s), with 0 collisions across ≥ 300 engine fault runs.
- With the engine TTL enabled, a SIGSTOPped bridge gives an engine-side stop at −3 m/s² in 20 of 20 runs.
- Falsification: ≥ 50k kinematic episodes with 0 in-ODD ego-at-fault collisions. Any failure found becomes a regression scenario with a fix.
- Camera-only (`vision`): ODD-X refused or a TOR on approach in 100%.
- Radar+occlusion crossroads: ≤ 1 collision per 200 trials, each root-caused, fixed and re-run to 0.
- Tracker OSPA < 1.0 m within 80 m. False tracks in the ego corridor < 0.1 per minute.
- 4-car CPU within 5% of M1.

### M6: safety case
- 100% of requirements have ≥ 1 passing automated test with a numeric threshold and an SPI.
- The SPI report is generated in one command.
- If the core patch is taken: on a non-SIMULATION build the event set and subscriptions are unchanged (a unit test asserts it), and process_replay shows no diff.
- Turns: the M2 bounds hold over 50 trials.

### Scenario suite
Files in `OP/l3/scenarios/`, each tagged with requirement IDs:

| id | world | description | key verdicts |
|---|---|---|---|
| XR-SG | crossroads | stuck_green at fault_at {0, 6, random 0-20}, arrival jitter ±15 m, all/mixed/1-L3 masks | 0 collisions, PET ≥ 1.5, no box stop, cleared |
| XR-NORM | crossroads | normal UK cycle | 0 red entries, throughput ≤ 1.3× baseline, false-intervention rate |
| XR-FAULT-* | crossroads | dark, flash_amber, stuck_red, seq_violation, partial_conflict, fault switching mid-approach | detection latency, stop at line, 0 collisions |
| XR-SYM | crossroads | 4 cars arriving within 0.3 s (8 seeds) | deterministic order, no deadlock |
| XR-STRICT | crossroads | R157-strict planned exit | TOR ≥ 10 s, MRM stops before line |
| HW-CI | highway | R157 cut-in sweep, inside/outside envelope | as in M4 |
| HW-BC | highway | brake check 6-10 m/s² | 0 collisions from ≥ d_min |
| HW-CO | highway | cut-out reveals stationary car | 0 collisions |
| HW-LC | highway | lane closure, declared and undeclared | stop behind, TOR, 0 collisions |
| ODD-EXIT | both | zone end, route end, ODD-X boundary, lane-prob drop, feed loss, openpilot disengage | TOR timing, MRM, refusal |
| DRV-* | both | ATTENTIVE/NDRT/LATE/UNRESPONSIVE/MISUSE × triggers | handover or MRC, no silent L2 |
| FLT-* | both | supervisor exception, GIL starvation, bridge SIGSTOP (TTL), modeld kill, V2X loss, radar dropout | FAILSAFE/TOR latency, 0 collisions |

---

## 8. Validation, metrics and the safety case

**Test tiers.** All tiers use the same scenario JSON and the same evaluator.
- **Tier 0:** pytest plus hypothesis, under 2 min, with no engine and no openpilot. Covers RSS, junction envelope and gap properties, queue permutations, signal monitor against every legal and illegal uk.json transition, TOR timing on a fake clock, the MRM location grid, arbiter invariants, DSSAD tamper, gate, isolation and core-diff.
- **Tier 1:** the kinematic closed loop, `-m l3_closed_loop`, about 10 min per change. The crossroads and highway libraries at 20 seeds each, plus the Annex 3 comparison.
- **Tier 2:** engine smoke, about 20 min, 12 scenarios. Run before merging a milestone.
- **Tier 3:** nightly engine campaign of a few hours, plus the soak. Realistic for one Mac, and not used as a per-change gate.

**The evaluator is tested itself.** Synthetic traces with an injected collision, a time-gap violation and a TOR timing violation must each be caught.

**Metrics and SPIs** (UL 4600 style, each tied to a claim with a leading threshold):
- **Collisions:** per trial and per hour, split ego-at-fault vs other and by perception label.
- **Junction:** occupied-conflict-zone entries per 1000 passes; min and median PET; red, stop-line and box-stop counts; deadlocks; time to clear; signal-fault detection latency.
- **Longitudinal:** min TTC p5; minimum RSS margin; time-gap violations per hour.
- **Takeover and MRM:** TOR lead time and escalation time; MRM mean and peak decel, final |d|, in-box flag, hazard latency.
- **Supervisor behaviour:** EM per hour (phantom EM); false-intervention rate; mode switches per minute; release jerk p99; supervisor authority (`a_op − a_out`); ODD-monitor false negatives (monitor vs evaluator-side ground truth).
- **Health:** step_ms p99; engine fps; camera hz_measured.

**Comparators:** the openpilot-only baseline on the same seeds, which must be beaten, and the R157 Annex 3 careful-and-competent model, which must not be beaten.

**Safety-case contents**, built up from M2 to M6:
- **ODD files.**
- **Requirement IDs:** L3-DEF-1, ODD-1/2, XR-1/2/3 (no MRC in the box) and XR-4 (deadlock-free), GEN-1, LK-1, LON-1/2, BC-1, CI-1, CO-1, CMP-1, EM-1, TOR-1/2, MRM-1, DM-1, HO-1, HMI-1, PER-1, LOG-1, UK-1, SIM-1 (sim-only gate), SAFE-1 (no core diff), WD-1 (heartbeat).
- **Traceability matrix:** requirement → scenario → metric → threshold.
- **HARA** covering H1-H11:
  - H1 brake-check rear-end;
  - H2 cut-in;
  - H3 junction cross traffic;
  - H4 lane departure;
  - H5 MRC in the box;
  - H6 mode confusion or silent L2;
  - H7 short TOR lead;
  - H8 undetected stale supervisor;
  - H9 phantom EM causing a rear-end;
  - H10 deadlock;
  - H11 infractions.
- **Safety goals** SG1-SG6.
- **SOTIF triggering-condition catalogue:** conflicting greens, dark heads, stuck phase, cut-in visible < 0.72 s, cut-out reveal, lane-prob loss, camera freeze or frame drop, V2X loss, symmetric arrival, occlusion.
- **UL 4600 claims tree** mapped to the 10 draft UK SoSP principles and to R157 clauses, with evidence links less than 7 days old.
- **The explicit gaps list** from §1.

---

## 9. Risks and open questions

1. **Oracle and V2X optimism.** The 40° camera cannot see cross traffic, so ODD-X results prove planning and supervision logic, not L3 perception.
   - Mitigations: mandatory labels, the fault truth withheld, V2X noise, mixed and non-cooperative fleets, the M5 coverage check, and the camera-only negative control.
2. **The arbiter is invisible to openpilot.** LongControl windup and plan divergence cause surges at hand-back, especially after a long stop-line hold.
   - Mitigations: the +2.5 m/s³ release limit, the release-jerk SPI.
   - Escape hatches, used only when an SPI fails: (a) a sim-only `BLOCK=radard` wrapper injecting a virtual stopped lead at the line, so openpilot brakes itself; (b) L3 owning longitudinal fully with openpilot in standby (the modular-l3 pattern).
3. **Resume from standstill (open question).** It is not yet known whether the Civic (openpilot long) resumes after a long arbiter brake hold without a resume press. M0 checks this, and the fallback is a bridge resume pulse.
4. **Braking authority is unmeasured.** EM floors and a_b are provisional until the M0 calibration. If authority is low, a_b = 2.0 and the approach starts earlier.
5. **Common-cause watchdog.** The supervisor, watchdog and bridge share one process and GIL.
   - Mitigations: the GIL-starvation test in M2, and the engine-side `vehicle_input` TTL in M5, which is the only out-of-process guard.
   - An unguarded window remains until M5. This is acceptable in sim only.
6. **SIMULATION=1 makes stale messages look alive.** No openpilot process consumes L3 messages, and every L3 freshness check uses its own timestamps.
7. **CPU and render budget.** No new processes, cameras or HTTP requests. The tick budget is ≤ 2 ms, and runs are marked INVALID on fps collapse.
   - Fallback: supervisor at 20 Hz (the arbiter still runs at 100 Hz), or run with 2-3 cars.
   - Engine jitter can still produce flaky verdicts. Flakes are tracked, with a ≤ 2% budget.
8. **Unmarked-junction liveness.** Mitigations: a deterministic `(t_stop, id)` order, the 8 s breaker, back-off and creep limits, and symmetric-arrival tests.
   - It depends on consistent observations across cars, so V2X-loss cases must be covered.
9. **Conservatism versus throughput.** PET 2 s, "a stopped agent may launch" and ρ_o 1 s may make normal-signal throughput poor. The time-to-clear and false-intervention SPIs track this. Tuning happens only through `params.json`, with a new `params_hash`.
10. **Parameter provenance.** These are `verified:false`: the R157 01-series figures (130 km/h gaps, 150 m range, lane-change and MRM lane-change values), the Annex 3 constants, the uk.json timings, and the HCM headways (US values, mirrored for left-hand traffic). v_max is held at 100 km/h, and the verified 60 km/h profile is reported separately.
11. **R157-strict needs about 180 m of approach.** The current spawn and approach legs may be shorter, in which case the road-kit approach sections in ZE assets must be lengthened (M3).
12. **TOR deceleration rule.** The TOR's own term (≥ −1.0) must never relax a guard or junction cap. `test_tor.py` asserts that `min()` composition holds.
13. **NPC followers.** IDM NPCs may rear-end an ego that is braking in an MRM or EM. These are counted separately, and a realistic NPC reaction delay is an option.
14. **Time alignment.** Engine query results lag by at least one frame, director poses arrive at 20 Hz over UDP, and NPC smooth moves lead the render.
    - Everything is stamped on `time.monotonic` and extrapolated at constant velocity to now. Feeds older than their threshold take the ODD-exit path.
    - PET margins absorb about 100 ms of skew.
15. **Behaviour drift when L3=0.** Possible sources: the `scripted` preset, the director broadcast and the fault modes.
    - Mitigations: byte-identical regression with L3 unset, existing presets left untouched, and the default timing kept at 12 s.
16. **Scope creep** toward ML or core patches. Every M5+ refinement needs a scenario that fails before it lands, and the core-diff gate stays green except for an explicitly reviewed M6 patch.
17. **Regulatory scope.** ODD-X is a research extension and is labelled that way everywhere. Nothing here transfers to real roads. Real-car code paths are untouched by construction, and every gate re-checks it.