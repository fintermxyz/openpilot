"""L3 host (PLAN §3, §6 M1): runs in the bridge's engine thread at 25 Hz. Builds the actor list from the director's latest
world message, runs signal monitor -> junction manager and hands the arbiter an L3Command. It never raises into the bridge:
any exception latches the arbiter's FAILSAFE."""
import dataclasses
import math
import threading
import time
import traceback
from collections.abc import Callable

from openpilot.tools.sim.l3.arbiter import Arbiter
from openpilot.tools.sim.l3.junction import ConflictGeometry, JunctionManager
from openpilot.tools.sim.l3.msgs import (ACTIVE, COMMIT, CONSTRAINED, EM, FAILSAFE, OFF, UNTRUSTED, Actor, EgoState, JunctionOut, L3Command,
                                         SignalHealth, WorldMsg)
from openpilot.tools.sim.l3.params import PARAMS, L3Params
from openpilot.tools.sim.l3.signal_monitor import SignalMonitor

NO_FEED = "no feed"  # signal cause before the first world message
MODE_COLORS = {OFF: [0.6, 0.6, 0.6], ACTIVE: [0.2, 1.0, 0.4], CONSTRAINED: [1.0, 0.75, 0.3], EM: [1.0, 0.25, 0.2], FAILSAFE: [1.0, 0.25, 0.2]}


class L3Host:
  def __init__(self, ego_id: str, route: str, cmap, params: L3Params = PARAMS, clock: Callable[[], float] = time.monotonic,
               signal: SignalMonitor | None = None, junction: JunctionManager | None = None, arbiter: Arbiter | None = None):
    # signal / junction / arbiter can be injected (tests); by default built from params
    self.ego_id = ego_id
    self.route = route
    self.cmap = cmap
    self.p = params
    self.clock = clock
    self.arbiter = arbiter if arbiter is not None else Arbiter(params.arbiter, keep_v=params.junction.v_clear)
    self.signal = signal if signal is not None else SignalMonitor(params.signal, route, cmap)
    if junction is None:
      junction = JunctionManager(ego_id, route, cmap, params.junction, ConflictGeometry(cmap, params.junction))
    self.junction = junction
    self._feed_lock = threading.Lock()
    self.trial: int | None = None  # the car's trial (None: accept every world message)
    self.reset()

  def reset(self, trial: int | None = None) -> None:
    """new trial: signal latch, junction FSM, arbiter latch and the feed. trial: the trial the car is now in; world messages of
    any other trial (a restarted director, a late packet) are ignored"""
    with self._feed_lock:
      self.trial = trial if trial is not None else self.trial
      self._msg: WorldMsg | None = None
      self._msg_rx: float | None = None
      self.wrong_trial_msgs = 0
    self._known: dict[str, Actor] = {}  # every other car seen this trial, as last observed: never dropped from the picture
    self.signal.reset()
    self.junction.reset()
    self.arbiter.reset()
    self.seq = 0
    self.failed = False
    self.fail_reason = ""
    self.last_cmd: L3Command | None = None
    self.last_signal: SignalHealth | None = None
    self.last_jx: JunctionOut | None = None
    self.last_actors: list[Actor] = []

  # ---- inputs ----
  def feed(self, msg: WorldMsg) -> None:
    """the director's world message (any thread)"""
    with self._feed_lock:
      if self.trial is not None and msg.trial != self.trial:
        self.wrong_trial_msgs += 1  # not our trial: its car list says nothing about the cars around us now
        return
      self._msg, self._msg_rx = msg, self.clock()

  def actors(self, now: float, msg: WorldMsg | None = None) -> list[Actor]:
    """the other cars at now: extrapolated, and 'unknown' (worst case arrival, never gone) once older than feed_stale_s"""
    if msg is None:
      with self._feed_lock:
        msg = self._msg
    if msg is not None:
      for c in msg.cars:
        if c.id != self.ego_id:
          k = self._known.get(c.id)
          if k is None or c.t_obs >= k.t_obs:
            self._known[c.id] = c
    out = []
    for c in self._known.values():
      if now - c.t_obs <= self.p.feed_stale_s:
        out.append(c.extrapolated(now))
        continue
      # stale, or no longer listed: worst case (it may arrive as early as physically possible, never gone). Only a car last seen
      # through the box is dropped: the routes are one way, it can never come back into a conflict zone
      r = self.cmap.routes.get(c.route)
      if r is not None and c.s - c.half_l > r.junction_s[1]:
        continue
      out.append(dataclasses.replace(c.extrapolated(now), source="unknown"))
    return out

  # ---- 25 Hz ----
  def tick(self, ego: EgoState) -> L3Command | None:
    now = self.clock()
    try:
      with self._feed_lock:
        msg = self._msg
      actors = self.actors(now, msg)
      if msg is None:
        # no world message since the start / reset yet: untrusted (unmarked junction rules) but not latched, else a tick that
        # beats the first message after a director reset would latch 'stale' for the whole trial
        sig = SignalHealth(UNTRUSTED, NO_FEED, now)
      else:
        sig = self.signal.update(msg.spat, msg.t, actors, now)
      jx = self.junction.step(ego, actors, sig, now)
      cmd = self._command(ego, sig, jx, now)
      self.arbiter.set_command(cmd)
    except Exception as e:  # never into the bridge: the arbiter takes over
      reason = f"L3 tick raised {type(e).__name__}: {e}"
      self.arbiter.fail(reason)
      if not self.failed:
        self.failed, self.fail_reason = True, reason
        print(f"[l3 {self.ego_id}] FAILSAFE: {reason}\n{traceback.format_exc()}", flush=True)
      return None
    self.last_actors, self.last_signal, self.last_jx, self.last_cmd = actors, sig, jx, cmd
    return cmd

  def _command(self, ego: EgoState, sig: SignalHealth, jx: JunctionOut, now: float) -> L3Command:
    nominal = self.p.arbiter.a_floor
    a_floor, a_min_keep = nominal, None
    if jx.em:
      # emergency stop: the arbiter may brake harder than the nominal floor
      a_floor = min(nominal, jx.a_floor if jx.a_floor is not None else -self.p.junction.a_em)
    elif jx.a_floor is not None:
      if jx.a_floor < nominal:
        a_floor = jx.a_floor
      else:
        a_min_keep = jx.a_floor  # COMMIT: keep moving through the box, overrides openpilot's braking
    constrained = math.isfinite(jx.a_cap) or a_min_keep is not None or jx.brake_hold
    clear_d = None
    if jx.state == COMMIT and a_min_keep is not None:
      clear_d = max(0.0, self.cmap.routes[self.route].junction_s[1] + ego.half_l + 1.0 - ego.s)
    if not ego.op_active:
      mode = OFF
    elif jx.em:
      mode = EM
    elif constrained:
      mode = CONSTRAINED
    else:
      mode = ACTIVE
    self.seq += 1
    reasons = list(jx.reasons)
    if sig.state == UNTRUSTED:
      reasons.append(f"signal untrusted ({sig.cause})")
    return L3Command(t_mono=now, seq=self.seq, mode=mode, a_cap=jx.a_cap, a_floor=a_floor, a_min_keep=a_min_keep, commit_clear_d=clear_d,
                     jerk_down=10.0 if jx.em else 2.5, brake_hold=jx.brake_hold, steer_source="op", hazards=jx.em,
                     hud=[line for line, _ in self._hud(mode, sig, jx)], reasons=reasons)

  # ---- outputs ----
  @property
  def mode(self) -> str:
    if self.failed or getattr(self.arbiter, "failsafe", False):
      return FAILSAFE
    return self.last_cmd.mode if self.last_cmd is not None else OFF

  @property
  def reason(self) -> str:
    if self.failed:
      return self.fail_reason
    if getattr(self.arbiter, "failsafe", False):
      return getattr(self.arbiter, "fail_reason", "")
    return ", ".join(self.last_cmd.reasons) if self.last_cmd is not None else ""

  def status(self) -> dict:
    """the l3 dict of the pose message (the director relays mode / jx_state / t_stop / token to the other cars)"""
    jx, sig, cmd = self.last_jx, self.last_signal, self.last_cmd
    a_cap = cmd.a_cap if cmd is not None else math.inf
    return {"mode": self.mode, "jx_state": jx.state if jx is not None else None, "t_stop": jx.t_stop if jx is not None else None,
            "token": bool(jx.token) if jx is not None else False, "a_cap": round(a_cap, 3) if math.isfinite(a_cap) else None,
            "signal": sig.state if sig is not None else None, "cause": sig.cause if sig is not None else "",
            "em": bool(jx.em) if jx is not None else False}

  def stats(self) -> dict:
    """per-trial counters for the results (not relayed to the other cars): the arbiter invariant, stale / FAILSAFE events"""
    st = dict(getattr(self.arbiter, "stats", {}))
    st["max_excess"] = round(st.get("max_excess", 0.0), 4)
    st["max_stale_s"] = round(st.get("max_stale_s", 0.0), 3)
    st["wrong_trial_msgs"] = self.wrong_trial_msgs
    st["tick_failed"] = self.failed
    return st

  def _hud(self, mode: str, sig: SignalHealth | None, jx: JunctionOut | None) -> list[tuple[str, list[float]]]:
    sig_s = "--" if sig is None else (f"{sig.state} ({sig.cause})" if sig.cause else sig.state)
    if jx is None:
      jx_s = "--"
    else:
      cap = f"{jx.a_cap:+.2f}" if math.isfinite(jx.a_cap) else "--"
      d = f"{jx.d_line:5.1f} m" if math.isfinite(jx.d_line) else "--"
      jx_s = f"{jx.state}  line {d}  cap {cap}{'  TOKEN' if jx.token else ''}{'  EMERGENCY BRAKE' if jx.em else ''}"
    lines = [(f"L3 {mode}   junction {jx_s}   signal {sig_s}", MODE_COLORS.get(mode, [1.0, 1.0, 1.0]))]
    if mode == FAILSAFE:
      lines.append((f"L3 FAILSAFE: {self.reason}", MODE_COLORS[FAILSAFE]))
    return lines

  def hud_lines(self) -> list[tuple[str, list[float]]]:
    return self._hud(self.mode, self.last_signal, self.last_jx)
