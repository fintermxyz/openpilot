"""signal monitor (PLAN §4.3, M1 rules a, b, f, g): is the SPaT for our approach believable? Trust is latched until reset().
The director's fault flag is never an input: only the aspects it broadcasts, their age and what the other cars are doing"""
from openpilot.tools.sim.l3.msgs import TRUSTED_AMBER, TRUSTED_GREEN, TRUSTED_RED, TRUSTED_RED_AMBER, UNTRUSTED, Actor, SignalHealth
from openpilot.tools.sim.l3.params import SignalParams

NON_RED = ("green", "amber", "red_amber")  # 'dark' is rule (b), not a conflict
CONFLICT_WINDOW_S = 2.0   # s: conflicting non-red accumulated over this window (not only contiguously) for conflict_debounce_s latches
SAMPLE_CAP_S = 0.1        # s: one SPaT sample stands for at most this long
TRUSTED = {"green": TRUSTED_GREEN, "amber": TRUSTED_AMBER, "red": TRUSTED_RED, "red_amber": TRUSTED_RED_AMBER}


class SignalMonitor:
  def __init__(self, params: SignalParams, ego_route: str, cmap):
    self.p = params
    self.cmap = cmap
    self.route = ego_route
    self.phase = cmap.routes[ego_route].phase
    self.cross_phases = sorted({r.phase for r in cmap.routes.values()} - {self.phase})  # every other phase conflicts (two-phase crossroads)
    self.reset()

  def reset(self) -> None:
    self.latched: SignalHealth | None = None
    self.conflict_t0: float | None = None
    self.conflict_samples: list[tuple[float, float]] = []  # (spat_t, seconds it stands for) of recent conflicting messages
    self.last_spat_t: float | None = None
    self.dark_t0: float | None = None
    self.last: SignalHealth | None = None

  def _untrusted(self, cause: str, now: float) -> SignalHealth:
    self.latched = SignalHealth(UNTRUSTED, cause, now)
    self.last = self.latched
    return self.latched

  def _behavioural(self, actors: list[Actor]) -> bool:
    """(f) a cross-phase car that can no longer stop for its line (or is already past it, short of the box exit) while we're green"""
    for a in actors:
      r = self.cmap.routes.get(a.route)
      if r is None or r.phase == self.phase or a.v <= self.p.behavioural_v:
        continue
      d = r.stop_line_s - a.s
      if d >= 0.0:
        if d < a.v ** 2 / (2.0 * self.p.behavioural_brake) + 2.0:
          return True
      elif a.s < r.junction_s[1]:
        return True
    return False

  def update(self, spat: dict[str, str] | None, spat_t: float | None, actors: list[Actor], now: float) -> SignalHealth:
    if self.latched is not None:
      return self.latched

    # (g) stale or missing SPaT
    if spat is None or spat_t is None or now - spat_t > self.p.stale_s:
      return self._untrusted("stale", now)

    own = spat.get(self.phase, "dark")

    # (a) conflicting phases both non-red, debounced: contiguously, or accumulated over CONFLICT_WINDOW_S (a head flickering
    # between green and red faster than the debounce is still a conflicting green)
    conflict = own in NON_RED and any(spat.get(ph, "dark") in NON_RED for ph in self.cross_phases)
    new_msg = self.last_spat_t is None or spat_t != self.last_spat_t
    if new_msg:
      if conflict:
        w = SAMPLE_CAP_S if self.last_spat_t is None else min(max(spat_t - self.last_spat_t, 0.0), SAMPLE_CAP_S)
        self.conflict_samples.append((spat_t, w))
      self.last_spat_t = spat_t
      self.conflict_samples = [(t, w) for t, w in self.conflict_samples if spat_t - t < CONFLICT_WINDOW_S]
    if conflict:
      if self.conflict_t0 is None:
        self.conflict_t0 = now
      if now - self.conflict_t0 >= self.p.conflict_debounce_s:
        return self._untrusted("conflict", now)
    else:
      self.conflict_t0 = None
    if sum(w for _, w in self.conflict_samples) >= self.p.conflict_debounce_s - 1e-9:
      return self._untrusted("conflict", now)

    # (b) own head dark / unknown
    if own not in TRUSTED:
      if self.dark_t0 is None:
        self.dark_t0 = now
      if now - self.dark_t0 >= self.p.dark_s:
        return self._untrusted("dark", now)
    else:
      self.dark_t0 = None

    # (f) behavioural: our green, yet a crossing car is committed
    if own == "green" and self._behavioural(actors):
      return self._untrusted("behavioural", now)

    if own not in TRUSTED:
      # dark but not yet debounced: hold the last trusted aspect, red if there was none (never invent a green)
      state = self.last.state if self.last is not None else TRUSTED_RED
    else:
      state = TRUSTED[own]
    if self.last is None or self.last.state != state:
      self.last = SignalHealth(state, "", now)
    return self.last
