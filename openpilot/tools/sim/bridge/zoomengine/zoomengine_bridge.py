import threading
import time
from multiprocessing import Queue

from openpilot.tools.sim.bridge.common import ManualInput, SimulatorBridge, control_cmd_gen
from openpilot.tools.sim.bridge.zoomengine.zoomengine_world import ZoomEngineWorld

REENGAGE_AFTER = 3.0  # seconds


class ZoomEngineBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, traffic: str = "heavy", map_path: str | None = None, port: int = 9090, seed: int = 0,
               auto_reengage: bool = True):
    super().__init__(dual_camera, high_quality)
    self.traffic = traffic
    self.map_path = map_path
    self.port = port
    self.seed = seed
    self.auto_reengage = auto_reengage

  def spawn_world(self, queue: Queue):
    return ZoomEngineWorld(queue, self.dual_camera, port=self.port, seed=self.seed, traffic=self.traffic, map_path=self.map_path)

  def manual_input(self) -> ManualInput | None:
    world = self.world
    return world.manual_input() if isinstance(world, ZoomEngineWorld) else None

  def _run(self, q: Queue):
    if self.auto_reengage:
      threading.Thread(target=self._reengage_loop, args=(q,), daemon=True).start()
    super()._run(q)

  def _reengage_loop(self, q: Queue):
    """Like an attentive driver: when openpilot drops out on its own (not by brake/cancel), turn cruise back on and set"""
    press_main = True
    while self._keep_alive:
      time.sleep(0.5)
      world = self.world
      if isinstance(world, ZoomEngineWorld) and world.disengaged_for() > REENGAGE_AFTER:
        q.put(control_cmd_gen("cruise_main" if press_main else "cruise_down"))
        press_main = not press_main
