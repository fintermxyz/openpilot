#!/usr/bin/env python3
import argparse

from typing import Any
from multiprocessing import Queue


def create_bridge(dual_camera, high_quality, sim="metadrive", traffic="heavy", map_path=None, car=0):
  queue: Any = Queue()

  if sim == "crossroads":
    from openpilot.tools.sim.bridge.zoomengine.crossroads_car import CrossroadsCarBridge
    simulator_bridge = CrossroadsCarBridge(car, map_path)
  elif sim == "zoomengine":
    from openpilot.tools.sim.bridge.zoomengine.zoomengine_bridge import ZoomEngineBridge
    simulator_bridge = ZoomEngineBridge(dual_camera, high_quality, traffic, map_path)
  else:
    from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge
    simulator_bridge = MetaDriveBridge(dual_camera, high_quality)
  simulator_process = simulator_bridge.run(queue)

  return queue, simulator_process, simulator_bridge

def main():
  _, simulator_process, _ = create_bridge(True, False)
  simulator_process.join()

def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between the simulator and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')
  parser.add_argument('--sim', choices=['metadrive', 'zoomengine', 'crossroads'], default='metadrive')
  parser.add_argument('--car', type=int, default=0, help='crossroads: which car (0-3) this bridge and its OPENPILOT_PREFIX drive')
  parser.add_argument('--traffic', choices=['light', 'heavy', 'chaos'], default='heavy', help='zoomengine traffic difficulty')
  parser.add_argument('--map', help='zoomengine: level.json of a BeamNG level converted with ZoomEngine tools/beamng_import.py')

  return parser.parse_args(add_args)

if __name__ == "__main__":
  args = parse_args()

  queue, simulator_process, simulator_bridge = create_bridge(args.dual_camera, args.high_quality, args.sim, args.traffic, args.map, args.car)

  if args.joystick:
    # start input poll for joystick
    from openpilot.tools.sim.lib.manual_ctrl import wheel_poll_thread

    wheel_poll_thread(queue)
  elif args.sim in ("zoomengine", "crossroads"):
    # driving keys come from the ZoomEngine window (sim_input), not this terminal; Q there quits
    simulator_process.join()
  else:
    # start input poll for keyboard
    from openpilot.tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

    keyboard_poll_thread(queue)

  simulator_bridge.shutdown()

  simulator_process.join()
