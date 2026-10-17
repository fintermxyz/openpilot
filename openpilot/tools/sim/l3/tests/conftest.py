import os

# The L3 package refuses to import unless the simulator gate is open. This directory deliberately has no
# __init__.py, so pytest imports this conftest as a top-level module (before anything imports the gated
# openpilot.tools.sim.l3 package) and the variable is set in time.
os.environ["L3"] = "1"


def pytest_configure(config):
  config.addinivalue_line("markers", "slow: long-running closed-loop tests")

import pytest

MAP = os.path.join(os.environ.get("ZOOMENGINE_DIR", os.path.expanduser("~/Documents/zoom/ZoomEngine-main")), "assets/maps/crossroads.built.json")


def pytest_collection_modifyitems(config, items):
  # the tests run on ZoomEngine's crossroads map (built by its road kit): skip them where there's no ZoomEngine checkout
  if not os.path.exists(os.environ.get("CROSSROADS_MAP", MAP)):
    skip = pytest.mark.skip(reason=f"no crossroads map at {MAP} (set ZOOMENGINE_DIR)")
    for item in items:
      item.add_marker(skip)
