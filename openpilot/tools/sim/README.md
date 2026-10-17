openpilot in simulator
=====================

openpilot implements a [bridge](run_bridge.py) that allows it to run in the [MetaDrive simulator](https://github.com/metadriverse/metadrive).

## Launching openpilot
First, start openpilot.
``` bash
# Run locally
./openpilot/tools/sim/launch_openpilot.sh
```

## Bridge usage
```
$ ./run_bridge.py -h
usage: run_bridge.py [-h] [--joystick] [--high_quality] [--dual_camera]
Bridge between the simulator and openpilot.

options:
  -h, --help            show this help message and exit
  --joystick
  --high_quality
  --dual_camera
```

#### Bridge Controls:
- To engage openpilot press 2, then press 1 to increase the speed and 2 to decrease.
- To disengage, press "S" (simulates a user brake)

#### All inputs:

```
| key  |   functionality       |
|------|-----------------------|
|  1   | Cruise Resume / Accel |
|  2   | Cruise Set    / Decel |
|  3   | Cruise Cancel         |
|  r   | Reset Simulation      |
|  i   | Toggle Ignition       |
|  q   | Exit all              |
| wasd | Control manually      |
```

## MetaDrive

### Launching Metadrive
Start bridge processes located in openpilot/tools/sim:
``` bash
./run_bridge.py
```
## ZoomEngine

[ZoomEngine](bridge/zoomengine) renders a divided highway loop with dense NPC traffic. openpilot drives it through two
car-mounted sensor cameras (road + wide) that the engine streams over shared memory, with the model's plan, lane lines
and lead drawn on picture-in-picture views of those cameras. The ego car uses openpilot's own vehicle model as its plant.

``` bash
# starts the engine, openpilot and the bridge
TRAFFIC=chaos ./openpilot/tools/sim/launch_zoomengine.sh
```

Traffic presets (`--traffic`):

| preset | cars | what makes it hard |
|--------|------|--------------------|
| light  | 30   | free-flowing traffic |
| heavy  | 78   | aggressive drivers, cut-ins, phantom braking, the car ahead brake-checking every ~45 s, a lane closure with zipper merges |
| chaos  | 124  | everything in heavy, more often and with more aggressive drivers |

The bridge logs a status line every second, every disengagement with its cause and every collision with who caused it
(`ego` rear-ended someone, an `npc` rear-ended ego, or a `cut-in`). If openpilot drops out on its own it is re-engaged
after 3 s, like an attentive driver would.
