#!/usr/bin/env bash
# openpilot driving a car in ZoomEngine: starts the engine's traffic scene, openpilot, and the bridge.
#   ZOOMENGINE_DIR  ZoomEngine checkout with the self-driving work, i.e. main (default ~/Documents/zoom/ZoomEngine-main)
#   TRAFFIC         light | heavy | chaos (default heavy)
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ZOOMENGINE_DIR="${ZOOMENGINE_DIR:-$HOME/Documents/zoom/ZoomEngine-main}"

if [[ ! -x "$ZOOMENGINE_DIR/build/ModernEngine" ]]; then
  echo "building ZoomEngine in $ZOOMENGINE_DIR"
  cmake -S "$ZOOMENGINE_DIR" -B "$ZOOMENGINE_DIR/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build "$ZOOMENGINE_DIR/build" -j"$(sysctl -n hw.ncpu)"
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2> /dev/null || true; done
  wait 2> /dev/null || true
}
trap cleanup EXIT INT TERM

(cd "$ZOOMENGINE_DIR" && exec ./build/ModernEngine traffic) &
PIDS+=($!)

"$DIR/launch_openpilot.sh" > /tmp/openpilot_zoomengine.log 2>&1 &
PIDS+=($!)
echo "openpilot log: /tmp/openpilot_zoomengine.log"

# the bridge reads keyboard input from this terminal
"$DIR/run_bridge.py" --sim zoomengine --dual_camera --traffic "${TRAFFIC:-heavy}"
