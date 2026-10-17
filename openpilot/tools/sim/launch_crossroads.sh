#!/usr/bin/env bash
# Crossroads: CARS openpilot-driven cars (default 4, one per approach), each with its own openpilot (OPENPILOT_PREFIX) and
# bridge, crossing a UK signalised junction whose signals stick on green both ways during every trial.
#   ZOOMENGINE_DIR  ZoomEngine checkout with the road kit (default ~/Documents/zoom/ZoomEngine-main)
#   CARS            number of openpilot cars, 1-4 (default 4)
#   FAULT_AT        seconds into each trial the signals fail (default 6, negative = never)
# Results of every trial: /tmp/crossroads_results.json. Logs: /tmp/crossroads_*.log
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ZOOMENGINE_DIR="${ZOOMENGINE_DIR:-$HOME/Documents/zoom/ZoomEngine-main}"
MAP="$ZOOMENGINE_DIR/assets/maps/crossroads.built.json"
CARS="${CARS:-4}"

if [[ ! -f "$MAP" ]]; then
  echo "no $MAP: build it with  python3 $ZOOMENGINE_DIR/tools/roadkit/assemble.py crossroads"
  exit 1
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2> /dev/null || true; done
  wait 2> /dev/null || true
}
trap cleanup EXIT INT TERM

(cd "$ZOOMENGINE_DIR" && exec ./build/ModernEngine traffic) > /tmp/crossroads_engine.log 2>&1 &
ENGINE=$!
PIDS+=($ENGINE)
sleep 5

"$DIR/bridge/zoomengine/crossroads_director.py" --map "$MAP" --cars "$CARS" --fault-at "${FAULT_AT:-6}" > /tmp/crossroads_director.log 2>&1 &
PIDS+=($!)
sleep 3

for ((i = 0; i < CARS; i++)); do
  export OPENPILOT_PREFIX="crossroads$i"
  mkdir -p "/tmp/msgq_$OPENPILOT_PREFIX"
  "$DIR/run_bridge.py" --sim crossroads --car "$i" --map "$MAP" > "/tmp/crossroads_bridge$i.log" 2>&1 &
  PIDS+=($!)
  # each car only needs the driving stack: no UI, sound, logging or driver monitoring
  BLOCK="ui,soundd" "$DIR/launch_openpilot.sh" > "/tmp/crossroads_openpilot$i.log" 2>&1 &
  PIDS+=($!)
  unset OPENPILOT_PREFIX
done

echo "crossroads running with $CARS openpilot cars. director: tail -f /tmp/crossroads_director.log   Ctrl-C (or closing the engine window) stops everything"
# everything goes down with the engine window
wait "$ENGINE" || true
