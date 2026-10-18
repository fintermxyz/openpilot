#!/usr/bin/env bash
# Town: CARS openpilot-driven cars (default 4, one per loop route), each with its own openpilot (OPENPILOT_PREFIX) and
# bridge, driving round a road-kit town of four signalised crossroads with NPC traffic, pedestrians, buildings and
# street furniture. The town director (bridge/zoomengine/town_director.py) runs signals, traffic and pedestrians.
#   ZOOMENGINE_DIR  ZoomEngine checkout with the road kit (default ~/Documents/zoom/ZoomEngine-main)
#   CARS            number of openpilot cars, 1-4 (default 4)
#   NPCS / PEDS     NPC cars / pedestrians (default 18 / 30)
#   CAR_ROUTES      which routes the cars drive, car 1 first, e.g. C_cw,C_ccw (default: the map's list)
#   MAP_NAME        town (default) or city (tools/roadkit/city_map.py: 9 junctions, a cloverleaf loop, an expressway on bridges)
#   ROADKIT_PROFILE uk (drive on the left) or us (drive on the right); default uk. The map is <MAP_NAME> for town + uk,
#                   otherwise <MAP_NAME>_<profile> (town_us, city_us)
#                   (not PROFILE: openpilot / tinygrad read that as a number)
# Keys in the engine window: R restarts the cars and their openpilots (the engine, its window and settings stay).
# Logs: /tmp/town_*.log
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ZOOMENGINE_DIR="${ZOOMENGINE_DIR:-$HOME/Documents/zoom/ZoomEngine-main}"
ROADKIT_PROFILE="${ROADKIT_PROFILE:-uk}"
MAP_NAME="${MAP_NAME:-town}"
TOWN="${MAP_NAME}_$ROADKIT_PROFILE"; [[ "$MAP_NAME" == "town" && "$ROADKIT_PROFILE" == "uk" ]] && TOWN="town"
MAP="$ZOOMENGINE_DIR/assets/maps/$TOWN.built.json"
CARS="${CARS:-4}"
export TOWN_CAR_ROUTES="${CAR_ROUTES:-}"   # read by the director and every car bridge (town.py TownMap)

if [[ ! -f "$MAP" ]]; then
  if [[ "$MAP_NAME" == "city" ]]; then
    echo "no $MAP: build it with  python3 $ZOOMENGINE_DIR/tools/roadkit/city_map.py --profile $ROADKIT_PROFILE"
  else
    echo "no $MAP: build it with  python3 $ZOOMENGINE_DIR/tools/roadkit/town_map.py --profile $ROADKIT_PROFILE && python3 $ZOOMENGINE_DIR/tools/roadkit/assemble.py $TOWN"
  fi
  exit 1
fi

ENGINE=""
STACK=()   # director, bridges and openpilots: restarted by R in the engine window
kill_tree() {   # a process and everything it started
  local p
  for p in $(pgrep -P "$1" 2> /dev/null); do kill_tree "$p"; done
  kill "$1" 2> /dev/null || true
}
stop_stack() {
  local pid
  for pid in "${STACK[@]}"; do kill_tree "$pid"; done
  sleep 1
  for pid in "${STACK[@]}"; do pkill -9 -P "$pid" 2> /dev/null || true; kill -9 "$pid" 2> /dev/null || true; done
  STACK=()
}
start_stack() {
  "$DIR/bridge/zoomengine/town_director.py" --map "$MAP" --cars "$CARS" --npcs "${NPCS:-18}" --peds "${PEDS:-30}" > /tmp/town_director.log 2>&1 &
  STACK+=($!)
  # the cars add themselves only once the director has built the town: its build starts by clearing the scene (road and
  # ground colliders too), so a car added earlier would fall through the world. A dense city takes ~20 s.
  for ((t = 0; t < 180; t++)); do
    grep -qs "town built" /tmp/town_director.log && break
    sleep 1
  done
  for ((i = 0; i < CARS; i++)); do
    export OPENPILOT_PREFIX="town$i"
    mkdir -p "/tmp/msgq_$OPENPILOT_PREFIX"
    "$DIR/run_bridge.py" --sim town --car "$i" --map "$MAP" > "/tmp/town_bridge$i.log" 2>&1 &
    STACK+=($!)
    # each car only needs the driving stack: no UI, sound, logging or driver monitoring
    BLOCK="ui,soundd" "$DIR/launch_openpilot.sh" > "/tmp/town_openpilot$i.log" 2>&1 &
    STACK+=($!)
    unset OPENPILOT_PREFIX
  done
}
cleanup() {
  stop_stack
  [[ -n "$ENGINE" ]] && kill "$ENGINE" 2> /dev/null || true
  wait 2> /dev/null || true
}
trap cleanup EXIT INT TERM

(cd "$ZOOMENGINE_DIR" && exec ./build/ModernEngine traffic) > /tmp/town_engine.log 2>&1 &
ENGINE=$!
sleep 5
start_stack
echo "town running with $CARS openpilot cars. director: tail -f /tmp/town_director.log   R in the engine window restarts the" \
     "cars and openpilot; Ctrl-C (or closing the engine window) stops everything"

# R in the engine window: a line per (debounced) R tap; the watcher exits when the engine goes away
while read -r _; do
  echo "R: restarting the cars and openpilot (the engine and its settings stay)"
  stop_stack
  start_stack
done < <(ENGINE_PID="$ENGINE" python3 - <<'PY'
import json, os, time, urllib.request
last, quiet_until = None, 0.0
while True:
  try:
    os.kill(int(os.environ["ENGINE_PID"]), 0)
  except OSError:
    break   # the engine window was closed
  try:
    q = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:9090/api", data=b'{"action":"query"}'), timeout=3))
  except Exception:
    time.sleep(1.0)   # a slow or missed reply: keep watching
    continue
  r = q.get("input", {}).get("taps", {}).get("r", 0)
  if last is not None and r > last and time.monotonic() > quiet_until:
    print("r", flush=True)
    quiet_until = time.monotonic() + 20.0   # one restart at a time
  last = r
  time.sleep(0.25)
PY
)
