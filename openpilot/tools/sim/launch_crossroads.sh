#!/usr/bin/env bash
# Crossroads: CARS openpilot-driven cars (default 4, one per approach), each with its own openpilot (OPENPILOT_PREFIX) and
# bridge, crossing a UK signalised junction whose signals stick on green both ways during every trial.
#   ZOOMENGINE_DIR  ZoomEngine checkout with the road kit (default ~/Documents/zoom/ZoomEngine-main)
#   CARS            number of openpilot cars, 1-4 (default 4)
#   FAULT_AT        seconds into each trial the signals fail (default 6, negative = never)
#   L3              1: the cars in L3_MASK run the sim-only L3 research layer (default unset: plain openpilot)
#   L3_MASK         comma list of car indices that run L3 when L3=1 (default all, e.g. 0,2)
#   TRIAL_LEN       seconds per trial (default 45, 90 when L3=1: L3 traffic needs ~40-60 s to clear a failed junction)
#   L3_LOG_DIR      L3=1 only: per-car arbiter JSONL logs (a_op, a_out, source, pedals) go here (default: none)
#   TRIALS          stop everything after this many finished trials (default unlimited)
#   SEED            director seed, stored in the results
#   OUT             results JSON (default /tmp/crossroads_results.json)
# Logs: /tmp/crossroads_*.log
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ZOOMENGINE_DIR="${ZOOMENGINE_DIR:-$HOME/Documents/zoom/ZoomEngine-main}"
MAP="$ZOOMENGINE_DIR/assets/maps/crossroads.built.json"
CARS="${CARS:-4}"
# L3 only reaches the bridges of the masked cars, never openpilot itself
L3_ON="${L3:-}"
L3_MASK="${L3_MASK:-}"
unset L3
if [[ "$L3_ON" == "1" && -z "$L3_MASK" ]]; then
  L3_MASK="0"
  for ((i = 1; i < CARS; i++)); do L3_MASK="$L3_MASK,$i"; done
fi
l3_car() {  # is car $1 in L3_MASK
  [[ "$L3_ON" == "1" && ",${L3_MASK// /}," == *",$1,"* ]]
}

if [[ ! -f "$MAP" ]]; then
  echo "no $MAP: build it with  python3 $ZOOMENGINE_DIR/tools/roadkit/assemble.py crossroads"
  exit 1
fi

PIDS=()
tree() {  # a pid and its descendants: a bridge's world process and openpilot's daemons would keep the UDP ports / msgq alive
  local child
  echo "$1"
  for child in $(pgrep -P "$1" 2> /dev/null); do tree "$child"; done
}
cleanup() {
  local all=()
  for pid in "${PIDS[@]}"; do all+=($(tree "$pid")); done
  kill "${all[@]}" 2> /dev/null || true
  wait 2> /dev/null || true
}
trap cleanup EXIT INT TERM

(cd "$ZOOMENGINE_DIR" && exec ./build/ModernEngine traffic) > /tmp/crossroads_engine.log 2>&1 &
ENGINE=$!
PIDS+=($ENGINE)
sleep 5

DIRECTOR_ARGS=(--map "$MAP" --cars "$CARS" --fault-at "${FAULT_AT:-6}")
if [[ -n "${TRIALS:-}" ]]; then DIRECTOR_ARGS+=(--trials "$TRIALS"); fi
if [[ -n "${TRIAL_LEN:-}" ]]; then DIRECTOR_ARGS+=(--trial-len "$TRIAL_LEN"); fi
if [[ -n "${SEED:-}" ]]; then DIRECTOR_ARGS+=(--seed "$SEED"); fi
if [[ -n "${OUT:-}" ]]; then DIRECTOR_ARGS+=(--out "$OUT"); fi
if [[ "$L3_ON" == "1" ]]; then DIRECTOR_ARGS+=(--l3-mask "$L3_MASK"); fi
"$DIR/bridge/zoomengine/crossroads_director.py" "${DIRECTOR_ARGS[@]}" > /tmp/crossroads_director.log 2>&1 &
DIRECTOR=$!
PIDS+=($DIRECTOR)
sleep 3

for ((i = 0; i < CARS; i++)); do
  export OPENPILOT_PREFIX="crossroads$i"
  mkdir -p "/tmp/msgq_$OPENPILOT_PREFIX"
  # openpilot refuses to start after an "excessive actuation" alert until the driver acknowledges it in the UI, which these
  # headless cars don't run: report it and acknowledge it the way the UI's button does (Params.remove), the check itself stays
  python3 -c "
from openpilot.common.params import Params
p = Params()
a = p.get('Offroad_ExcessiveActuation')
if a is not None:
  print('car $i: openpilot flagged excessive actuation on its last drive:', a.get('extra', a) if isinstance(a, dict) else a, '- acknowledged')
  p.remove('Offroad_ExcessiveActuation')
" | tee -a /tmp/crossroads_excessive_actuation.log
  if l3_car "$i"; then
    L3=1 "$DIR/run_bridge.py" --sim crossroads --car "$i" --map "$MAP" > "/tmp/crossroads_bridge$i.log" 2>&1 &
  else
    "$DIR/run_bridge.py" --sim crossroads --car "$i" --map "$MAP" > "/tmp/crossroads_bridge$i.log" 2>&1 &
  fi
  PIDS+=($!)
  # each car only needs the driving stack: no UI, sound, logging or driver monitoring
  BLOCK="ui,soundd" "$DIR/launch_openpilot.sh" > "/tmp/crossroads_openpilot$i.log" 2>&1 &
  PIDS+=($!)
  unset OPENPILOT_PREFIX
done

L3_MSG=""
if [[ "$L3_ON" == "1" ]]; then L3_MSG=", L3 on cars $L3_MASK"; fi
echo "crossroads running with $CARS openpilot cars$L3_MSG. director: tail -f /tmp/crossroads_director.log"
echo "Ctrl-C (or closing the engine window) stops everything"
if [[ -n "${TRIALS:-}" ]]; then
  # everything goes down with the engine window, or once the director has run its trials (bash 3.2: no wait -n)
  while kill -0 "$ENGINE" 2> /dev/null && kill -0 "$DIRECTOR" 2> /dev/null; do sleep 1; done
else
  # everything goes down with the engine window
  wait "$ENGINE" || true
fi
