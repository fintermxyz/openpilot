#!/usr/bin/env bash
# openpilot driving one car round a generated block map (straights, curves, crossroads, T junctions, roundabouts chained
# MetaDrive style; ZoomEngine tools/roadkit/blocks_map.py). Same car, cameras and keys as launch_zoomengine.sh.
#   ZOOMENGINE_DIR  ZoomEngine checkout (default ~/Documents/zoom/ZoomEngine-main)
#   BLOCKS          blocks in the chain, 5 or 7 (default 7)
#   SEED            map seed (default 1); the map is generated unless assets/maps/blocks_<BLOCKS>_s<SEED>.built.json exists
#   EXCLUDE         block letters to leave out, e.g. O for no roundabouts (map blocks_<BLOCKS>_s<SEED>_no<EXCLUDE>)
#   TRAFFIC         light | heavy | chaos (default light): cars ahead of you only
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
export ZOOMENGINE_DIR="${ZOOMENGINE_DIR:-$HOME/Documents/zoom/ZoomEngine-main}"
EXCLUDE="$(echo "${EXCLUDE:-}" | fold -w1 | sort -u | tr -d '\n' | tr a-z A-Z)"
NAME="blocks_${BLOCKS:-7}_s${SEED:-1}${EXCLUDE:+_no$EXCLUDE}"
export MAP="$ZOOMENGINE_DIR/assets/maps/$NAME.built.json"
if [[ ! -f "$MAP" ]]; then
  python3 "$ZOOMENGINE_DIR/tools/roadkit/blocks_map.py" --blocks "${BLOCKS:-7}" --seed "${SEED:-1}" ${EXCLUDE:+--exclude "$EXCLUDE"}
fi
export TRAFFIC="${TRAFFIC:-light}"
exec "$DIR/launch_zoomengine.sh"
