#!/usr/bin/env bash
# guardrail (PLAN §1.3): the L3 research layer must not touch openpilot core, opendbc or panda.
# git diff --exit-code (working tree and index) against HEAD's merge base with master; prints OK when there is no diff
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
BASE="$(git merge-base HEAD master 2>/dev/null || git merge-base HEAD origin/master)"
PATHS=(openpilot/selfdrive openpilot/cereal openpilot/system openpilot/common opendbc_repo panda)

fail=0
git diff --exit-code --stat "$BASE" -- "${PATHS[@]}" || fail=1           # working tree vs base
git diff --cached --exit-code --stat "$BASE" -- "${PATHS[@]}" || fail=1  # index vs base
if [ "$fail" -ne 0 ]; then
  echo "FAIL: core files differ from $BASE (L3 changes must stay in openpilot/tools/sim)" >&2
  exit 1
fi
echo "OK"
