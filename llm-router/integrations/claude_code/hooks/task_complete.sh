#!/usr/bin/env bash
# Stop hook: emit task_complete when the agent finishes its turn.
# Resets per-task failure counters so a later hard subtask starts fresh.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION=$(cat | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("session_id",""))
except Exception: print("")')

"$HERE/emit_event.sh" task_complete "$SESSION" ""
exit 0
