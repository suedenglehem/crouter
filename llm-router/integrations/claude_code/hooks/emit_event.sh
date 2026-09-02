#!/usr/bin/env bash
# emit_event.sh <event> [session_id] [task_id] [metadata_json]
# Shared helper: POST one lifecycle event to the router's /events endpoint.
# Never fails the caller (hooks must not break the main flow).
set -uo pipefail

URL="${LLM_ROUTER_URL:-http://127.0.0.1:8000}"
EVENT="${1:?usage: emit_event.sh <event> [session_id] [task_id] [metadata_json]}"
SESSION="${2:-${LLM_ROUTER_SESSION_ID:-}}"
TASK="${3:-}"
META="${4:-{}}"

python3 - "$EVENT" "$SESSION" "$TASK" "$META" <<'PY' | curl -sS --max-time 5 -X POST "${URL}/events" -H 'Content-Type: application/json' --data @- || echo "emit_event: router unreachable (event=$EVENT)" >&2
import json, sys

event, session, task, meta = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
payload = {"event": event}
if session:
    payload["session_id"] = session
if task:
    payload["task_id"] = task
try:
    payload["metadata"] = json.loads(meta) if meta and meta != "{}" else {}
except ValueError:
    pass
print(json.dumps(payload))
PY
