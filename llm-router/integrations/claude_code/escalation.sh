#!/usr/bin/env bash
# escalation.sh — call a deeper model tier from shell scripts, hooks, or Claude Code's Bash tool.
#
# Usage:
#   ./escalation.sh --route deep    --prompt "why does this deadlock?"
#   ./escalation.sh --escalate frontier --prompt "prove the invariant"
#   ./escalation.sh --session <id> --task <id> --stream --prompt "..."
#   ./escalation.sh --raw ...       # print full JSON instead of just content
#
# Env:
#   LLM_ROUTER_URL  (default http://127.0.0.1:8000)
set -euo pipefail

URL="${LLM_ROUTER_URL:-http://127.0.0.1:8000}"
ROUTE=""
ESCALATE=""
SESSION="${LLM_ROUTER_SESSION_ID:-}"
TASK="${LLM_ROUTER_TASK_ID:-}"
PROMPT=""
STREAM=0
RAW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --route)    ROUTE="$2"; shift 2 ;;
    --escalate) ESCALATE="$2"; shift 2 ;;
    --session)  SESSION="$2"; shift 2 ;;
    --task)     TASK="$2"; shift 2 ;;
    --prompt)   PROMPT="$2"; shift 2 ;;
    --stream)   STREAM=1; shift ;;
    --raw)      RAW=1; shift ;;
    -h|--help)  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$PROMPT" ]] || { echo "--prompt is required" >&2; exit 2; }
if [[ -z "$ROUTE" && -z "$ESCALATE" ]]; then ROUTE="auto"; fi

# Build the JSON body with python3 (no jq dependency).
BODY=$(python3 - "$PROMPT" "$STREAM" <<'PY'
import json, sys
prompt = sys.argv[1]
stream = sys.argv[2] == "1"
print(json.dumps({
    "model": "auto",
    "stream": stream,
    "messages": [{"role": "user", "content": prompt}],
}))
PY
)

HDRS=(-H 'Content-Type: application/json')
[[ -n "$SESSION" ]] && HDRS+=(-H "X-LLM-Session-ID: $SESSION")
[[ -n "$TASK"    ]] && HDRS+=(-H "X-LLM-Task-ID: $TASK")
if [[ -n "$ESCALATE" ]]; then
  HDRS+=(-H "X-LLM-Escalate: $ESCALATE")
elif [[ -n "$ROUTE" && "$ROUTE" != "auto" ]]; then
  HDRS+=(-H "X-LLM-Route: $ROUTE")
fi

if [[ "$RAW" == "1" || "$STREAM" == "1" ]]; then
  curl -sS "${HDRS[@]}" "$URL/v1/chat/completions" -d "$BODY"
else
  # Non-stream, non-raw: extract the assistant content.
  RESP=$(curl -sS --fail-with-body "${HDRS[@]}" "$URL/v1/chat/completions" -d "$BODY") || exit 1
  printf '%s\n' "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
fi
