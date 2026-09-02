#!/usr/bin/env bash
# Run a test command and emit test_failure to the router when it fails.
# Usage:
#   ./test_failure.sh pytest tests/            # runs the command, reports failure if non-zero
#   ./test_failure.sh --exit-code 1 "pytest"   # report an already-known exit code
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${LLM_ROUTER_SESSION_ID:-}"
TASK="${LLM_ROUTER_TASK_ID:-}"

if [[ "${1:-}" == "--exit-code" ]]; then
  CODE="$2"; CMD="${3:-unknown}"
else
  CMD="$*"
  "$@"; CODE=$?
fi

if [[ "$CODE" != "0" ]]; then
  "$HERE/emit_event.sh" test_failure "$SESSION" "$TASK" "{\"command\": \"$CMD\", \"exit_code\": $CODE}"
fi
exit 0   # hooks must not break the main flow
