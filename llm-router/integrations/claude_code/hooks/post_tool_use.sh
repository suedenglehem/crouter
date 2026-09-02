#!/usr/bin/env bash
# PostToolUse hook: emit tool_failure when a tool call failed.
# Claude Code pipes the hook payload (JSON) on stdin; it includes session_id,
# tool_name and tool_response (with exit_code for Bash).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD=$(cat)

read -r SESSION TOOL EXIT_CODE ERROR <<EOF2
$(printf '%s' "$PAYLOAD" | python3 -c '
import json, sys
try:
    p = json.load(sys.stdin)
except ValueError:
    print("","","",""); sys.exit(0)
resp = p.get("tool_response") or {}
exit_code = resp.get("exit_code", "")
error = "1" if (p.get("error") or resp.get("is_error")) else ""
print(p.get("session_id", ""), p.get("tool_name", ""), exit_code, error)
')
EOF2

# Failed when: explicit error flag, or a numeric non-zero exit code.
if [[ -n "$ERROR" ]] || { [[ "$EXIT_CODE" =~ ^[0-9]+$ ]] && [[ "$EXIT_CODE" != "0" ]]; }; then
  "$HERE/emit_event.sh" tool_failure "$SESSION" "" "{\"tool\": \"$TOOL\", \"exit_code\": ${EXIT_CODE:-null}}"
fi
exit 0
