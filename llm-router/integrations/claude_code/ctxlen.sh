#!/usr/bin/env bash
# ctxlen.sh — set/reset the router's context-length bounds at runtime (no restart).
#
# Usage:
#   ./ctxlen.sh fast=32000     # fast tier bound -> 32000 tokens
#   ./ctxlen.sh deep=reset     # deep back to its startup value (config or queried size)
#   ./ctxlen.sh                # show current + initial bounds for all tiers
#
# The context floor reads the bound live per request, so a change applies from
# the very next routed request. In-memory only: restarting the router reverts
# to the configuration file.
#
# Env:
#   LLM_ROUTER_URL  (default http://127.0.0.1:8000)
set -euo pipefail

URL="${LLM_ROUTER_URL:-http://127.0.0.1:8000}"
SPEC="$*"

if [[ -z "$SPEC" ]]; then
  curl -sS "$URL/ctxlen"; echo
else
  curl -sS --fail-with-body "$URL/ctxlen/$SPEC"; echo
fi
