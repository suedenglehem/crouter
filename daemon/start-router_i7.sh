#!/usr/bin/env bash
# Start llm-router (OpenAI-compatible proxy) on 127.0.0.1:8000 using config.yaml.
#
#   ./start-router.sh          start in background, wait for /health
#   ./start-router.sh --fg     run in foreground (Ctrl-C to stop)
#   ./start-router.sh status   show pid + health
#   ./start-router.sh -c       check fast/deep/frontier answer real requests
#   ./start-router.sh stop     stop the background instance
set -euo pipefail

ROUTER_DIR=/dd2/andrei/crouter
DIR="$ROUTER_DIR/llm-router"
CONFIG="$DIR/config_i7.yaml"
LOG=/tmp/llm-router.log
PIDFILE=/tmp/llm-router.pid
URL=http://127.0.0.1:8000

port_up() { curl -sf "$URL/health" >/dev/null 2>&1; }

case "${1:-start}" in
  start)
    if port_up; then
      echo "already running at $URL — log: $LOG"
      exit 0
    fi
    rm -f "$PIDFILE"   # stale from a previous run
    cd "$DIR"
    nohup .venv/bin/llm-router --config "$CONFIG" >>"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    echo "started llm-router (pid $(cat "$PIDFILE")), log: $LOG"

    for _ in $(seq 1 30); do
      if curl -sf "$URL/health" >/dev/null; then
        echo "ready at $URL"
        exit 0
      fi
      sleep 1
    done
    echo "not ready after 30s — check $LOG" >&2
    tail -n 20 "$LOG" >&2 || true
    exit 1
    ;;

  --fg)
    cd "$DIR"
    exec .venv/bin/llm-router --config "$CONFIG"
    ;;

  status)
    if port_up; then
      echo "running at $URL"
      curl -sf "$URL/health"; echo
    else
      echo "not running (no response on :8000)"
      exit 1
    fi
    ;;

  -c|--chk-ready)
    "$ROUTER_DIR/llms-ready.py" --config "$CONFIG" "${@:2}"
    ;;

  stop)
    pid=""
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      pid=$(cat "$PIDFILE")
    else
      # started outside this script — find it by command line
      pid=$(pgrep -f "\.venv/bin/llm-router --config" | head -n1 || true)
    fi
    if [[ -n "$pid" ]]; then
      kill "$pid" && echo "stopped (pid $pid)"
    else
      echo "not running"
    fi
    rm -f "$PIDFILE"
    ;;

  *)
    echo "usage: $0 [start|--fg|status|-c|--chk-ready|stop]" >&2
    exit 2
    ;;
esac
