# Troubleshooting

## Nothing reaches my model — where do I look?

Order of checks:

1. `curl -s http://127.0.0.1:8000/health` — is the router up, and which backends are healthy?
2. For an unhealthy backend: hit it directly (`curl http://127.0.0.1:8001/v1/models`). If that fails, the problem is llama-server, not the router.
3. Router logs (journald or terminal): every request gets a line like
   `request=… session=… task=… route=fast backend=local-fast latency=4.82s status=200`.
   A 502 with `"type": "backend_error"` means the router reached its fallback policy and gave up.

## Backend failure vs model failure (the most common confusion)

- **Backend failure** = connection refused, timeout, HTTP 5xx from llama-server. The router treats this as *infrastructure*: it applies `routing.fallbacks` (or returns 502). It does **not** escalate the task — a crashed server is not "the model was too weak" (PRD 12).
- **Model/task failure** = repeated test/tool failures reported via `/events`, or explicit escalation. This moves the task up the chain.

If you see requests falling back to `deep` constantly, fix the fast backend first — check VRAM (`nvidia-smi`) and llama-server logs before touching routing config.

## Requests keep hitting the cloud when I don't want that

Check, in order:

1. `cloud.allow_automatic_escalation` — must be `false` to block automatic cloud calls (recommended for first deployment).
2. `routing.escalation.signals` — which signals can trigger escalation at all.
3. `X-LLM-Escalate` / `X-LLM-Route: frontier` headers in your client/hook scripts are *manual* and bypass the automatic gate by design (`cloud.allow_manual_when_limited`).
4. Watch `router_cloud_requests_total` and `router_cloud_blocked_total` at `/metrics`.

## I got an "escalation required" response

HTTP 200, content starts with `[llm-router] escalation required:`, and
`x_router.escalation.required == true`. The task reached the frontier tier but a cloud limit blocked it (`reason` tells you which: `cloud_auto_disabled`, `cloud_rate_limit`, `cloud_cost_limit`, `cloud_disabled`). Options: retry with `X-LLM-Escalate: frontier` (manual), raise the limits, or let the client/hook decide.

## Streaming looks buffered / my client hangs

- The router proxies SSE line-by-line and appends one extra metadata chunk before `data: [DONE]`. Strict OpenAI SDKs tolerate unknown fields; if your client chokes on it, read the last non-DONE data event for `x_router` instead of ignoring extras.
- Behind a reverse proxy, disable response buffering (e.g. nginx `proxy_buffering off`; the router already sends `X-Accel-Buffering: no`).

## Escalation never happens / escalates too eagerly

- Never: make sure hooks actually reach `/events` (`curl -s localhost:8000/events -d ...`), that `session_id`/`task_id` match between events and requests, and that the relevant signal is enabled in `routing.escalation.signals`.
- Too eager: raise `failure_threshold`, or note that rule D (retry limit) only fires when at least one failure signal was recorded — pure request volume alone never escalates a task.

## Session state seems to have vanished

State is in-memory by design and idle sessions are swept after an hour of no activity. Restarting the router also clears it. That's fine for v1: worst case, a task re-escalates from scratch.

## API key issues

- Key comes from the environment variable named by `api_key_env` (default `OPENROUTER_API_KEY`). It is read per request — you can rotate it without restarting.
- If frontier requests get HTTP 401: verify with `env | grep OPENROUTER` in the router's process environment (`systemctl show llm-router -p Environment`, or check your `EnvironmentFile`).
- The key never appears in logs; if you suspect a leak, enable DEBUG and look for unmasked `Authorization` values (header dumps mask them).

## Port conflicts / binding

The router binds `127.0.0.1:8000` by default. If something else owns the port, use `--port` or change `server.port`. Don't bind `0.0.0.0` without adding auth (PRD 39).
