# llm-router

A small, reliable **local-first LLM routing & escalation service** for use with Claude Code.

It sits between your agent and three model tiers — a fast local model, a stronger local model, and an optional cloud (OpenRouter) frontier model — and routes each request to the right one:

```text
                         Claude Code / any OpenAI client
                              │  OpenAI-compatible API :8000
                              ▼
                 ┌──────────────────────────┐
                 │        llm-router        │
                 │  Router + Controller     │
                 │  Session/task state      │
                 │  Escalation policy       │
                 │  Health / metrics        │
                 └────────────┬─────────────┘
             ┌────────────────┼─────────────────┐
             ▼                ▼                 ▼
        LOCAL FAST       LOCAL DEEP        OPENROUTER (frontier)
        :8001            :8002              HTTPS
```

Design goals: **simple enough to understand and debug**, deterministic escalation (no LLM judging difficulty), transparent proxying (no prompt rewriting), and hard cloud-cost safeguards. See the PRD, *Local LLM Routing & Escalation System*, for the full specification; this README is the operator's guide.

---

## Tiers

| Tier | Alias | Typical model | Backend |
|------|-------|---------------|---------|
| fast | `local-fast` | 12–16B | llama-server :8001 (single GPU) |
| deep | `local-deep` | ~30–40B | llama-server :8002 (dual GPU) |
| frontier | `frontier` | any OpenRouter model | https://openrouter.ai/api/v1 |

Plus the special alias **`auto`**, which activates automatic routing/escalation.

The router is completely model- and GPU-agnostic: it only talks to OpenAI-compatible HTTP endpoints. Both local models are expected to stay running (no model loading/unloading in v1).

## Quick start

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure
cp config.example.yaml config.yaml      # edit backends, model names, limits
cp .env.example .env                    # set OPENROUTER_API_KEY (or export it)

# 3. Run
llm-router --config config.yaml         # or: python -m router --config config.yaml

# 4. Smoke test
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-fast","messages":[{"role":"user","content":"hi"}]}'
```

CLI options: `--config PATH` (default `config.yaml`), `--host`, `--port`, `--log-level`.

## API

OpenAI-compatible, so any OpenAI client works by pointing its base URL at the router.

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `POST /v1/messages` | Anthropic Messages API — run Claude Code through the router directly (PRD §51) |
| `GET  /v1/models` | Logical model aliases (`auto`, `local-fast`, `local-deep`, `frontier`) |
| `GET  /health` | Overall status + per-backend health/latency |
| `GET  /health/backends` | Per-backend health only |
| `GET  /metrics` | Prometheus text format |
| `POST /events` | Lifecycle events from Claude Code hooks (see below) |

### Request headers

```http
X-LLM-Session-ID: <id>     # groups requests into a session (generated if absent)
X-LLM-Task-ID:    <id>     # one unit of work; escalation state is per-task
X-LLM-Escalate:   deep|frontier   # rule A: force this tier now (subject to cloud limits)
X-LLM-Route:      fast|deep|frontier|auto  # direct routing for this request only
```

Routing priority: **escalation header → route header → explicit model → session/task route → auto policy → configured default**.

### Responses

Every response carries an `x_router` block with the metadata you need to observe and act on routing decisions:

```json
{
  "id": "...", "object": "chat.completion", "choices": [ ... ],
  "x_router": {
    "request_id": "abc123",
    "session_id": "sess-...",
    "task_id": "task-...",
    "route": "deep",
    "backend": "local-deep",
    "reason": "auto_policy",
    "escalation": { "from": "fast", "to": "deep", "reason": "repeated_test_failure", "attempts": 2 }
  }
}
```

`escalation` is present when the task's route moved (or, for blocked cloud calls, as `{"required": true, "to": "frontier", "reason": "..."}`). For streaming responses the same block arrives in a final SSE chunk just before `data: [DONE]`.

### Escalation-required response

When automatic cloud escalation is disabled or a budget limit is hit, the router returns HTTP 200 with a valid completion whose content starts with `[llm-router] escalation required:` and whose `x_router.escalation.required` is `true` — so OpenAI-compatible clients don't crash while still being able to detect the situation (e.g. via a hook) and retry with `X-LLM-Escalate: frontier`.

### Claude Code (`POST /v1/messages`, PRD §51)

Claude Code speaks the Anthropic Messages API; this endpoint translates it to OpenAI chat-completions on the way in and back on the way out — including streaming (Anthropic SSE events), tool calls, and usage accounting. Point Claude Code at the router and its whole agent loop runs through routing/escalation:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_API_KEY=local
claude --model auto        # or local-fast / local-deep / frontier
```

Notes:

- `x_router` arrives in the response body (non-streaming) and inside `message_start` (streaming).
- Session identity falls back to Claude Code's own `X-Claude-Code-Session-ID` header when `X-LLM-Session-ID` is absent, so escalation state works without extra configuration.
- The model name `auto` is not in Claude Code's built-in catalog; set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to your real window (e.g. the deep tier's) to silence the unknown-model notice and size auto-compact correctly.
- Real prompt/completion token counts are requested from llama.cpp (`stream_options.include_usage`) and reported in `message_delta.usage`; `message_start` carries a chars-per-token estimate as its seed.

## Automatic routing & escalation

`auto` always starts on `routing.default` (`fast`). It escalates only on **deterministic signals** — no LLM is invoked to judge difficulty, and the model never has to know it's struggling:

| Rule | Signal | Default threshold |
|------|--------|-------------------|
| A | Explicit request (`X-LLM-Escalate` header or `explicit_escalation` event) | immediate |
| B | Repeated **test** failures within one task, on the current tier | 2 → next tier |
| C | Repeated **tool** failures within one task, on the current tier | 2 → next tier |
| D | Retry limit: `max_*_attempts` requests to the current tier **and** ≥1 recorded failure signal | fast: 2, deep: 2 |

Chain: `fast → deep → frontier` (configurable). Failure counters reset when a task escalates or completes. Each rule can be switched off under `routing.escalation.signals`.

**Context floor.** Independent of complexity, every request's estimated context need (prompt + completion headroom) is checked against the selected tier's `max_context` (per backend in config). If it doesn't fit, the request is bumped up the chain until it does — even over an explicitly requested tier, since that backend would reject the prompt anyway. This is per-request and does not change task state; long conversations move to the bigger window automatically. The estimate is a conservative chars-per-token heuristic over the serialized body (`routing.context.chars_per_token`, default 3) plus `max_tokens` if given, else `routing.context.completion_reserve`. Tiers without a configured `max_context` are assumed to fit. Bumps show up as `x_router.reason: context_overflow` and count in `router_escalations_total`.

**Queried context sizes (PRD §52).** A manual `max_context` can go stale when the server is restarted with different arguments. Set `query_context_size: true` on a backend and the router asks it at startup for its real window — llama-server reports the effective `--ctx-size` via `GET /props` (`default_generation_settings.n_ctx`) — and uses that value instead of the config one (logged as `using queried context size N`). Backends without `/props` or failed queries keep the manual value.

**Backend failure ≠ model failure.** A crashed llama-server (connection refused, 5xx, timeout) does *not* mean "the model was too weak". Backend failures go through the configured **fallback policy** (`routing.fallbacks`) and do not change task state — unless you explicitly enable the `timeout`/`backend_error` signals.

## Cloud cost protection

The frontier tier is guarded by hard limits:

- `cloud.enabled` — master switch (off = even manual frontier requests are blocked)
- `cloud.allow_automatic_escalation` — **opt-in**; recommended `false` for the first deployment
- `cloud.max_requests_per_hour` — rolling hourly request cap
- `cloud.max_estimated_cost_usd_per_day` — daily cost estimate from response `usage` × `cloud.pricing` (set prices for your OpenRouter model)
- `cloud.allow_manual_when_limited` — explicit `X-LLM-Escalate: frontier` still works at the limit

When a limit blocks an *automatic* call you get the escalation-required response above; no uncontrolled billing loop.

## Claude Code integration

See [`integrations/claude_code/README.md`](integrations/claude_code/README.md) for the full guide, including:

- pointing Claude Code at the router (direct or via a translation proxy),
- an installable **CLAUDE.md escalation policy** (`integrations/claude_code/CLAUDE.md`),
- hook scripts that emit `test_failure` / `tool_failure` events to `/events`,
- the `escalation.sh` helper for calling deeper tiers from shell/subagents.

The intended workflow: the main agent works on `local-fast`; when a subtask is hard it delegates to a **deep subagent** (a fresh request with `X-LLM-Escalate: deep` or `model: local-deep`) instead of restarting the whole conversation at another model; frontier is the last resort.

## Events (`POST /events`)

```json
{ "session_id": "abc", "task_id": "task-123", "event": "test_failure",
  "metadata": { "command": "pytest", "exit_code": 1 } }
```

Supported events: `task_start`, `task_complete`, `task_failure`, `tool_failure`, `test_failure`, `explicit_escalation` (optional `metadata.target`). Unknown events are accepted and logged at debug level. The response reports the resulting route:

```json
{ "status": "ok", "route": "deep", "escalated": true, ... }
```

## Observability

- One structured log line per request: `request=… session=… task=… route=… backend=… latency=… status=…`
- Escalations logged as `ESCALATION session=… task=… from=fast to=deep reason=repeated_test_failure count=2`
- Prometheus metrics at `/metrics`: `router_requests_total`, `router_request_latency_seconds`, `router_backend_requests_total`, `router_backend_errors_total`, `router_escalations_total`, `router_cloud_requests_total`, `router_cloud_blocked_total`, `router_tool_failures_total`, `router_test_failures_total` (low-cardinality labels; never raw session IDs)
- Prompt/response logging is **off by default** (`logging.log_prompts` / `log_responses`)

## Security

Binds to `127.0.0.1` by default. API keys live in environment variables (named via `api_key_env`), are read lazily, and are never printed or logged — header dumps mask `Authorization`/`X-Api-Key`.

## Testing

The whole suite runs with **no GPU, no llama-server, no OpenRouter, no Claude Code** — every backend is a mock:

```bash
pip install -e ".[dev]"
pytest            # 60+ tests across routing, escalation, sessions, streaming, tools, backends, cloud limits
```

## Project layout

```text
llm-router/
├── pyproject.toml
├── config.example.yaml      .env.example   LICENSE   README.md
├── router/                  # the service (see PRD §35)
│   ├── api.py               # FastAPI app + request flow
│   ├── routing.py           # Router, route priority, CloudBudget
│   ├── escalation.py        # EscalationController (rules A–D)
│   ├── sessions.py          # in-memory session/task state
│   ├── events.py            # POST /events handling
│   ├── config.py  models.py metrics.py server.py __main__.py
│   └── backends/            # base, openai_compatible, openrouter, mock
├── integrations/claude_code/# README, CLAUDE.md policy, escalation.sh, hooks/
├── deploy/llm-router.service# example systemd unit
├── docs/backends.md         # llama-server startup commands
└── tests/                   # full mock-based suite
```

## Not in v1 (deliberately)

AI request classification, prompt/response rewriting, RAG/vector DBs, web search, GUI, database, conversation summarization, GPU/model lifecycle management, automatic llama-server restarts. See PRD §41–44 for the future roadmap.
