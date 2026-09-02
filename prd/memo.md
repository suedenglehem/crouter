# llm-router — session memo (2026-09-02)

**Status: COMPLETE.** The PRD ("Local LLM Routing & Escalation System") is fully implemented in `llm-router/` and verified. 84/84 tests pass; live smoke test passed.

## Where things are

```
/dd2/andrei/crouter/
├── Local LLM Routing & Escalation System.md   # the PRD
├── memo.md                                    # this file
└── llm-router/                                # the implementation (41 files)
    ├── router/            # FastAPI service: api, routing+CloudBudget, escalation A–D,
    │                      # sessions, events, metrics (hand-rolled Prometheus), backends/
    ├── tests/             # 8 test files + conftest — all mock-backed, no GPU/network needed
    ├── config.example.yaml  .env.example  LICENSE  README.md
    ├── integrations/claude_code/   # CLAUDE.md policy, escalation.sh, hooks/, settings example
    ├── deploy/llm-router.service   # systemd unit
    └── docs/              # backends.md (llama-server commands), troubleshooting.md
```

## Run it

```bash
cd /dd2/andrei/crouter/llm-router
source .venv/bin/activate        # venv already exists with all deps installed
pytest                           # 84 passed in ~0.6s
cp config.example.yaml config.yaml   # then edit model names/pricing
export OPENROUTER_API_KEY=...    # only needed for the frontier tier
llm-router --config config.yaml  # serves on 127.0.0.1:8000
```

## What it does (one paragraph)

OpenAI-compatible proxy on :8000 in front of three tiers — `local-fast` (llama-server :8001), `local-deep` (:8002, dual GPU), `frontier` (OpenRouter). Requests start on fast and escalate deterministically (no LLM judging difficulty): rule A explicit request (`X-LLM-Escalate` header or event), B repeated test failures, C repeated tool failures, D retry limit with ≥1 recorded failure. Backend crashes use the fallback policy instead of escalating. Cloud is cost-guarded: automatic escalation is opt-in (`cloud.allow_automatic_escalation`, default false in the example config), plus hourly rate and daily cost limits; blocked calls return a valid 200 completion whose content starts `[llm-router] escalation required:`. Every response carries an `x_router` block (request/session/task/route/reason/escalation); streaming gets it as a final SSE chunk before `[DONE]`.

## Key design decisions (if you want to change something)

- **Mock backends** make the whole suite GPU/network-free; behaviors: success/stream/tool_call/error/timeout/echo.
- Rule D requires ≥1 failure signal, so long healthy conversations never auto-escalate on attempt count alone.
- Explicit/manual requests don't consume the retry budget and bypass `allow_automatic_escalation` (but not rate/cost limits unless `allow_manual_when_limited`).
- API keys come from env vars named in config (`api_key_env`), read lazily, never logged; header dumps mask `Authorization`/`X-Api-Key`.
- State is in-memory by design (PRD: no database in v1); lost on restart.

## Fixes made during verification (so you don't re-chase them)

- cloud cost/rate recording was missing for non-streaming frontier responses — added
- streaming path crashed if a backend answered with plain JSON (`result.stream is None`) — guarded
- mock `echo` now returns the model-rewritten payload (what was actually forwarded)
- rule A now honors `signals.explicit_request: false`
- httpx 0.28 quirk: `MockTransport` no longer enforces client timeouts, so the timeout-kind test raises `ReadTimeout` directly instead of sleeping

## Not done / next steps

- Nothing blocking. Optional: real llama-server + OpenRouter integration pass; `git init` (the dir is not a repo yet); deploy via `deploy/llm-router.service`.
- PRD §41 "not in v1" list was deliberately honored: no RAG, GUI, database, model loading/unloading, GPU management.
