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

## Live on this machine (2026-09-07)

Running at `127.0.0.1:8000` with `llm-router/config.yaml` (created from the example). Real backends instead of mocks; OpenRouter not used yet:

| tier | model | llama-server | started by |
|------|-------|--------------|------------|
| fast | Qwen3.5-9B-Claude-HighIQ | :8081 (GPU2) | `/dd2/llama-server/qw9claude.sh` |
| deep | Qwen3.8-27B-Uncensored | :8080 (dual 3090, MTP draft) | `/dd2/llama-server/qw_uncensored_mtp_q8_claude.sh` |

Config notes: chain is `[fast, deep, frontier]` (frontier is a placeholder — OpenRouter not wired up yet), `cloud.enabled: false`; each backend's `model:` is the **full GGUF path** llama-server reports at `/v1/models` (the router rewrites every request's model field to it). Verified live: health on both tiers, non-streaming + streaming completions, `auto`→fast, rule-B escalation via `POST /events` (2× test_failure → deep), metrics. Start command: `.venv/bin/llm-router --config config.yaml`.

**Context floor added (2026-09-07).** Routing now also switches on context length, not just complexity: each backend has a `max_context` (fast 32768, deep 256000); a request whose estimated size (serialized body / `chars_per_token`=3 + client `max_tokens` or `completion_reserve`=8192) exceeds the selected tier's limit is bumped up the chain until it fits — even over an explicit model choice. Per-request, no task-state change; shows as `x_router.reason: context_overflow`, counts in `router_escalations_total`. Code: `router/context.py` (estimator), `Router._apply_context_floor` in `routing.py`; config under `routing.context:`. 97/97 tests pass (`tests/test_context_routing.py` covers it). Live-verified: ~120KB body → deep, ~900KB body → frontier placeholder → clean "escalation required (cloud_disabled)" response since cloud is off. To enable real overflow-to-cloud later: set the frontier `model`, export `OPENROUTER_API_KEY`, flip `cloud.enabled` to true.
