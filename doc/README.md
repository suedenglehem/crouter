# llm-router — How It Works (Deep Dive)

This is the in-depth companion to the [root README](../README.md), which stays a short operator's guide. Here: every component, every decision rule, every config key, and how a request actually flows through the system — plus ideas for what could be added next.

---

## Table of contents

1. [What it is](#1-what-it-is)
2. [Design principles](#2-design-principles)
3. [Architecture & file map](#3-architecture--file-map)
4. [Request lifecycle (step by step)](#4-request-lifecycle-step-by-step)
5. [Tiers, aliases & model rewriting](#5-tiers-aliases--model-rewriting)
6. [The routing decision — full priority order](#6-the-routing-decision--full-priority-order)
7. [The context floor (size-based bumping)](#7-the-context-floor-size-based-bumping)
8. [The complexity classifier (difficulty-based routing)](#8-the-complexity-classifier-difficulty-based-routing)
9. [Prompt markers (@@fast / @@deep / @@frontier)](#9-prompt-markers)
10. [Escalation engine — rules A–D](#10-escalation-engine--rules-ad)
11. [Sessions & tasks (the state model)](#11-sessions--tasks-the-state-model)
12. [Backends: implementations, health, fallbacks](#12-backends-implementations-health-fallbacks)
13. [The Anthropic endpoint (/v1/messages)](#13-the-anthropic-endpoint-v1messages)
14. [Cloud cost protection](#14-cloud-cost-protection)
15. [Admin & observability endpoints](#15-admin--observability-endpoints)
16. [x_router reference (reasons catalog)](#16-x_router-reference-reasons-catalog)
17. [Configuration reference](#17-configuration-reference)
18. [Claude Code integration details](#18-claude-code-integration-details)
19. [Testing strategy](#19-testing-strategy)
20. [Operations notes](#20-operations-notes)
21. [Ideas & future extensions](#21-ideas--future-extensions)

---

## 1. What it is

llm-router is a small, local-first **routing and escalation service** that sits between an agent (Claude Code or any OpenAI-compatible client) and three model tiers:

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
                 └───────┬──────────┬───────┘
                         │          │ complexity verdict (opt-in):
            route request│          │ fast model judges each new user
                         │          │ message once, cached per session
             ┌───────────┼──────────┴─────────────────┐
             ▼           ▼                            ▼
        LOCAL FAST    LOCAL DEEP              OPENROUTER (frontier)
        :8001         :8002                    HTTPS
```

Its job, in one sentence: **decide which model should answer each request — cheaply and predictably — and move the conversation to a stronger model only when there is evidence it needs one.**

The three decision inputs, in increasing order of "smartness":

1. **Explicit intent** — the client (or you, via markers/headers/pins) says where it goes.
2. **Size** — the context floor: if the request won't fit the chosen model's window, bump up until it does.
3. **Difficulty** — optionally, a cheap local model judges each new user message as FAST or DEEP before routing (opt-in).

Everything else (escalation rules A–D) is *reactive*: it moves an ongoing task up the chain when failures accumulate on the current tier.

## 2. Design principles

- **Local-first, deterministic by default.** The core path uses no LLM to judge difficulty — only headers, config, size estimates and failure counters. An LLM-based complexity verdict exists but is opt-in (`routing.complexity.enabled`), advisory (any failure falls back to plain auto policy), and cached.
- **Transparent proxying.** The request body is forwarded verbatim except for the `model` field (rewritten per backend) and, when present, prompt markers (stripped). No summarizing, truncating or rewriting of your prompts.
- **Per-request vs per-task effects are explicit.** Size-based bumps and marker steers affect *one request* (or one user message in Claude Code); escalation rules move *task state*. The docs and `x_router.reason` always tell you which happened.
- **Fail soft, fail visible.** A down judging model degrades to auto policy; a down backend falls back per config or returns a clean error; every response carries an `x_router` block explaining the decision.
- **Hard cloud-cost safeguards.** The frontier (cloud) tier is gated by rate and cost budgets; when blocked you get a well-formed "escalation required" sentinel instead of a crash.
- **Model- and GPU-agnostic.** The router only talks to OpenAI-compatible HTTP endpoints. It doesn't load models, know about GPUs, or care what the model id string actually is (it can be a full GGUF path).

## 3. Architecture & file map

Single Python package (`router/`), FastAPI + httpx + Pydantic v2, one uvicorn process:

| File | Responsibility |
|------|----------------|
| `router/__main__.py` | CLI entry point (`llm-router --config ...`) |
| `router/api.py` | The FastAPI app: all endpoints, the shared `_dispatch()` pipeline, admin state (pin/bounds) |
| `router/routing.py` | `Router.resolve()` — the routing priority; prompt-marker helpers; context-floor walk |
| `router/escalation.py` | `EscalationController` — rules A–D over task/session state |
| `router/sessions.py` | In-memory `SessionStore`: sessions, tasks, TTL sweep |
| `router/context.py` | Context-size estimation (chars-per-token heuristic + completion term) |
| `router/complexity.py` | Opt-in LLM complexity classifier (judge protocol + LRU cache) |
| `router/anthropic.py` | Anthropic Messages API ↔ OpenAI chat-completions translation, both directions, incl. streaming SSE and thinking blocks |
| `router/backends/base.py` | `Backend` ABC, `BackendResult`, `BackendError`, cached health helper |
| `router/backends/openai_compatible.py` | llama-server / LM Studio / any OpenAI-compatible endpoint; model rewriting; `/props` context query |
| `router/backends/openrouter.py` | Cloud frontier via OpenRouter |
| `router/backends/mock.py` | Test backend with scripted behaviors (success/stream/tool_call/error/timeout/echo) |
| `router/config.py` | Pydantic config models + YAML loading + cross-validation |
| `router/events.py` | `POST /events` lifecycle-event handling |
| `router/metrics.py` | Minimal Prometheus-format counters/histograms |
| `router/models.py` | Request/response builders, the `x_router` meta block |

There is no database, no queue, no worker pool: one event loop, in-memory state, direct HTTP fan-out to backends. Deliberately boring.

## 4. Request lifecycle (step by step)

Both public chat endpoints — `POST /v1/chat/completions` and `POST /v1/messages` — funnel into the same pipeline (`_dispatch()` in `api.py`). The Anthropic endpoint first translates its body to OpenAI shape; everything after that is shared.

```text
client request
   │
   ▼
[0] parse JSON, lower-case headers
   │
   ▼
[1] session/task identity
    X-LLM-Session-ID  (fallback: X-Claude-Code-Session-ID) → SessionState
    X-LLM-Task-ID     (fallback: {session}-r{seq}, a NEW task per request)
   │
   ▼
[2] prompt-marker extraction
    scan the LAST user message for @@fast|@@deep|@@frontier,
    strip all occurrences in place  → marker_tier or None
   │
   ▼
[3] complexity verdict (opt-in; only if it will be used)
    gate: classifier enabled AND no pin AND no marker AND request is "plain auto"
    judge call to the configured tier, LRU-cached per (session, sha1(message))
    → complexity_tier or None   (any failure → None, never blocks)
   │
   ▼
[4] context-size estimate
    required = est_prompt_tokens(body) + min(completion_term, max_completion_reserve?)
   │
   ▼
[5] Router.resolve()  — the priority order (§6):
    pin > marker > X-LLM-Escalate > X-LLM-Route > explicit model
        > auto (complexity verdict | task.current_route) > claude-family mapping
    → RouteDecision(tier, reason, manual, escalation?)
   │
   ▼
[6] context floor  — if required > tier.max_context, walk up the chain
    until it fits (reason becomes context_overflow)
   │
   ▼
[7] apply_attempt()  — auto-routed requests count toward rule D (§10)
   │
   ▼
[8] cloud gate (frontier only)  — rate/cost budgets; blocked → "escalation required" sentinel
   │
   ▼
[9] dispatch to backend, with configured fallbacks on transport/5xx failure
    (backend_fallback is NOT an escalation and changes no state)
   │
   ▼
[10] response + x_router meta block; metrics recorded; session bookkeeping
```

Notes:

- Steps 2–3 happen **before** the size estimate, so marker stripping keeps the estimate honest.
- The complexity judge (step 3) is awaited *before* dispatching the real request — one extra round trip to the judging tier per new user message, deduplicated by cache for Claude Code tool loops.
- A non-2xx from a backend (e.g. HTTP 400) is forwarded to the client as-is; if the `backend_error` signal is enabled it also counts toward rule D.

## 5. Tiers, aliases & model rewriting

| Tier | Client alias(es) | Typical role |
|------|------------------|--------------|
| fast | `local-fast`, `fast` | Cheap local model; default start for `auto` |
| deep | `local-deep`, `deep` | Stronger local model; escalation target |
| frontier | `frontier` | Cloud (OpenRouter); overflow + cost-gated |
| auto | `auto` | "route it" — activates the whole machinery |

Backends are configured by tier key in YAML. Each backend has a `model` field holding **the exact model id that server reports** at `/v1/models` — for local llama-servers that is usually a full GGUF path like `/mnt/.../Qwen3.5-9B-....gguf`. On every outgoing request the router rewrites `body.model` to that value, so clients can keep using friendly aliases (`auto`, `local-fast`) while servers see their own names.

Tiers without a configured `max_context` (or with `null`) are treated as "unknown window — assume it fits": the context floor never bumps *away* from them on size grounds. That's how the frontier placeholder behaves before OpenRouter is wired up.

## 6. The routing decision — full priority order

`Router.resolve()` applies, highest first:

| # | Signal | Source | Reason string | `manual`? |
|---|--------|--------|---------------|-----------|
| 0 | **Authoritative pin** | `GET /route/all/<tier>` (admin) | `pinned` | yes |
| 1 | **Prompt marker** | `@@fast`/`@@deep`/`@@frontier` in last user message | `prompt_marker` | yes |
| 2a | Explicit escalation header | `X-LLM-Escalate: <tier>` (also records rule-A state) | `explicit_escalate` | yes |
| 2b | Explicit route header | `X-LLM-Route: <tier>` (this request only) | `explicit_route_header` | yes |
| 3 | Explicit model name | `model: local-fast\|local-deep\|frontier` | `explicit_model` | yes |
| 4a | **Complexity verdict** (opt-in) | classifier judged this user message | `complexity` | no |
| 4b | Auto policy | task's current route (post-escalation state) | `auto_policy` | no |
| 5 | Configured default | `routing.default` when no model given at all | `default` | no |

Then, as a post-step that can override *any* of the above: the **context floor** (§7), which bumps the tier up until the estimated size fits (`reason: context_overflow`).

Details worth knowing:

- **Pin (0)** short-circuits everything and also *skips the context floor*: the pin already runs the tier at its maximum known window, so a too-big prompt should get the backend's own precise error rather than a silent bump away from the pinned tier.
- **Marker (1)** outranks headers and model names — it is the most recent human intent — but not the pin. It changes no state: the next user message reverts to automatic routing (§9).
- **`X-LLM-Escalate`** is special among explicit signals: when rule A's signal is enabled, it also *records* an escalation in task state (so subsequent auto requests of that task stay on the stronger tier), whereas `X-LLM-Route` and markers are per-request only.
- **Step 4a vs 4b:** a complexity verdict replaces the task route for this request but keeps `manual=False`, so rule-D attempt accounting continues exactly as if it were plain auto policy.
- **Claude-family fallback (after step 4):** Claude Code's *background* tasks (the auto-mode Bash safety classifier, context collapse, small helpers) send their own model names — e.g. the classifier sends `claude-sonnet-5` regardless of what you run CC with. Unknown names would otherwise 404 and block every gated command ("classifier unavailable"). So any `claude-*` name maps by family: `haiku* → fast`, everything else `→ deep` (falling back to fast if the config has no deep tier), reason `claude_family`. Non-claude unknown names still 404.
- **`X-LLM-Route: auto`** is a subtle case: `auto` is not a tier alias, so it falls *through* to step 4 (the auto policy) rather than matching step 2b.

## 7. The context floor (size-based bumping)

Independent of how the tier was chosen, every request's **estimated context need** is checked against that tier's `max_context`; if it doesn't fit, the tier walks up the escalation chain until it does. Per-request, no state change — long conversations must use the bigger window even when the task itself is easy.

### The estimate

The router deliberately does not tokenize (no tokenizer dependency). It estimates from the serialized request body — messages **and** tool schemas, i.e. everything the backend will see:

```text
est_prompt = max(1, int(len(json.dumps(body)) / chars_per_token))     # default divisor 3.0
completion = body.max_tokens or body.max_completion_tokens or completion_reserve   # reserve default 8192
if max_completion_reserve is set:  completion = min(completion, max_completion_reserve)
required   = est_prompt + completion
```

Why each piece exists:

- **chars_per_token = 3.0** — conservative for code-heavy agent traffic; the server-side chat template adds per-message tokens the client never sends, so overestimating is the safe direction (a false bump costs one slower model; an undercount costs a failed request).
- **completion term** — llama-server will happily generate until EOS or a full context if you don't cap it; the router reserves headroom for the answer.
- **max_completion_reserve (the cap)** — a client's `max_tokens` is an *upper bound, not a requirement*. Claude Code sends its full output budget on every request; without a cap that alone pushes every turn past small tiers' windows. The cap only ever lowers the term — small explicit caps pass through unchanged.

### Worked example (real numbers from this deployment)

Fast tier ctx = 132,864 (queried at startup), deep ctx = 172,768, `max_completion_reserve = 65536`. A fresh Claude Code turn: baseline prompt ≈ 45k tokens (system + tools + MCP), CC sends `max_tokens=65536`:

```text
required ≈ 45,000 + min(65,536, 65,536) ≈ 110,536  ≤ 132,864  → stays on FAST ✓
```

Without the cap (the original bug): `45,000 + 210,000 = 255,000` — past fast *and* deep → bumped to frontier → "escalation required" on every single turn, even for "Say hi".

### The walk

```text
while tier.max_context is not None and required > tier.max_context:
    tier = next tier in escalation.chain   # respects allow_frontier
# stop at the top of the chain; if even that can't fit, go there anyway —
# the backend's own error is more precise than ours
```

The bump is reported as `x_router.reason: context_overflow` with an `escalation` block whose `attempts` field carries the estimated token requirement (not a failure count), and it counts in `router_escalations_total`.

## 8. The complexity classifier (difficulty-based routing)

The opt-in exception to "no LLM judges difficulty" (`routing.complexity.enabled: true`). A configured backend — usually the cheap one — answers exactly one word, **FAST** or **DEEP**, for each new user message; the verdict feeds step 4a of the priority order.

### Judge protocol

```text
system: "You classify coding tasks by difficulty for model routing.
         Reply with exactly one word, nothing else: FAST or DEEP."
user:   <rubric> + "\n\nTask:\n" + last_user_message[:max_input_chars]

body:   max_tokens=1024, temperature=0, no tools, stream=false
        (+ chat_template_kwargs={"enable_thinking": false} when disable_thinking)
```

- **Why 1024 tokens:** the local fast model is a *thinking* model — it emits reasoning before content. A tight budget gets fully consumed by reasoning and leaves `content` empty (→ no verdict). `disable_thinking: true` sends `enable_thinking=false`, which Qwen3 llama.cpp servers honor (skips reasoning entirely); servers that ignore unknown fields just think anyway; a strict 400 degrades safely to "no verdict".
- **Parsing:** only `choices[0].message.content` is read — never `reasoning_content`, whose prose may mention both words. The content is normalized (`[^A-Za-z]` stripped, uppercased) and must be exactly `FAST` or `DEEP`; anything else → no verdict.

### When it runs (the gate)

The judge is called only when its result will actually be used:

```text
classifier enabled  AND  no active pin  AND  no prompt marker in this message
AND  the request would reach step 4 ("plain auto": no explicit headers/model)
```

So explicit models, header-routed requests, claude-family background tasks and pinned traffic never pay for a verdict.

### Caching & failure semantics

- Verdicts are LRU-cached per `(session_id, sha1(user_message))` (`cache_size`, default 256). Claude Code re-sends the same last user message across every tool-loop request of a turn — so one new user message costs **one** judging round trip, not one per tool call.
- **Only successful verdicts are cached.** A timeout while the judging tier is busy must not pin "no verdict" for the whole tool loop; the next request retries the judge.
- Any failure (timeout via `asyncio.wait_for`, backend error, non-200, unparseable answer) → `None` → plain auto policy. The judge never blocks or fails the real request.

### Config

```yaml
routing:
  complexity:
    enabled: false          # off by default; opt in per deployment
    tier: fast              # judging backend (usually the cheap one)
    timeout_seconds: 10     # per judge call; expiry -> auto policy
    max_input_chars: 2000   # user message truncated to this before judging
    cache_size: 256         # LRU of (session, message-hash) verdicts
    disable_thinking: true  # send enable_thinking=false (Qwen3 llama.cpp)
```

## 9. Prompt markers

Append `@@fast`, `@@deep` or `@@frontier` anywhere in your message to steer it — the lightest-weight steering mechanism, no endpoint and no state.

Semantics:

- **Rightmost token wins** if several appear; case-insensitive (`@@DEEP` works); `\b` boundary so `@@fastest` doesn't match.
- **All occurrences are stripped in place** before forwarding — the model never sees them, and the context estimate reflects what is actually sent.
- Only the **last user message** is scanned; markers in earlier history (already stripped when they were sent) or in assistant/tool messages don't count.
- Priority: above headers and explicit model names, below an authoritative pin (§6). The context floor still applies afterwards — a too-big prompt on `@@fast` bumps up as usual (`reason` becomes `context_overflow`; the extracted marker is logged so this stays visible).
- **No state changes.** In Claude Code this means *per user message* in practice: every tool-loop request of a turn carries the same last user message, so one marker covers the whole agentic turn; the next human message reverts to automatic routing.

```text
you:  create a python code to find oldest files in a folder @@fast
      → this whole turn (all its tool calls) runs on fast
you:  now refactor it into a proper CLI with tests @@deep
      → this turn runs on deep; next plain message reverts to auto
```

## 10. Escalation engine — rules A–D

The `EscalationController` keeps task state and decides, with **deterministic signals only**, whether a task stays put or moves one step up the chain (`fast → deep → frontier`). It never performs inference; the model doesn't have to know it's struggling.

| Rule | Signal | Default threshold | Effect |
|------|--------|-------------------|--------|
| **A** — explicit request | `X-LLM-Escalate` header or `explicit_escalation` event via `/events` | immediate | jump to the requested tier (records state when the signal is enabled) |
| **B** — repeated test failure | `test_failure` events within one task, on the current tier | 2 (`failure_threshold`) | escalate one step up |
| **C** — repeated tool failure | `tool_failure` events, same scope | 2 | escalate one step up |
| **D** — retry limit | `max_*_attempts` auto-routed requests to the current tier **and** ≥1 recorded failure signal | fast: 2, deep: 2 | escalate one step up ("retries without success") |

Mechanics:

- Failure counters are **per task and per tier**: when a task escalates (or completes), its `failures` map is cleared — the stronger model starts fresh.
- Rule D counts only **auto-routed** attempts (`manual=False`). Explicit choices (headers, markers, explicit models) don't count — you asked for that tier, so its failures are your call. Complexity verdicts keep `manual=False`, so they count like any auto request.
- Failure signals can also come from the request path itself: backend errors/timeouts observed during dispatch (`note_failure`), gated by the `backend_error`/`timeout` signal switches (both default off — a crashed llama-server is an infrastructure problem, not "the model was too weak").
- Each escalation appends to the task's `escalation_history`, updates `session.current_route`, and increments `session.escalation_count`.

## 11. Sessions & tasks (the state model)

State is **in-memory by design** — counters and timestamps only, no conversation content; lost on router restart (acceptable for v1). Idle sessions are swept lazily once the store grows past 1024 entries (`ttl_seconds`, default 3600s).

- **Session identity:** `X-LLM-Session-ID` if present, else Claude Code's own `X-Claude-Code-Session-ID` (so CC works out of the box), else a generated `sess-…`.
- **Task identity:** `X-LLM-Task-ID` if present; otherwise each request gets its own task `{session}-r{seq}`. Consequence: **Claude Code traffic creates a fresh task per API call**, so for CC only the context floor and (opt-in) complexity verdict move tiers between turns — rule-D accumulation across turns would require clients to send stable `X-LLM-Task-ID`s (a Claude Code hook could do this; see §21).
- **`task.current_route`** is where auto policy starts for that task; it only moves via recorded escalations.

## 12. Backends: implementations, health, fallbacks

### Implementations

| Type | Used for | Notes |
|------|----------|-------|
| `openai_compatible` | llama-server, LM Studio, any OpenAI-compatible endpoint | Rewrites `model`; sends `Authorization: Bearer <key>` when configured (literal `api_key` wins over `api_key_env`); streaming proxied line-by-line; can query real context size via `GET /props` (`query_context_size`) |
| `openrouter` | Cloud frontier | API key from env by default |
| `mock` | Tests | Scripted behaviors: `success`, `stream`, `tool_call`, `error`, `timeout`, `echo` (returns the received request JSON as content — handy to assert verbatim forwarding) |

### Health & context-size query

- Each backend's health is checked every `health.check_interval_seconds` (default 10s, 2s timeout) and cached; `/health` reports overall status (`ok`/`degraded`) plus per-backend latency.
- With `query_context_size: true`, the router asks llama-server at startup for its **real** window via `GET /props` → `default_generation_settings.n_ctx` and uses that instead of the (possibly stale) configured `max_context`. A failed query keeps the manual value. This is why a config can say 132768 while the router logs "using queried context size 132864".

### Fallbacks ≠ escalations

A *backend* failure (connection refused, 5xx, timeout) goes through `routing.fallbacks` — e.g. `deep: [fast]` to degrade rather than error on a 27B crash. This is **not** an escalation: no task state changes, and the served tier shows up as `x_router.reason: backend_fallback`. Cloud fallback only happens when `cloud.enabled` is true.

## 13. The Anthropic endpoint (/v1/messages)

Claude Code speaks the Anthropic Messages API; this endpoint translates it to OpenAI chat-completions on the way in and back on the way out, so CC's *whole agent loop* runs through routing/escalation:

- **Request translation:** system prompt → first message; text blocks joined; `tool_use` → OpenAI `tool_calls`; `tool_result` → `role:"tool"` messages; images mapped to `image_url` parts; thinking/redacted-thinking blocks dropped on the way in.
- **Response translation (non-streaming):** OpenAI completion → Anthropic message with proper `content` blocks, including a `thinking` block when the backend returned `reasoning_content`.
- **Streaming:** full Anthropic SSE event protocol is synthesized from the proxied stream — `message_start`, `content_block_start/delta/stop`, `message_delta` (with real usage), `message_stop`; the router's `x_router` meta rides in a final chunk before `[DONE]`. A gotcha fixed along the way: the translated request must copy the client's `stream` field itself, or llama-server answers one non-streamed JSON body and the SSE translator sees nothing.
- **Usage accounting:** real prompt/completion token counts are requested from llama.cpp (`stream_options.include_usage`) and reported in `message_delta.usage`; `message_start` carries a chars-per-token estimate as its seed.

## 14. Cloud cost protection

The frontier tier is guarded by hard limits so an unattended agent can't burn money:

- **Rolling hourly request cap** (`cloud.max_requests_per_hour`, default 100).
- **Daily estimated-cost cap** (`cloud.max_estimated_cost_usd_per_day`) computed from response `usage` × `cloud.pricing` (per-1M-token prices; zero pricing ≈ cost protection off, which is how the simulated-frontier test configs run).
- **Automatic escalation is opt-in** (`cloud.allow_automatic_escalation`, default false): when disabled, a deep failure must *not* silently call OpenRouter — an "escalation required" response is returned instead.
- **Manual escape hatch:** explicit frontier requests (header/model/marker) are still allowed at the limit when `allow_manual_when_limited` (default true).

**The sentinel:** when blocked, the router returns HTTP 200 with a *valid* completion whose content starts with `[llm-router] escalation required: …` and whose `x_router.escalation.required` is `true` — so OpenAI-compatible clients don't crash while still being able to detect it (e.g. via a hook) and retry with `X-LLM-Escalate: frontier`.

## 15. Admin & observability endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Overall status + per-backend health/latency |
| `GET /health/backends` | Per-backend health only |
| `GET /metrics` | Prometheus text format (see below) |
| `POST /events` | Lifecycle events from Claude Code hooks: `{session_id, task_id, event, metadata}`. Known events: `task_start`, `task_complete`, `task_failure`, `tool_failure`, `test_failure`, `explicit_escalation`; unknown ones are accepted and logged at debug (a hook must never crash the agent). Response includes current route + whether it escalated |
| `GET /ctxlen` | Current context-length bounds per tier |
| `GET /ctxlen/<tier>=<n\|reset>` | Set a tier's effective `max_context` on the live router (in-memory, applies from the next request); `reset` restores the *effective startup* value — queried size wins over stale config. Returns `{tier, max_context, previous_max_context}`; unknown tier → 404, bad value → 400 |
| `GET /route` | Current routing mode (pin) + bounds |
| `GET /route/all/<tier>` | **Pin** all traffic to one tier at its max known window (context floor skipped while pinned); snapshots pre-pin bounds first |
| `GET /route/reset` | Back to defaults: no pin, startup bounds |
| `GET /route/last` | Restore the pre-pin context bounds (and clear the pin) |

**Metrics** (`router_*`, low-cardinality labels only): `requests_total`, `request_latency_seconds` (histogram), `backend_requests_total{backend}`, `backend_errors_total{backend,kind}`, `escalations_total{from,to,reason}`, `cloud_requests_total`, `cloud_blocked_total{reason}`, `tool_failures_total`, `test_failures_total`.

**Logs:** every request logs id/session/task/model/stream/headers (secrets masked); notable decisions log their own lines — `ESCALATION …`, `CONTEXT-OVERFLOW … required_tokens~N`, `PROMPT-MARKER … tier=…`, `COMPLEXITY session=… tier=…` — so `/tmp/llm-router.log` is a first-class debugging surface.

## 16. x_router reference (reasons catalog)

Every response carries:

```json
"x_router": {
  "request_id": "abc123", "session_id": "sess-…", "task_id": "…",
  "route": "deep", "backend": "local-deep", "reason": "auto_policy",
  "escalation": { "from": "fast", "to": "deep", "reason": "repeated_tool_failure", "attempts": 2 }
}
```

`reason` values you can observe:

| reason | Meaning |
|--------|---------|
| `pinned` | An authoritative `/route/all/<tier>` pin decided it (floor skipped) |
| `prompt_marker` | A `@@fast/@deep/@frontier` marker in the last user message decided it |
| `explicit_escalate` | `X-LLM-Escalate` header |
| `explicit_route_header` | `X-LLM-Route` header (this request only) |
| `explicit_model` | Client named a tier alias explicitly |
| `default` | No model given; configured default applied |
| `complexity` | The opt-in classifier's verdict decided it |
| `auto_policy` | Plain automatic routing: task's current route |
| `claude_family` | A Claude Code background task sent its own model name; mapped by family (haiku→fast, else deep) |
| `context_overflow` | Context floor bumped the tier up to fit the estimated size (`escalation.attempts` carries the token estimate) |
| `backend_fallback` | The decided backend failed and a configured fallback served instead (not an escalation) |
| `backend_unavailable` | Every candidate for the tier failed; 502 returned with this meta |

Escalation-block `reason` values: `explicit_request`, `repeated_test_failure`, `repeated_tool_failure`, `retry_limit`, `backend_error`, `timeout`. For blocked cloud calls the block is instead `{"required": true, "to": "frontier", "reason": "<why>"}`.

## 17. Configuration reference

YAML → Pydantic models with 1:1 key mapping; cross-validation checks that escalation-chain/fallback/complexity tiers reference real backends. Full example in `yaml/config.example.yaml`; the live deployment config is `config_i7.yaml`.

```yaml
server: {host, port}                 # bind address (default 127.0.0.1:8000)

defaults:
  route: auto                        # model used when a request omits "model"

backends:                            # keyed by tier name
  <tier>:
    type: openai_compatible | openrouter | mock
    name: …                          # shown in x_router.backend / health
    base_url: http://host:port/v1
    model: …                         # EXACT id the server reports at /v1/models (may be a GGUF path)
    api_key: "…"                     # literal; wins over api_key_env (llama-server --api-key setups)
    api_key_env: ENV_VAR             # read lazily at request time (rotated key needs no restart)
    timeout_seconds: 300
    extra_headers: {}
    max_context: 132768              # hard window; null = unknown, assumed to fit
    query_context_size: true         # ask llama-server /props for the real n_ctx at startup (PRD §52)
    # mock-only: behavior, response_text, error_status, delay_seconds

routing:
  default: fast                      # tier that auto starts on
  fallbacks: {deep: [fast]}          # backend-failure policy (NOT escalation); cloud target needs cloud.enabled
  context:                           # the context floor (§7)
    enabled: true
    chars_per_token: 3.0             # conservative divisor for the serialized body
    completion_reserve: 8192         # headroom when client gives no max_tokens
    max_completion_reserve: 65536    # cap on the completion term (null = trust client)
  complexity:                        # opt-in difficulty classifier (§8)
    enabled: false
    tier: fast
    timeout_seconds: 10
    max_input_chars: 2000
    cache_size: 256
    disable_thinking: true
  escalation:
    enabled: true
    max_fast_attempts: 2             # rule D per tier
    max_deep_attempts: 2
    failure_threshold: 2             # rules B/C
    allow_frontier: true             # automatic escalation may reach the cloud tier (cloud gate still applies)
    signals:                         # which deterministic signals are active
      explicit_request: true         # rule A
      repeated_tool_failure: true    # rule C
      repeated_test_failure: true    # rule B
      timeout: false                 # backend timeouts count as failures when true
      backend_error: false           # backend 5xx/4xx responses count when true
    chain: [fast, deep, frontier]

cloud:                               # cost protection for the frontier tier (§14)
  enabled: false                     # master switch (off = even manual frontier blocked)
  allow_automatic_escalation: false  # opt-in; recommended false initially
  max_requests_per_hour: 100
  max_estimated_cost_usd_per_day: 10.0
  allow_manual_when_limited: true
  pricing: {input_per_mtok: 0, output_per_mtok: 0}

logging: {level, log_requests, log_responses, log_prompts}
metrics: {enabled: true}
health: {check_interval_seconds: 10, timeout_seconds: 2}
```

## 18. Claude Code integration details

Running CC through the router (`ANTHROPIC_BASE_URL=http://127.0.0.1:8000`, `claude --model auto`) has several interactions worth understanding:

- **Session identity for free:** CC sends `X-Claude-Code-Session-ID`; the router falls back to it, so session/task state works without extra configuration.
- **Background model names:** CC's internal helpers send their own model ids (the auto-mode Bash safety classifier sends `claude-sonnet-5` no matter what). The claude-family mapping (§6) keeps them from 404-ing — otherwise *every* gated command would be blocked with "classifier unavailable".
- **Env var sizing** (`CLAUDE_CODE_MAX_CONTEXT_TOKENS` / `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, set in the launcher): CC's auto-compact threshold is `window − min(MAX_OUTPUT, 20k) − 13k`. The two constraints: threshold well above the ~45k session baseline (else compaction thrashes), and worst-case request at compact time (threshold input + MAX_OUTPUT) within the biggest local tier's ctx. Recommended for a fast(132k)/deep(172k) setup: **140000 / 65536** → threshold 107k, worst case ≈ 172.5k ≤ deep ctx; and MAX_OUTPUT matching `max_completion_reserve` keeps CC's per-request budget equal to what the router reserves when routing (full rationale in the root README).
- **Markers are per user message:** because every tool-loop request of a turn re-sends the same last user message, one `@@deep` covers the whole agentic turn and reverts at your next plain message.
- **The complexity judge is CC-friendly by design:** cached per (session, message hash), so a 20-tool-call turn costs exactly one judging round trip; the gate skips it for claude-family background requests so gated Bash commands don't pay extra latency.

## 19. Testing strategy

`pytest`, run via `.venv/bin/pytest` — the whole suite runs **without GPUs or network** because every backend in tests is a mock:

- `tests/conftest.py` builds a fully mock-backed app (`make_config(**overrides)` deep-merges into a base config with three mock backends answering distinct texts, so you can assert *which* tier served).
- Tests are mostly **HTTP-level** (TestClient → assert on response JSON + `x_router.route/reason`) — they exercise the real pipeline end to end; estimator and classifier logic additionally get direct unit tests.
- The mock backend's behaviors (`success`, `stream`, `tool_call`, `error`, `timeout`, `echo`) cover streaming, tool round-trips, failure paths and verbatim-forwarding assertions.
- New features follow the same pattern: e.g. marker extraction is unit-tested as a pure function *and* via HTTP (priority vs headers/pin); the classifier is tested against a stub backend for parsing, timeouts, errors, cache hits/eviction and no-negative-caching.

## 20. Operations notes

- **Start/stop:** `start-router.sh [start|--fg|status|-c|--chk-ready|stop]` — pidfile `/tmp/llm-router.pid`, log `/tmp/llm-router.log`; `-c` runs a readiness probe that makes real requests to each tier. The root `start-router.sh` is a symlink to the active launcher in `daemon/`.
- **Restart semantics:** all session/task state is in-memory and lost on restart; context bounds revert to config (or re-queried sizes). Pins and `/ctxlen` changes are likewise in-memory.
- **API keys:** both local llama-servers run with `--api-key`; the key lives literally in the config (`backends.<tier>.api_key`) and is sent as `Authorization: Bearer`. If a backend suddenly 401s or reports unhealthy, the key changed — update the config and restart. (llama.cpp ignores a stray Authorization header when started without `--api-key`, so mixed setups work.)
- **Model ids:** if a backend 404s on model or health fails, re-fetch `curl <host>:<port>/v1/models` and paste the exact id into the config.

## 21. Ideas & future extensions

Ranked roughly by value-to-effort for this deployment:

1. **Sticky / task-scoped markers.** Today a marker is per user message (which in CC = per turn). A `@@deep:sticky` variant, or a Claude Code hook that sends a stable `X-LLM-Task-ID`, would let one steer cover an entire multi-turn task — and would also make rule-D accumulation meaningful for CC traffic.
2. **Heuristic pre-filter around the judge.** A cheap keyword/length check (e.g. "refactor", "debug", "investigate" → skip straight to deep; trivially short + no code refs → fast) could avoid most judging round trips, with the LLM judge as tie-breaker only for ambiguous messages.
3. **Prompt-cache-aware hysteresis.** Switching tiers mid-conversation breaks the backend's prompt cache (and re-prefills the whole history). Consider: never *downgrade* within a session unless explicitly told (marker/pin), and prefer staying put when the verdict is borderline.
4. **Persistent state.** SQLite for sessions/tasks/escalation history would survive router restarts — useful since CC long-running tasks currently "forget" their escalation state on every restart.
5. **Load-aware routing.** llama-servers here run `--parallel 1`; track per-tier in-flight depth and, when deep is busy with a long generation, hold borderline tasks on fast (with a timeout) instead of queueing behind it.
6. **Structured judge output.** llama-server supports JSON response formats — ask for `{tier, confidence, category}` instead of one word; enables double-checking low-confidence verdicts and per-category stats ("debugging tasks: 80% deep").
7. **Feedback learning from outcomes.** Log which tier each task *ended* on (escalated or not) alongside the initial verdict; over weeks this reveals whether the rubric is miscalibrated for your workload, and could auto-tune it.
8. **Token-accurate estimation (optional).** The chars/3 heuristic is deliberately simple; an optional real tokenizer (or calibrating against `prompt_tokens` reported by backends) would tighten the floor at the cost of a dependency.
9. **Per-tier rate limiting / concurrency caps.** Protect a single-GPU fast model from CC's parallel subagents hammering it while deep idles.
10. **Local "cost" accounting.** Estimate wall-time (and kWh, if you track power) per tier and expose in `/metrics`, so "fast vs deep" becomes an explicit cost/latency tradeoff like the cloud budget already is for frontier.
11. **Subagent-aware routing.** Detect CC Task-tool subagents (distinct session patterns/headers); they're often simpler and parallel — a natural fast-tier candidate with its own escalation policy.
12. **Per-project routing profiles.** Different configs per repo (a math-heavy project defaulting to deep, a scripting one to fast) selected by working directory or an env var in the launcher.
13. **Multi-user auth & quotas.** Currently open on 127.0.0.1; if ever exposed beyond localhost, per-key identity + budgets before anything else.

---

*Keep this document honest: when a rule changes, update §6/§7/§8 and the reasons catalog (§16) in the same commit.*
