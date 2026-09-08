# crouter — local LLM routing & escalation for Claude Code

Operator guide for running the router on this machine. Full specification: [`prd/Local LLM Routing & Escalation System.md`](prd/Local%20LLM%20Routing%20%26%20Escalation%20System.md). Service code and its own docs: [`llm-router/README.md`](llm-router/README.md).

```text
                 Claude Code (cl2) / any OpenAI client
                          │  :8000
                          ▼
              ┌───────────────────────┐
              │      llm-router       │   routing · escalation · sessions
              │  OpenAI + Anthropic   │   context floor · cloud gate
              └──────┬───────────┬────┘
                     ▼           ▼
                fast :8081    deep i7:8080 (or local :8080)
                Qwen3.5-9B    Qwen3.8-27B          [frontier = OpenRouter, off]
```

## Layout

| Path | What |
|---|---|
| `llm-router/` | the service (FastAPI), tests, integration scripts |
| `prd/` | implementation specification (§1–§51) |
| `start-router.sh`, `start-router_i7.sh` | launch with **`config_i7.yaml`** (deep on LAN host `i7`) — this is the active setup |
| `start-router_local.sh` | launch with **`config.yaml`** (fully local: deep at 127.0.0.1:8080) |

## Prerequisites

Both llama-servers must already be running (the router does not manage model lifecycle):

- **fast**: `:8081` — Qwen3.5-9B-Claude-HighIQ, started by `/dd2/llama-server/qw9claude.sh`
- **deep**: `i7:8080` (192.168.0.33) in the i7 config — Qwen3.8-27B-Uncensored; or local `:8080` via `/dd2/llama-server/qw_uncensored_mtp_q8_claude.sh`

## Running

```bash
./start-router.sh            # background, uses config_i7.yaml (deep on i7)
./start-router_local.sh      # fully local variant (config.yaml)
./start-router.sh status     # pid + health summary
./start-router.sh stop
```

- PID file: `/tmp/llm-router.pid` · log: `/tmp/llm-router.log` (one structured line per request: `request=… session=… task=… route=… backend=… latency=… status=…`)
- Raw start if needed: `cd llm-router && .venv/bin/llm-router --config config_i7.yaml`

Check it's up:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool   # all backends healthy?
curl -s http://127.0.0.1:8000/v1/models                        # auto, local-fast, local-deep, frontier
```

## Configuration (the parts that matter)

`config_i7.yaml` / `config.yaml`:

- **`backends.<tier>.model`** must be the *exact* model id the server reports at `/v1/models` — for llama-server that's the full GGUF file path. If a backend 404s on model, re-fetch and paste it.
- **`backends.<tier>.max_context`** feeds the context floor (below). Keep it in sync with each server's `--ctx-size` — or set **`query_context_size: true`** on that backend and let the router fetch the real value at startup from llama-server (`GET /props` → effective `n_ctx`, PRD §52); a failed query falls back to the manual number. (On this box the queried values are 132864 for fast and 172800 for deep — both differ slightly from what's written in the config files.)
- **`backends.<tier>.api_key`** — both llama-servers now run with `--api-key`, so the key is set literally here for `fast` and `deep` (repo is private, that's fine). The router sends it as `Authorization: Bearer <key>` on chat completions, health checks, *and* the `/props` context-size query. Prefer keeping secrets out of YAML? Use **`api_key_env: VAR_NAME`** instead — a literal `api_key` takes precedence over the env var when both are set.
- **`routing.context.chars_per_token`** (default 3) — how conservatively prompt size is estimated from serialized body chars. Lower = more aggressive bumping to deeper tiers.
- **`cloud.enabled: false`** right now — requests that overflow deep get a clean "escalation required" response instead of hitting OpenRouter.

## Running Claude Code through the router (cl2)

`/usr/local/bin/cl2` points `ANTHROPIC_BASE_URL=http://127.0.0.1:8000` and runs `claude --model auto`. The router's native Anthropic endpoint (`POST /v1/messages`, PRD §51) translates both directions — streaming SSE, tool calls, thinking blocks, real usage included. Session identity falls back to Claude Code's own `X-Claude-Code-Session-ID` header, so escalation tracking works with no extra config.

### Context settings (measured on this hardware, 2026-09-08)

How routing interacts with your env vars: Claude Code sizes each request's `max_tokens` to fill `CLAUDE_CODE_MAX_CONTEXT_TOKENS`. The router's context floor then estimates the request as **prompt chars / 3 + max_tokens** and bumps it up the tier chain until a backend's `max_context` fits. So the window you configure *is* roughly the routing threshold:

| cl2 settings | estimated need | where requests land |
|---|---|---|
| `MAX_CONTEXT=172000`, `MAX_OUTPUT=172000` | ~170k | **always deep** (over fast's 132768) |
| `MAX_CONTEXT=100000`, `MAX_OUTPUT=100000` | ~135–143k | **always deep** — over by a hair, grows with the conversation |
| `MAX_CONTEXT=70000`, `MAX_OUTPUT=80000` | ~66–78k + 67k budget | fast ✓ routing-wise, but see below |
| **`MAX_CONTEXT=70000`, `MAX_OUTPUT=32000`** ← recommended | ~66–78k | **fast**, with margin; long conversations still bump to deep automatically |

Two findings that shaped the recommendation:

1. **Keep `CLAUDE_CODE_MAX_OUTPUT_TOKENS` ≤ 32000.** The fast model is a *thinking* model (`reasoning_content`). With a ~67k generation budget it over-deliberates — observed asking for clarification instead of acting on trivial tool tasks. At a 32k cap it just acts, and 32k per response is far more than an agent turn needs.
2. **Don't set the window to match deep's context.** That defeats tiering: every request estimates above fast's limit and "routine" work silently runs on the 27B. Size the window for *fast* (≤ ~100k, recommended 70k); genuinely big conversations get bumped to deep by the floor anyway (`x_router.reason: context_overflow`, logged as `CONTEXT-OVERFLOW`).

Sanity check after changing settings — run a tool round-trip and confirm routing in the log:

```bash
cl2 -p "Use the Bash tool to run exactly this command: echo hi. Then reply with only its output." --allowedTools Bash
grep "route=" /tmp/llm-router.log | tail    # expect route=fast backend=local-fast for routine work
```

## Steering context bounds at runtime (`/ctxlen`)

Each tier's `max_context` bound (the context floor) can be changed on the live router without a restart — e.g. after restarting a llama-server with a different `--ctx-size`, or to pin a long session onto fast:

```bash
curl -s http://127.0.0.1:8000/ctxlen/fast=32000   # fast bound -> 32000 tokens
curl -s http://127.0.0.1:8000/ctxlen/deep=reset   # deep back to its startup value
curl -s http://127.0.0.1:8000/ctxlen              # show current + initial bounds per tier
```

The floor reads the bound live on every request, so a change applies from the very next routed request (bumps show up as `x_router.reason: context_overflow`). `reset` restores the *effective startup* value — the queried size when `query_context_size` overrode it at boot, otherwise the config number. Changes are in-memory only: restarting the router reverts to the YAML. A successful call returns `{"tier", "max_context", "previous_max_context"}`; unknown tier → 404, bad value (`abc`, `0`, `-5`) → 400.

### From Claude Code (cl2)

- **Slash command** — `/ctxlen <tier> <tokens|reset>` is installed at `~/.claude/commands/ctxlen.md` (source: [`llm-router/integrations/claude_code/ctxlen.md`](llm-router/integrations/claude_code/ctxlen.md)). Mid-session:
  ```
  /ctxlen fast 32000     # set the fast bound to 32000 tokens
  /ctxlen deep reset     # restore deep's startup value
  /ctxlen                # show current bounds
  ```
  It runs the curl and reports the JSON result so you can verify what changed. Plain English works too ("set the router's fast context bound to 32000").
- **Shell helper** — `llm-router/integrations/claude_code/ctxlen.sh [tier=value]` (honors `LLM_ROUTER_URL`, default `http://127.0.0.1:8000`).
- **Optional: auto-reset at session start** — a `SessionStart` hook in `~/.claude/settings.json` that runs the helper with `fast=reset && deep=reset || true`, so every cl2 session begins from known-good bounds even after backend restarts. Ready-to-edit example: [`llm-router/integrations/claude_code/hooks/settings.example.json`](llm-router/integrations/claude_code/hooks/settings.example.json).

## API quick reference (port 8000)

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible (streaming + non-streaming) |
| `POST /v1/messages` | Anthropic Messages API — Claude Code direct (PRD §51) |
| `GET  /v1/models` | logical aliases: `auto`, `local-fast`, `local-deep`, `frontier` |
| `GET  /health`, `/metrics` | status, Prometheus metrics |
| `POST /events` | lifecycle events from Claude Code hooks (`test_failure`, …) |
| `GET  /ctxlen`, `/ctxlen/<tier>=<n\|reset>` | show/set/reset per-tier context bounds at runtime (admin) — see above |

Every response carries an `x_router` block (route, backend, reason, escalation info). Routing priority: escalation header → route header → explicit model → session/task state → auto policy. Escalation is deterministic (repeated test/tool failures, retry limits — PRD §19–§20); backend *crashes* are fallbacks, not escalations.

## Tests

```bash
cd llm-router && .venv/bin/pytest    # 145 tests, all mock backends — no GPU needed
```
