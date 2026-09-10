# crouter — local LLM routing & escalation for Claude Code
I have a threadripper box (tr4) with 2 rtx 3090 and one 3080ti (couldn't get a third 3090 yet), qwen3.8 27b is just fine for vibe coding and i could run it at full context on 2 3090. 3080ti was sitting kinda idle. found that qwen 3.5 9b was fine for simple tasks and 12gb of 3080ti are perfect for it. so, the idea was to max out machine, running 2 models and somehow switching b/w them on the fly in claude (or cline etc) for different kinds of tasks, 9b for simple but fast coding, 27b for more complex stuff and some frontier for above. I have a second box i(i7) with 4080s and 4060ti, so, it was easy to debug it all.

Service code and its own docs: [`llm-router/README.md`](llm-router/README.md). Deep dive on how the router actually works — every component, decision rule and config key, plus future-extension ideas: [`doc/README.md`](doc/README.md).

```text
                 Claude Code (/usr/local/bin/cl) / any OpenAI client
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
| `start-router.sh` | symlink → `daemon/start-router_i7.sh` — the active setup |
| `llms-ready.py` | readiness probe — sends a real chat request to each backend (`start-router.sh -c`) |
| `daemon/` | launcher shells, one per configuration: `start-router_i7.sh` (**config_i7.yaml**, deep on LAN host `i7`), `start-router_local.sh` (**config.yaml**, fully local), `start-router_df.sh` (**yaml/config_df.yaml**, deep + frontier only) |
| `llm-router/yaml/` | extra router configs: the frontier-simulation **config_df.yaml** and config.example.yaml (the active i7/local configs stay in `llm-router/`) |
| `claude_part/`, `llama_part/` | tracked copies of the Claude Code launcher (`cl`) and example llama-server start scripts (with `--api-key`; adjust `CUDA_VISIBLE_DEVICES` per card) |

## Prerequisites

Both llama-servers must already be running (the router does not manage model lifecycle):

- **fast**: `:8081` — Qwen3.5-9B-Claude-HighIQ, started by `./llama_part/qw9claude.sh`
- **deep**: `i7:8080` (192.168.0.33) in the i7 config — Qwen3.8-27B-Uncensored; or local `:8080` via `./llama_part/qw_uncensored_mtp_q8_claude.sh`

## Running

All launchers live in `daemon/` (the root `start-router.sh` is a symlink to the i7 one). They share port 8000, `/tmp/llm-router.pid` and `/tmp/llm-router.log`, so only one router runs at a time — any of them can `stop` the running instance.

```bash
./start-router.sh                # background, config_i7.yaml (deep on i7) — active setup
./daemon/start-router_local.sh   # fully local variant (config.yaml)
./daemon/start-router_df.sh      # deep + frontier only  (yaml/config_df.yaml, chain [deep -> frontier])
./start-router.sh status         # pid + health summary
./start-router.sh -c             # readiness check: do fast/deep/frontier actually answer?
./start-router.sh stop
```

**Frontier simulation (df config).** `frontier` is not real OpenRouter here — it points at the llama.cpp server on `i7:8080`, so overflow to "cloud" is fully local. The df config sets `cloud.enabled: true` + `allow_automatic_escalation: true` with zero pricing, i.e. cost protection is effectively off (only the 100 req/h rate limit stays active). Validated 2026-09-08: a request over deep's bound escalated `deep -> frontier` (`x_router.reason: context_overflow`) and was served by the i7 model; explicit `model: frontier` calls work too. Note df needs `routing.default: deep` — otherwise auto traffic starts on the unused fast tier and can never leave it (fast is not in its chain).

- PID file: `/tmp/llm-router.pid` · log: `/tmp/llm-router.log` (one structured line per request: `request=… session=… task=… route=… backend=… latency=… status=…`)
- Raw start if needed: `cd llm-router && .venv/bin/llm-router --config config_i7.yaml`

Check it's up:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool   # all backends healthy?
curl -s http://127.0.0.1:8000/v1/models                        # auto, local-fast, local-deep, frontier
```

### Readiness check (`-c` / `--chk-ready`)

`status` only proves the router is listening; `-c` goes one step further and sends a real minimal `/chat/completions` request straight to each backend's `base_url`, bypassing the router — so it works even when llm-router itself is down. It runs [`llms-ready.py`](llms-ready.py) with whichever config the active launcher uses:

```bash
./start-router.sh -c                                  # check the active setup
./llms-ready.py --config llm-router/config_i7.yaml    # check a specific setup directly
./llms-ready.py --json                                # machine-readable output
```

fast and deep are always probed; frontier is probed only when `cloud.enabled: true` in that config (otherwise reported as SKIP). Both local models are thinking models, so the probe counts *any* generated token (`content` or `reasoning_content`) as a live answer. Exit codes: 0 = all enabled backends answered, 1 = at least one failed, 2 = bad config. A healthy run:

```text
llms-ready — config ./llm-router/config.yaml
  fast     OK   http://127.0.0.1:8081/v1  257 ms  thinking only
  deep     OK   http://127.0.0.1:8080/v1  3002 ms  thinking only
  frontier SKIP   cloud disabled (cloud.enabled=false)

READY — all enabled backends respond (fast, deep)
```

## Configuration (the parts that matter)

`config_i7.yaml` / `config.yaml`:

- **`backends.<tier>.model`** must be the *exact* model id the server reports at `/v1/models` — for llama-server that's the full GGUF file path. If a backend 404s on model, re-fetch and paste it.
- **`backends.<tier>.max_context`** feeds the context floor (below). Keep it in sync with each server's `--ctx-size` — or set **`query_context_size: true`** on that backend and let the router fetch the real value at startup from llama-server (`GET /props` → effective `n_ctx`, PRD §52); a failed query falls back to the manual number. (On this box the queried values are 132864 for fast and 172800 for deep — both differ slightly from what's written in the config files.)
- **`backends.<tier>.api_key`** — both llama-servers now run with `--api-key`, so the key is set literally here for `fast` and `deep` (repo is private, that's fine). The router sends it as `Authorization: Bearer <key>` on chat completions, health checks, *and* the `/props` context-size query. Prefer keeping secrets out of YAML? Use **`api_key_env: VAR_NAME`** instead — a literal `api_key` takes precedence over the env var when both are set.
- **`routing.context.chars_per_token`** (default 3) — how conservatively prompt size is estimated from serialized body chars. Lower = more aggressive bumping to deeper tiers.
- **`cloud.enabled: false`** right now — requests that overflow deep get a clean "escalation required" response instead of hitting OpenRouter.

## Integrating with Claude Code (install)

The router speaks the native Anthropic Messages API (`POST /v1/messages`), so no proxy or plugin is needed — Claude Code runs *through* it by pointing `ANTHROPIC_BASE_URL` at it. Four steps; only step 1 is required, the rest add steering and escalation signals:

### 1. Launcher script (required)

Install `/usr/local/bin/cl` :

```sh
#!/bin/sh
# Refuse to start if the router isn't up yet
nl=`ps aux | grep llm-router | wc -l`
[ $nl -eq 0 ] && echo "start llm-router first (./start-router.sh)" && exit 1

export ANTHROPIC_BASE_URL=http://127.0.0.1:8000   # the router, not api.anthropic.com
export ANTHROPIC_API_KEY=local                     # any non-empty value; the router doesn't check it

# Window sizing — full reasoning in claude_part/cl (the tracked launcher):
# CC 2.1.x auto-compact threshold = window - min(MAX_OUTPUT, 20k) - 13k.
#   * 140000 -> threshold 107k, well above this install's ~45k session
#     baseline (system + tools + MCP), so CC doesn't compact every turn.
#   * worst case at compact time = 107k input + 65536 output ~= 172.5k,
#     which still fits deep's ctx (~172.8k) — no spurious "escalation required".
#   * any MAX_OUTPUT >= 20k gives the SAME threshold (formula clamps at 20k),
#     so 65536 is chosen to match the router's max_completion_reserve cap:
#     CC fills max_tokens up to it on every request, and with both equal a
#     fresh turn estimates ~45k + 65k ~= 110k < fast ctx -> stays on fast;
#     only genuinely large conversations bump to deep (context_overflow).
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=140000
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=65536

exec $HOME/.local/bin/claude --model auto "$@"
```

`chmod +x /usr/local/bin/cl`, then just run `cl`. What each part does:

- **`ANTHROPIC_BASE_URL`** — the whole agent loop (streaming SSE, tool calls, thinking blocks, usage) goes to the router's `/v1/messages`, which translates Anthropic ↔ OpenAI in both directions. Session identity works out of the box: the router falls back to Claude Code's own `X-Claude-Code-Session-ID` header, so escalation tracking needs no extra config.
- **`--model auto`** — not in Claude Code's built-in model catalog; that's fine, but you must set **`CLAUDE_CODE_MAX_CONTEXT_TOKENS`** to a real window size: it silences the unknown-model notice, sizes auto-compact correctly, and (because Claude Code fills `max_tokens` up to it) it *is* roughly the routing threshold — see the table in the next section.
- **`CLAUDE_CODE_MAX_OUTPUT_TOKENS=65536`** — matches the router's `max_completion_reserve` cap so CC's per-request budget equals what the router reserves when deciding tiers (see "Context settings" below). Drop to ~32768 if the fast thinking model over-deliberates with a large generation budget.

### 2. Steering slash commands (recommended)

```bash
cp ./llm-router/integrations/claude_code/ctxlen.md ~/.claude/commands/
cp ./llm-router/integrations/claude_code/route.md  ~/.claude/commands/
```

User-level (`~/.claude/commands/`) works from any project; use a project's `.claude/commands/` instead for per-project installs. Then, mid-session: `/ctxlen fast 32000`, `/route all deep`, `/route reset`, … (see the sections below for what each does).

### 3. Escalation event hooks (optional)

The router's deterministic escalation (repeated test/tool failures → deeper tier) is fed by lifecycle events from Claude Code hooks. Copy the ready-to-edit wiring into your project's `.claude/settings.json` (or `~/.claude/settings.json`):

```bash
# start from this file — it contains PostToolUse/Stop event hooks + a SessionStart
# ctxlen auto-reset, with paths to fix up:
less ./llm-router/integrations/claude_code/hooks/settings.example.json
```

The hook scripts live in `llm-router/integrations/claude_code/hooks/` and forward signals (`test_failure`, `tool_failure`, `task_complete`) to the router's `/events`. Set `LLM_ROUTER_SESSION_ID` (in your shell profile or Claude Code's `env`) so events land on the right session.

### 4. Behavioral policy (optional)

Append [`llm-router/integrations/claude_code/CLAUDE.md`](llm-router/integrations/claude_code/CLAUDE.md) to your project's CLAUDE.md — it tells the agent *when* to escalate deliberately and how to delegate a hard subtask to a deeper tier via `escalation.sh`. The router never depends on it; the deterministic rules are authoritative, this just makes the agent a better citizen.

### Verify the integration

```bash
cl -p "Use the Bash tool to run exactly this command: echo hi. Then reply with only its output." --allowedTools Bash
grep "route=" /tmp/llm-router.log | tail    # expect route=fast backend=local-fast for routine work
```

## Running Claude Code through the router (cl)

`/usr/local/bin/cl` points `ANTHROPIC_BASE_URL=http://127.0.0.1:8000` and runs `claude --model auto`. The router's native Anthropic endpoint (`POST /v1/messages`, PRD §51) translates both directions — streaming SSE, tool calls, thinking blocks, real usage included. Session identity falls back to Claude Code's own `X-Claude-Code-Session-ID` header, so escalation tracking works with no extra config.

### Context settings (measured on this hardware, updated 2026-09-10)

How routing interacts with your env vars: Claude Code fills each request's `max_tokens` up to `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, and the router's context floor estimates the request as **prompt chars / 3 + min(max_tokens, max_completion_reserve)** — config_i7.yaml caps that term at 65536 — then bumps it up the tier chain until a backend's `max_context` fits. CC's auto-compact threshold is `window − min(MAX_OUTPUT, 20k) − 13k`, so two constraints pin the values:

| cl settings | compact threshold | worst case at compact time | where fresh turns land |
|---|---|---|---|
| `MAX_CONTEXT=70000` (old default) | ~37k | — | **below** this install's ~45k session baseline → CC compacts every turn and thrashes |
| **`MAX_CONTEXT=140000`, `MAX_OUTPUT=65536`** ← recommended | 107k | ≈ 172.5k ≤ deep ctx (~172.8k) | ~110k estimated → **fast**; long conversations bump to deep automatically (`context_overflow`) |
| `MAX_CONTEXT=172000`, `MAX_OUTPUT=65536` | ~139k | ≈ 204k > deep ctx | risks "escalation required" at compact time instead of a clean compact |

Two findings that shaped the recommendation:

1. **`MAX_OUTPUT` doesn't change compact timing once it's ≥ 20k** (the threshold formula clamps there) — so pick it for routing consistency, not compaction: matching the router's `max_completion_reserve` cap means CC's per-request budget equals what the router reserves when deciding tiers. A fresh turn then estimates ~45k + 65k ≈ 110k < fast ctx and stays on the cheap model; only genuinely large conversations bump to deep.
2. **Don't set `MAX_CONTEXT` to match deep's context.** That defeats tiering: every request estimates above fast's limit and "routine" work silently runs on the 27B. Size it so fresh turns stay under fast's window (140k does, via the cap); genuinely big conversations get bumped by the floor anyway (`x_router.reason: context_overflow`, logged as `CONTEXT-OVERFLOW`).

If the fast thinking model over-deliberates with a large generation budget, drop `MAX_OUTPUT` to ~32768 — threshold unchanged (still clamped at 20k), worst case at compact drops to ≈ 140k.

Sanity check after changing settings — run a tool round-trip and confirm routing in the log:

```bash
cl -p "Use the Bash tool to run exactly this command: echo hi. Then reply with only its output." --allowedTools Bash
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

### From Claude Code (cl)

- **Slash command** — `/ctxlen <tier> <tokens|reset>` is installed at `~/.claude/commands/ctxlen.md` (source: [`llm-router/integrations/claude_code/ctxlen.md`](llm-router/integrations/claude_code/ctxlen.md)). Mid-session:
  ```
  /ctxlen fast 32000     # set the fast bound to 32000 tokens
  /ctxlen deep reset     # restore deep's startup value
  /ctxlen                # show current bounds
  ```
  It runs the curl and reports the JSON result so you can verify what changed. Plain English works too ("set the router's fast context bound to 32000").
- **Shell helper** — `llm-router/integrations/claude_code/ctxlen.sh [tier=value]` (honors `LLM_ROUTER_URL`, default `http://127.0.0.1:8000`).
- **Optional: auto-reset at session start** — a `SessionStart` hook in `~/.claude/settings.json` that runs the helper with `fast=reset && deep=reset || true`, so every cl session begins from known-good bounds even after backend restarts. Ready-to-edit example: [`llm-router/integrations/claude_code/hooks/settings.example.json`](llm-router/integrations/claude_code/hooks/settings.example.json).

## Pinning all traffic to one tier (`/route`)

An authoritative switch that overrides *everything* — model, headers, escalation policy and the context floor: while pinned, every request goes to one tier at its maximum known window (configured `max_context`, or the size queried from `/props` at startup). Requests show up as `x_router.reason: pinned`. A prompt bigger than even that window gets the backend's own precise error instead of a silent bump away from the pinned tier.

```bash
curl -s http://127.0.0.1:8000/route/all/fast       # ALL traffic -> fast, at its max window
curl -s http://127.0.0.1:8000/route/all/deep       # same for deep (also: all/frontier)
curl -s http://127.0.0.1:8000/route/reset          # back to defaults: no pin, startup bounds
curl -s http://127.0.0.1:8000/route/last           # restore the context-length settings from right before /route/all/...
curl -s http://127.0.0.1:8000/route                # show current pin + bounds
```

Semantics: `all <tier>` snapshots the current context bounds first, then pins and raises the pinned tier's bound to its startup maximum; re-pinning to another tier keeps the original snapshot. `reset` is a full return to the default configuration (no pin, all bounds at startup values). `last` unpin *and* restores exactly the bounds that were in effect right before the first `/route/all/...`. In-memory only — restarting the router clears the pin and reverts to the YAML.

### From Claude Code (cl)

Slash command `/route`, installed at `~/.claude/commands/route.md` (source: [`llm-router/integrations/claude_code/route.md`](llm-router/integrations/claude_code/route.md)):

```
/route all fast       # pin everything to the 9B, full window
/route all deep       # pin everything to the 27B on i7
/route reset          # back to normal routing + startup bounds
/route last           # undo the last /route all ... (bounds included)
/route                # show current mode
```

## API quick reference (port 8000)

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible (streaming + non-streaming) |
| `POST /v1/messages` | Anthropic Messages API — Claude Code direct (PRD §51) |
| `GET  /v1/models` | logical aliases: `auto`, `local-fast`, `local-deep`, `frontier` |
| `GET  /health`, `/metrics` | status, Prometheus metrics |
| `POST /events` | lifecycle events from Claude Code hooks (`test_failure`, …) |
| `GET  /ctxlen`, `/ctxlen/<tier>=<n\|reset>` | show/set/reset per-tier context bounds at runtime (admin) — see above |
| `GET  /route`, `/route/all/<tier>`, `/route/reset`, `/route/last` | pin all traffic to one tier / restore routing mode (admin) — see above |

Every response carries an `x_router` block (route, backend, reason, escalation info). Routing priority: escalation header → route header → explicit model → session/task state → auto policy. Escalation is deterministic (repeated test/tool failures, retry limits — PRD §19–§20); backend *crashes* are fallbacks, not escalations.

## Tests

```bash
cd llm-router && .venv/bin/pytest    # 162 tests, all mock backends — no GPU needed
```
