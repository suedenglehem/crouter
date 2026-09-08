# Claude Code integration for llm-router

Four pieces, all optional and independent — use as much or as little as you want:

1. **Routing** — point Claude Code (or its subagents) at the router so requests hit `local-fast` by default.
2. **Behavioral policy** — an installable CLAUDE.md section that tells the agent *when* to ask for a stronger model.
3. **Hooks + helper script** — emit lifecycle events (`test_failure`, `tool_failure`, ...) to `/events` and call deeper tiers from shell/subagents.
4. **Context-bound steering** — change each tier's context-length bound on the live router from Claude Code (slash command or hook), no restart needed.

The router itself never depends on CLAUDE.md (PRD §16): the policy is a behavioral guideline, not the authoritative escalation mechanism. The deterministic rules in the controller are.

---

## 1. Routing Claude Code at the router

Claude Code speaks the Anthropic Messages API; llm-router now serves both that and OpenAI-compatible (PRD §51). Two options:

### Option A — subagent/shell routing (zero extra dependencies)

Keep your normal Claude Code setup and let it call the router for work through Bash. This matches the PRD's subagent-friendly design (§28):

```bash
# routine request on the fast tier
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-fast","messages":[{"role":"user","content":"summarize this diff"}]}'

# or use the helper (see below)
./escalation.sh --route deep --prompt "explain why this test fails"
```

### Option B — run Claude Code through the router directly (built-in translator)

Point `ANTHROPIC_BASE_URL` at the router: its `POST /v1/messages` endpoint translates Anthropic <-> OpenAI, so the *entire* main-agent loop runs through routing/escalation — streaming, tool calls and usage included. No external proxy needed.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_API_KEY=local
claude --model auto        # or local-fast / local-deep / frontier
```

`auto` is not in Claude Code's built-in model catalog, so set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to your real window (e.g. the deep tier's) — that silences the unknown-model notice and sizes auto-compact correctly. Session identity works out of the box: the router falls back to Claude Code's own `X-Claude-Code-Session-ID` header when `X-LLM-Session-ID` is absent.

Whichever option you pick, a stable session/task identity keeps escalation state coherent:

- `X-LLM-Session-ID` — one value per Claude Code conversation (e.g. the session UUID).
- `X-LLM-Task-ID` — one value per user task/subtask; change it when starting a new unit of work so an escalated old task doesn't drag new work to a stronger model.

## 2. Behavioral policy (CLAUDE.md)

Copy [`CLAUDE.md`](CLAUDE.md) into your project's CLAUDE.md (or append the section). It instructs the agent to escalate deliberately and to explain *why* when it does — which is exactly what a deep subagent needs as context.

## 3. Hooks and helper script

### `escalation.sh`

A thin curl wrapper for calling deeper tiers from shell scripts, hooks, or Claude Code's Bash tool:

```bash
./escalation.sh --route deep   --prompt "why does this deadlock?"          # direct routing
./escalation.sh --escalate frontier --prompt "prove the invariant"         # rule A escalation
./escalation.sh --session <id> --task <id> --stream --prompt "..."         # tracked + streaming
```

It prints the assistant content (or raw JSON with `--raw`) and exits non-zero on HTTP errors. Configure via env: `LLM_ROUTER_URL` (default `http://127.0.0.1:8000`).

### Hooks (`hooks/`)

Claude Code hooks receive a JSON payload on stdin. The provided scripts forward the relevant signal to `/events`:

| Script | Hook event it fits | Emits |
|--------|--------------------|-------|
| `post_tool_use.sh` | `PostToolUse` (detects failed tool output) | `tool_failure` |
| `test_failure.sh`   | any hook that runs tests / watches test output | `test_failure` |
| `task_complete.sh`  | `Stop` (agent finished its turn) | `task_complete` |

Example wiring in your project's `.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      { "matcher": "Bash",
        "hooks": [{ "type": "command", "command": "/path/to/llm-router/integrations/claude_code/hooks/post_tool_use.sh" }] }
    ],
    "Stop": [
      { "hooks": [{ "type": "command", "command": "/path/to/llm-router/integrations/claude_code/hooks/task_complete.sh" }] }
    ]
  }
}
```

See [`hooks/settings.example.json`](hooks/settings.example.json) for a ready-to-edit file. The scripts read `session_id` from the hook payload when present and fall back to `$LLM_ROUTER_SESSION_ID`; set that env var (e.g. in your shell profile or Claude Code's `env`) so events land on the right session.

### End-to-end example (PRD §47)

```text
1.  Claude Code starts a task            -> X-LLM-Task-ID: task-123, model auto
2.  Requests go to local-fast
3.  Claude edits code; tests fail
4.  PostToolUse hook emits test_failure  -> POST /events (count=1)
5.  Claude retries; tests fail again
6.  Hook emits test_failure              -> count=2 => controller escalates fast->deep
7.  Next auto request routes to local-deep (x_router shows the escalation)
8.  Deep solves it; Stop hook emits task_complete (counters reset)
9.  No cloud request was made
```

If deep fails twice as well, the same mechanism reaches `frontier` — provided `cloud.allow_automatic_escalation: true`; otherwise you get an "escalation required" response and can force it with `X-LLM-Escalate: frontier`.

## 4. Steering context bounds at runtime (`/ctxlen`)

Each tier's `max_context` (the context floor's bound) can be changed on the live router without a restart — e.g. after restarting a llama-server with a different `--ctx-size`, or to keep a long session pinned to fast:

```bash
curl -s http://127.0.0.1:8000/ctxlen/fast=32000   # fast bound -> 32000 tokens
curl -s http://127.0.0.1:8000/ctxlen/deep=reset   # deep back to its startup value
curl -s http://127.0.0.1:8000/ctxlen              # show current + initial bounds per tier
```

Or via the helper (same `LLM_ROUTER_URL` env as `escalation.sh`):

```bash
./ctxlen.sh fast=32000 ; ./ctxlen.sh deep=reset ; ./ctxlen.sh
```

The floor reads the bound live on every request, so a change applies from the very next routed request. `reset` restores the *effective startup* value — the config value, or the queried size when `query_context_size` overrode it at boot. Changes are in-memory only: restarting the router reverts to the configuration file. A successful call returns `{"tier", "max_context", "previous_max_context"}`; unknown tier → 404, bad value (`abc`, `0`, `-5`) → 400.

### From inside Claude Code — slash command (manual steering)

Copy [`ctxlen.md`](ctxlen.md) to `~/.claude/commands/ctxlen.md` (user-level: works from any project) or `.claude/commands/ctxlen.md` (project-level). Then, mid-session:

```
/ctxlen fast 32000     # set the fast bound to 32000 tokens
/ctxlen deep reset     # restore deep's startup value
/ctxlen                # show current bounds
```

The command runs the curl against the router and reports the JSON result, so you can verify what changed. Plain English works too ("set the router's fast context bound to 32000") — Claude will run the same curl via Bash.

### Automatically at session start — hook

If every Claude Code session should begin from known-good bounds (e.g. after a backend restart), add a `SessionStart` hook to your settings — `~/.claude/settings.json` for all projects, or `.claude/settings.json` for one project:

```json
{
  "hooks": {
    "SessionStart": [
      { "hooks": [{ "type": "command",
          "command": "/path/to/llm-router/integrations/claude_code/ctxlen.sh fast=reset && /path/to/llm-router/integrations/claude_code/ctxlen.sh deep=reset || true" }] }
    ]
  }
}
```

The `|| true` keeps the hook quiet when the router is not up yet at session start. [`hooks/settings.example.json`](hooks/settings.example.json) includes this alongside the event hooks, ready to edit.
