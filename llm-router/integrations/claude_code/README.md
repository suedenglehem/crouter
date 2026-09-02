# Claude Code integration for llm-router

Three pieces, all optional and independent — use as much or as little as you want:

1. **Routing** — point Claude Code (or its subagents) at the router so requests hit `local-fast` by default.
2. **Behavioral policy** — an installable CLAUDE.md section that tells the agent *when* to ask for a stronger model.
3. **Hooks + helper script** — emit lifecycle events (`test_failure`, `tool_failure`, ...) to `/events` and call deeper tiers from shell/subagents.

The router itself never depends on CLAUDE.md (PRD §16): the policy is a behavioral guideline, not the authoritative escalation mechanism. The deterministic rules in the controller are.

---

## 1. Routing Claude Code at the router

Claude Code speaks the Anthropic Messages API; llm-router speaks OpenAI-compatible. Two options:

### Option A — subagent/shell routing (zero extra dependencies)

Keep your normal Claude Code setup and let it call the router for work through Bash. This is the recommended v1 mode and matches the PRD's subagent-friendly design (§28):

```bash
# routine request on the fast tier
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-fast","messages":[{"role":"user","content":"summarize this diff"}]}'

# or use the helper (see below)
./escalation.sh --route deep --prompt "explain why this test fails"
```

### Option B — full translation proxy

Run an OpenAI→Anthropic translator in front of Claude Code (e.g. `claude-code-router`, LiteLLM, or any Messages-API shim) with its upstream pointed at `http://127.0.0.1:8000/v1` and model name `auto`. Then the *entire* main-agent loop runs through the router, including streaming and tool calls (both are proxied transparently).

```bash
export ANTHROPIC_BASE_URL=http://localhost:<proxy-port>   # per your proxy's docs
```

Whichever option you pick, set a stable session/task identity so escalation state works:

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
