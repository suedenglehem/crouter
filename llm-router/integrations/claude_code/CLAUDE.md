## MODEL ESCALATION POLICY (llm-router)

This project routes through a local LLM router (`http://127.0.0.1:8000`) with three tiers: `local-fast` (default), `local-deep`, and `frontier` (cloud). Use the current local model for routine work — do not reach for stronger models by default, and do not repeatedly attempt speculative fixes on a failing approach.

### When to request escalation

If you encounter any of the following, stop guessing and escalate:

- repeated test failures
- repeated tool failures
- difficult architectural decisions
- substantial uncertainty about correctness
- complicated cross-component interactions
- difficult debugging (no clear causal chain)
- security-sensitive reasoning
- inability to explain *why* a proposed fix should work

### How to escalate

Prefer delegating the hard **subtask** to a stronger model rather than restarting the whole conversation at another model. From Bash:

```bash
# deep tier for a difficult subproblem (streaming, tracked)
./escalation.sh --route deep --session "$LLM_ROUTER_SESSION_ID" --task "$LLM_ROUTER_TASK_ID" \
  --prompt "<self-contained description of the subtask>"

# frontier only when deep could not resolve it
./escalation.sh --escalate frontier --session "$LLM_ROUTER_SESSION_ID" --task "$LLM_ROUTER_TASK_ID" \
  --prompt "<self-contained description of the subtask>"
```

Or, for a one-off direct call:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "X-LLM-Escalate: deep" \
  -d '{"model":"auto","messages":[{"role":"user","content":"..."}]}'
```

### When escalating, provide a short explanation of:

1. what has been attempted
2. what failed (include the actual error output)
3. what remains uncertain
4. what kind of reasoning is needed (e.g. "needs careful concurrency analysis", not "be smarter")

Make the subtask prompt **self-contained**: the stronger model does not share your working context unless you include it.

### When NOT to escalate

- a single isolated failure (fix and retry once)
- work that is clearly mechanical (renames, boilerplate, formatting)
- when you can already articulate why the fix works — do it on the current tier
