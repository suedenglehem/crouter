---
description: Authoritatively pin all router traffic to one tier (fast/deep/frontier), or reset/restore
argument-hint: "all <tier> | reset | last"
---

Steer the local LLM router's routing mode.

The arguments are: `$ARGUMENTS`. If that is empty, look at this prompt text for any of these exact phrases and use it as the argument instead: `all fast`, `all deep`, `all frontier`, `reset`, `last`. If you find none, treat the argument as empty.

Pick exactly ONE command from this table, based on the resolved argument, and run it with Bash:

| Argument | Command to run |
|---|---|
| `all fast` | `curl -s "http://127.0.0.1:8000/route/all/fast"` |
| `all deep` | `curl -s "http://127.0.0.1:8000/route/all/deep"` |
| `all frontier` | `curl -s "http://127.0.0.1:8000/route/all/frontier"` |
| `reset` | `curl -s "http://127.0.0.1:8000/route/reset"` |
| `last` | `curl -s "http://127.0.0.1:8000/route/last"` |
| (empty) | `curl -s "http://127.0.0.1:8000/route"` |

Meaning, for your final sentence: `all <tier>` pins ALL traffic to that tier at its maximum known context window; `reset` returns to the default configuration (no pin, startup bounds); `last` restores the context-length settings from right before the last `/route all ...`; empty shows the current state.

Report the JSON response verbatim in one line, then add ONE sentence stating the routing mode now (which tier is pinned, or that normal routing is active). Do not truncate or reformat the JSON.
