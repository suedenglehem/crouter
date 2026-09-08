---
description: Set or reset llm-router context-length bounds (fast/deep)
argument-hint: "<tier> <tokens|reset>"
---

Steer the local LLM router's context-length bound. The arguments given are: `$ARGUMENTS` — format `<tier> <value>` where tier is `fast` or `deep` and value is a token count (e.g. 32000) or `reset`. If no arguments were given, show the current bounds instead of changing anything.

Run exactly one Bash command — substituting the given values:

- with arguments: `curl -s "http://127.0.0.1:8000/ctxlen/<tier>=<value>"`
- without arguments: `curl -s "http://127.0.0.1:8000/ctxlen"`

Then report the JSON response verbatim in one line, and add a single sentence stating what the bound is now (and what it was before, if you changed it). Do not truncate or reformat the JSON.
