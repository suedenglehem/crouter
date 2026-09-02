# Starting the local backends (llama-server)

The router assumes both llama-servers are **always running** and stay hot — no
model loading/unloading in v1 (PRD 43, 48). Example startup commands for the
target hardware:

```text
GPU0: RTX 3090 24 GB ─┐
                      ├─ NODE pair -> deep model (~30-40B) on :8002
GPU1: RTX 3090 24 GB ─┘
GPU2: RTX 3080 Ti 12 GB -> fast model (12-16B) on :8001
```

## Fast tier — 12–16B on GPU2 (:8001)

```bash
CUDA_VISIBLE_DEVICES=2 llama-server \
  -m models/Qwen2.5-14B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8001 \
  --n-gpu-layers 99 \
  --ctx-size 16384 \
  --jinja                      # chat template handling for tool calls
```

## Deep tier — ~30–40B on GPU0+GPU1 (:8002)

The two 3090s are the preferred pair (same NUMA node). Split tensors across
both GPUs; `--tensor-split` weights follow relative VRAM (14:10 ≈ 24GB:24GB is
fine to leave at default, shown here for clarity):

```bash
CUDA_VISIBLE_DEVICES=0,1 llama-server \
  -m models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8002 \
  --n-gpu-layers 99 \
  --tensor-split 1,1 \
  --ctx-size 16384 \
  --jinja
```

Notes:

- A Q4_K_M 32B model is ~19 GB of weights — it fits across two 24 GB GPUs with
  room for KV cache. If you go larger (e.g. 70B Q4), expect to shrink ctx or
  quantize harder; the router does not care either way.
- `--jinja` makes llama-server apply the model's chat template, which matters
  for tool-call fidelity when Claude Code sends `tools`.
- Verify each server: `curl -s http://127.0.0.1:8001/v1/models` and
  `curl -s http://127.0.0.1:8002/v1/models`, then check the router's view with
  `curl -s http://127.0.0.1:8000/health`.

## Frontier tier (OpenRouter)

No local process — configured in YAML (`type: openrouter`) with the model name
and `OPENROUTER_API_KEY` from the environment.

## Keeping them running

For v1, any supervisor works (systemd units per server, tmux, screen). The
router's health endpoint will show a down backend as `"healthy": false`; see
the fallback policy in `config.yaml` (`routing.fallbacks`) for what happens to
requests while it is down. Automatic llama-server restarts are explicitly out
of scope (PRD 41) — that belongs to the future GPU lifecycle manager (PRD 43).
