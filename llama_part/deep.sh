#!/bin/sh

tpl_dir=/home/andrei/.local/tpl
llama_bin=/dd2/andrei/bench/build_cuda13/bin/llama-server

m="/dd2/lmstudio/Models/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf"
d="/dd2/lmstudio/Models/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-draft-Q8_0.gguf"

ctx=256000

CUDA_VISIBLE_DEVICES=0,1 \
$llama_bin  \
    --model $m \
    --alias Qwen3.8-27B \
    --host 0.0.0.0 \
    --port  8080 \
    --split-mode tensor \
    --tensor-split 1.6,1.4 \
    --fit off \
    --n-gpu-layers 999 \
    --ctx-size $ctx \
    --batch-size 2048 \
    --ubatch-size 512 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --cache-prompt \
    --jinja \
    --parallel 1 \
    --cont-batching \
    --metrics \
    --temp 0.1 \
    --top-p 0.95 \
    --min-p 0.05 \
    --repeat-penalty 1.1 \
    --spec-type draft-mtp --spec-draft-n-max 5 --model-draft $d \
    --chat-template-file $tpl_bin/qw38_q6/qwen3.8.jinja \
    --api-key 'sk-lm-...'
