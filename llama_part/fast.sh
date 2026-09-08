#!/bin/sh

tpl_dir=/home/andrei/.local/tpl
llama_bin=/dd2/andrei/bench/build_cuda13/bin/llama-server

m="/mnt/models_sas_ssd/lmstudio/Models/mradermacher/Qwen3.5-9B-Claude-4.6-HighIQ-THINKING-HERETIC-UNCENSORED-GGUF/Qwen3.5-9B-Claude-4.6-HighIQ-THINKING-HERETIC-UNCENSORED.Q8_0.gguf"

ctx=132768

CUDA_VISIBLE_DEVICES=2 \
$llama_bin \
    --model $m \
    --host 0.0.0.0 \
    --port  8081 \
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
    --repeat-penalty 1.1 --chat-template-file $tpl_dir/qw35/qwen3.5_heretic_9b.jinja \
    --api-key 'sk-lm-...'
