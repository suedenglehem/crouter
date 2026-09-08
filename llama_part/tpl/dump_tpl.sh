#!/bin/sh

./gguf-py/gguf/scripts/gguf_dump.py \
    --no-tensors \
    --json \
    "$1" \
  | jq -r '.metadata["tokenizer.chat_template"].value' \
  > qwen3.5.jinja
