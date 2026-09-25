#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:2}"

MODEL_ROOT="${MODEL_ROOT:-models}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_ROOT}/Llama-3.2-3B}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/models/Llama-3.2-3B_diga_TR_GS_non_target_TA}"

# Both T/R use one-pass GS; non-target tensors use base + 0.2 * sum(delta).
"$PYTHON_BIN" ./merge/diga.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --expert_model_paths \
    "${MODEL_ROOT}/Llama-3.2-3B_coding" \
    "${MODEL_ROOT}/Llama-3.2-3B_instruction" \
    "${MODEL_ROOT}/Llama-3.2-3B_math" \
    "${MODEL_ROOT}/Llama-3.2-3B_multilingual" \
    "${MODEL_ROOT}/Llama-3.2-3B_safety" \
  --output_path "${OUTPUT_PATH}" \
  --device "${DEVICE}" \
  "$@"
