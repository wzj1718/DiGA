#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ROOT="${MODEL_ROOT:-models}"
BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT}/Llama-3.1-8B}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${MODEL_ROOT}/Llama-3.1-8B_OFT_adapters}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_DIR/merged_models/diga_OFT-}"

echo "============================================================"
echo "Base Model:    $BASE_MODEL"
echo "Adapter Root:  $ADAPTER_ROOT"
echo "Output Path:   $OUTPUT_PATH"
echo "CUDA Devices:  $CUDA_VISIBLE_DEVICES"
echo "============================================================"

"$PYTHON_BIN" ./merge/diga_OFT.py \
    --language_model_name "$BASE_MODEL" \
    --adapter_paths \
        "$ADAPTER_ROOT/llama3-1_8b_finetune_socialiqa/" \
        "$ADAPTER_ROOT/llama3-1_8b_finetune_magicoder/" \
        "$ADAPTER_ROOT/llama3-1_8b_finetune_commonsense/" \
        "$ADAPTER_ROOT/llama3-1_8b_finetune_numinamath/" \
        "$ADAPTER_ROOT/llama3-1_8b_finetune_scienceqa/" \
    --output_path "$OUTPUT_PATH" \
    --device cuda:0
