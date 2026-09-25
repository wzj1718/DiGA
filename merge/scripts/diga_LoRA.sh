#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ROOT="${MODEL_ROOT:-models}"
BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT}/Llama-3.1-8B}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${MODEL_ROOT}/OrthoMerge_LoRA_Model}"
OUTPUT_PATH="${OUTPUT_PATH:-merged_models/Llama-3.1-8B_diga_LoRA_tensors-ta-}"
DEVICE="${DEVICE:-cuda:0}"

# Materialize LoRA experts, then apply the fixed T/R tensor merge.
"$PYTHON_BIN" ./merge/diga_lora_tensors.py \
  --base_model "${BASE_MODEL}" \
  --adapter_paths \
    "${ADAPTER_ROOT}/llama3-1_8b_finetune_socialiqa/" \
    "${ADAPTER_ROOT}/llama3-1_8b_finetune_magicoder/" \
    "${ADAPTER_ROOT}/llama3-1_8b_finetune_commonsense/" \
    "${ADAPTER_ROOT}/llama3-1_8b_finetune_numinamath/" \
    "${ADAPTER_ROOT}/llama3-1_8b_finetune_scienceqa/" \
  --expert_names socialiqa magicoder  commonsense numinamath   scienceqa \
  --output_path "${OUTPUT_PATH}" \
  --device "${DEVICE}" \
  "$@"
