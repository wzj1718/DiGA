#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ROOT="${MODEL_ROOT:-models}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/models/MLLM-821/ties/Qwen2.5-VL_diga_MLLM_tensors-ta-1GS-perm00-sense-ocr-huatuo}"
DEVICE="${DEVICE:-cuda:0}"


"${PYTHON_BIN}" merge/MLLM/diga_tensors_qwen25vl-1GS.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --expert_model_paths \
    "${MODEL_ROOT}/SenseNova-SI-1.1-Qwen2.5-VL-7B" \
    "${MODEL_ROOT}/olmOCR-2-7B-1025" \
    "${MODEL_ROOT}/HuatuoGPT-Vision-7B-Qwen2.5VL" \
  --output_path "${OUTPUT_PATH}" \
  --device "${DEVICE}" \
  "$@"
