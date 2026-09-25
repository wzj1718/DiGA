#!/bin/bash

set -e

CUDA_DEVICES="${1:-${CUDA_VISIBLE_DEVICES:-1}}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=true

LMMS_EVAL_DIR="lmms-eval"
PYTHON_BIN="${PYTHON_BIN:-python}"

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_PIXELS="${MAX_PIXELS:-12845056}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_mllm}"

cd "${LMMS_EVAL_DIR}"

models=( )


tasks=(
    mmsi_bench
    embspatial
    ocrbench
    mmmu_val  
     pathvqa_test
)

for model_path in "${models[@]}"; do
    model_name="$(basename "${model_path%/}")"

    for task in "${tasks[@]}"; do
        output_path="${OUTPUT_ROOT}/${model_name}/${task}"
        mkdir -p "${output_path}"

        echo "============================================================"
        echo "MODEL: ${model_path}"
        echo "TASK : ${task}"
        echo "OUT  : ${output_path}"
        echo "CUDA : ${CUDA_VISIBLE_DEVICES}"
        echo "============================================================"

        "${PYTHON_BIN}" -m lmms_eval \
            --model qwen2_5_vl \
            --model_args "pretrained=${model_path},max_pixels=${MAX_PIXELS},interleave_visuals=False,attn_implementation=${ATTN_IMPLEMENTATION}" \
            --tasks "${task}" \
            --batch_size "${BATCH_SIZE}" \
            --output_path "${output_path}" \
            --log_samples
    done
done
