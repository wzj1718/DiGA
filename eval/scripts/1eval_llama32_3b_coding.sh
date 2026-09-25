#!/bin/bash

set -e

CUDA_DEVICES="${1:-${CUDA_VISIBLE_DEVICES:-5}}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
export PYTHONUNBUFFERED=1

cd bigcode-evaluation-harness

models=( )

tasks=(
    humanevalplus
    mbppplus
)

seeds=(42 8686)

for model_path in "${models[@]}"; do
    model_name=$(basename "${model_path}")

    for task in "${tasks[@]}"; do
        for seed in "${seeds[@]}"; do
            output_path=results/${model_name}/${task}/seed_${seed}
            mkdir -p "${output_path}"

            echo "============================================================"
            echo "MODEL: ${model_path}"
            echo "TASK : ${task}"
            echo "SEED : ${seed}"
            echo "OUT  : ${output_path}"
            echo "CUDA : ${CUDA_VISIBLE_DEVICES}"
            echo "============================================================"

            "${ACCELERATE_BIN:-accelerate}" launch main.py \
                --model "${model_path}" \
                --max_length_generation 2048 \
                --precision bf16 \
                --tasks "${task}" \
                --seed "${seed}" \
                --temperature 0.2 \
                --n_samples 10 \
                --batch_size 10 \
                --allow_code_execution \
                --save_generations \
                --save_generations_path "${output_path}/generations.json" \
                --metric_output_path "${output_path}/code_eval.json" \
                --use_auth_token

            echo "COMPLETED: MODEL=${model_name} TASK=${task} SEED=${seed}"
        done
    done
done
