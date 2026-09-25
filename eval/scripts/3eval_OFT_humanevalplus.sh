#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=4
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR="${TMPDIR:-/tmp}"
export TMP=$TMPDIR
export TEMP=$TMPDIR
mkdir -p "$TMPDIR"

cd bigcode-evaluation-harness

models=( )

tasks=(
    humanevalplus
)

seeds=(42 8686)

for model_path in "${models[@]}"; do
    model_name=$(basename "${model_path}")

    for task in "${tasks[@]}"; do
        for seed in "${seeds[@]}"; do
            output_dir=results/${model_name}--/${task}/seed_${seed}
            mkdir -p "${output_dir}"

            echo "============================================================"
            echo "MODEL: ${model_path}"
            echo "TASK : ${task}"
            echo "SEED : ${seed}"
            echo "OUT  : ${output_dir}"
            echo "============================================================"

            "${ACCELERATE_BIN:-accelerate}" launch main.py \
                --model "${model_path}" \
                --tasks "${task}" \
                --seed "${seed}" \
                --max_length_generation 4096 \
                --precision bf16 \
                --temperature 0.2 \
                --n_samples 10 \
                --batch_size 10 \
                --trust_remote_code \
                --allow_code_execution \
                --save_generations \
                --save_generations_path "${output_dir}/generations.json" \
                --metric_output_path "${output_dir}/metrics.json"
        done
    done
done
