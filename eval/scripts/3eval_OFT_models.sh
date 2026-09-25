#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

cd lm-evaluation-harness

models=( )

tasks=(
    commonsense_qa 
     social_iqa
    minerva_math500   
     arc_challenge_mt_da  arc_challenge_mt_de  arc_challenge_mt_el  arc_challenge_mt_es  arc_challenge_mt_fi      arc_challenge_mt_hu
    arc_challenge_mt_it    arc_challenge_mt_nb   arc_challenge_mt_pl  arc_challenge_mt_pt 
     arc_challenge_mt_sv     arc_challenge_mt_is
    agieval

)

for model_path in "${models[@]}"; do
    model_name=$(basename ${model_path})

    for task in "${tasks[@]}"; do
        output_path=results/${model_name}/${task}
        mkdir -p ${output_path}

        echo "============================================================"
        echo "MODEL: ${model_path}"
        echo "TASK : ${task}"
        echo "OUT  : ${output_path}"
        echo "============================================================"

        "${PYTHON_BIN:-python}" -m lm_eval \
            --model hf \
            --model_args pretrained=${model_path},dtype=bfloat16,trust_remote_code=True \
            --tasks ${task} \
            --device cuda:0 \
            --batch_size 8 \
            --output_path ${output_path} \
            --trust_remote_code \
            --confirm_run_unsafe_code
    done
done
