# Run and exactly reproduce qwen2vl results!
# mme as an example
export HF_HOME="~/.cache/huggingface"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip3 install qwen_vl_utils
# use `interleave_visuals=True` to control the visual token position, currently only for mmmu_val and mmmu_pro (and potentially for other interleaved image-text tasks), please do not use it unless you are sure about the operation details.

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_vl \
#     --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=True \
#     --tasks mmmu_pro \
#     --batch_size 1

# mmsi_bench,mindcube,ocrbench,ocrbench_v2
# ,attn_implementation=flash_attention_2
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
export HF_HOME="~/.cache/huggingface"
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args="pretrained=${MODEL_PATH},max_pixels=12845056,interleave_visuals=False,attn_implementation=flash_attention_2" \
    --tasks mmsi_bench,erqa,ocrbench,ocrbench_v2 \
    --batch_size 8

conda activate "${MEDICAL_EVAL_ENV:-huatuogpt}"
cd "${MEDICAL_EVAL_DIR:-./HuatuoGPT-Vision}"
accelerate launch eval.py \
    --data_path "${MEDICAL_EVAL_DATA:-${DIFA_DATA_ROOT:-./data}/Medical_Multimodal_Evaluation_Data/medical_multimodel_evaluation_data.json}" \
    --model_path "${MODEL_PATH}"
