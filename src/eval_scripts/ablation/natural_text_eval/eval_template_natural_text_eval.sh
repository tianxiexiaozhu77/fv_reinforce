# #!/bin/bash
# source activate fv
# export CUDA_VISIBLE_DEVICES="7"

# SAVE_PATH_ROOT="/storage/3090group2/hyl/function_vectors/src/results/ablation/natural_text_eval/gptj_average_3"
# MODEL_NAME="/storage/3090group2/zjx/model_pt/EleutherAI/gpt-j-6b"


# python /storage/3090group2/hyl/function_vectors/src/natural_text_eval.py \
#     --save_path_root="${SAVE_PATH_ROOT}" \
#     --model_name="${MODEL_NAME}"

#!/bin/bash
source activate fv
export CUDA_VISIBLE_DEVICES="0"

SAVE_PATH_ROOT="/storage/3090group2/hyl/function_vectors/src/results/ablation/natural_text_eval/gptj_average_3"
MODEL_NAME="/storage/3090group2/zjx/model_pt/EleutherAI/gpt-j-6b"

python /storage/3090group2/hyl/function_vectors/src/natural_text_eval.py \
    --save_path_root="${SAVE_PATH_ROOT}" \
    --model_name="${MODEL_NAME}" \
    --datasets antonym capitalize 

# country-capital  present-past singular-plural
# english-french