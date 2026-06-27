#!/bin/bash
export CUDA_VISIBLE_DEVICES="0, 3" 
# datasets=('capitalize_first_letter' 'prev_item' 'capitalize_last_letter' 'lowercase_last_letter' 'word_length' )
# datasets=('antonym' 'capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural')
cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    # python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/llama3-8b/ori" --model_name='/opt/data/private/hyl/model/meta-llama/Meta-Llama-3-8B' --n_top_heads='30'
    # python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/deepseek-7b/ori" --model_name='/opt/data/private/zjx/ckpt/deepseek-llm-7b-base' --n_top_heads='20'
    python evaluate_function_vector.py --dataset_name="${d_name}" --save_path_root="results/gptj/gptj_ori" --model_name='/opt/data/private/zjx/ICL/inform_zjx/EleutherAI/gpt-j-6b' --n_top_heads='10'
done