#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

datasets=('antonym' 'capitalize')

cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    # python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/qwen3-8b/all_layer" --model_name='/opt/data/private/zjx/ICL/inform_zjx/Qwen/Qwen3-8B'
    # python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/gpt_neox/reinforce" --model_name='/opt/data/private/zjx/ckpt/gpt-neox-20b'
    # python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/deepseek-7b/all_layer" --model_name='/opt/data/private/zjx/ckpt/deepseek-llm-7b-base'
    # python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/gptj/gptj_all_layer_average" --model_name='/opt/data/private/zjx/ICL/inform_zjx/EleutherAI/gpt-j-6b'
    # python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/llama2-7b/average_one" --model_name='/opt/data/private/hyl/model/meta-llama/Llama-2-7b-hf'
    python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/gptj/gptj_average_one" --model_name='/opt/data/private/zjx/ICL/inform_zjx/EleutherAI/gpt-j-6b'
done