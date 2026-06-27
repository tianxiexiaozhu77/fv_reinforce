#!/bin/bash
source activate fv
export CUDA_VISIBLE_DEVICES="2"

# datasets=('antonym' 'capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural')
datasets=('english-french' 'present-past' 'singular-plural')
cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    python /storage/3090group2/hyl/function_vectors/src/evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="/storage/3090group2/hyl/function_vectors/src/results/gptj/gptj_above0.8" --model_name='/storage/3090group2/zjx/model_pt/EleutherAI/gpt-j-6b' --head_selection_threshold="above_0.8"
done