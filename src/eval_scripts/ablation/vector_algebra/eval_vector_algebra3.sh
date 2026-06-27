#!/bin/bash
export CUDA_VISIBLE_DEVICES="4"

cd ../../../

folders=('present-past' 'product-company' 'singular-plural')
for d_name in "${folders[@]}"
do
    echo "Running Script for: ${d_name}"
    dataset_names="[${d_name}_first_copy, first_${d_name}, ${d_name}_last_copy, last_${d_name}]"

    python vector_algebra.py --dataset_names="${dataset_names}" --dataset_folder="${d_name}" --save_path_root="results/ablation/vector_algebra/gptj_average_3" --model_name='/storage/3090group2/zjx/model_pt/EleutherAI/gpt-j-6b'
done