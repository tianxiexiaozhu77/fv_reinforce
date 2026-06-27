#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
# all
# datasets=('ag_news' 'capitalize_second_letter' 'english-german' 'national_parks' 'person-occupation' 'sentiment' 'commonsense_qa' 'english-spanish' 'next_capital_letter' 'person-sport' 'synonym' 'capitalize_first_letter' 'country-currency' 'lowercase_first_letter' 'park-country' 'prev_item' 'capitalize_last_letter' 'lowercase_last_letter' 'person-instrument' 'product-company' 'landmark-country' 'next_item' 'word_length' 'adjective_v_verb_3' 'alphabetically_last_5' 'choose_last_of_3' 'color_v_animal_5' 'conll2003_person' 'squad_val' 'adjective_v_verb_5' 'animal_v_object_3' 'choose_last_of_5' 'concept_v_object_3' 'fruit_v_animal_3' 'verb_v_adjective_3' 'alphabetically_first_3' 'animal_v_object_5' 'choose_middle_of_3' 'concept_v_object_5' 'fruit_v_animal_5' 'verb_v_adjective_5' 'alphabetically_first_5' 'choose_first_of_3' 'choose_middle_of_5' 'conll2003_location' 'object_v_concept_3' 'alphabetically_last_3' 'choose_first_of_5' 'color_v_animal_3' 'conll2003_organization' 'object_v_concept_5')
datasets=('che ountry-capital')
# datasets=('antonym' 'capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural')
cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    # python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/qwen3-8b/all_layer" --model_name='/opt/data/private/zjx/ICL/inform_zjx/Qwen/Qwen3-8B'
    # python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/gpt_neox/reinforce" --model_name='/opt/data/private/zjx/ckpt/gpt-neox-20b'
    # python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/deepseek-7b/all_layer" --model_name='/opt/data/private/zjx/ckpt/deepseek-llm-7b-base'
    # python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/gptj/gptj_all_layer_average" --model_name='/opt/data/private/zjx/ICL/inform_zjx/EleutherAI/gpt-j-6b'
    # python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/llama2-7b/average_one" --model_name='/opt/data/private/hyl/model/meta-llama/Llama-2-7b-hf'
    # python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/llama3-8b/average_one" --model_name='/opt/data/private/hyl/model/meta-llama/Meta-Llama-3-8B'
    python evaluate_function_vector_reinforce.py --dataset_name="${d_name}" --save_path_root="results/ablation/vector_algebra/gptj_average_3" --model_name='/opt/data/private/zjx/ICL/inform_zjx/EleutherAI/gpt-j-6b' --head_selection_threshold="above_mean_thrice"
done