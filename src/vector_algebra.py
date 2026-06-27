import os, json
import torch, numpy as np
import argparse

# Include prompt creation helper functions
from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *
from compute_indirect_effect import compute_indirect_effect
from run_reinforce import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_names', help='Name of the datasets to be loaded', type=str, required=True)
    parser.add_argument('--dataset_folder', help='Name of the dataset_folder to be loaded', type=str, required=True)
    parser.add_argument('--n_top_heads', help='Number of attenion head outputs used to compute function vector', required=False, type=int, default=10)
    parser.add_argument('--edit_layer', help='Layer for intervention. If -1, sweep over all layers', type=int, required=False, default=-1) # 
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='/storage/3090group2/hyl/function_vectors/dataset_files/algebra')
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='/storage/3090group2/hyl/function_vectors/src/results')
    parser.add_argument('--ie_path_root', help='File path to load indirect effects from', type=str, required=False, default=None)
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mean_activations_path', help='Path to file containing mean_head_activations for the specified task', required=False, type=str, default=None)
    parser.add_argument('--indirect_effect_path', help='Path to file containing indirect_effect scores for the specified task', required=False, type=str, default=None)    
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)    
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=10)
    parser.add_argument('--n_mean_activations_trials', help="Number of in-context prompts to average over for mean_activations", type=int, required=False, default=100)
    parser.add_argument('--n_indirect_effect_trials', help="Number of in-context prompts to average over for indirect_effect", type=int, required=False, default=25)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
    parser.add_argument('--compute_baseline', help='Whether to compute the model baseline 0-shot -> n-shot performance', type=bool, required=False, default=True)
    parser.add_argument('--generate_str', help='Whether to generate long-form completions for the task', action='store_true', required=False)
    parser.add_argument("--metric", help="Metric to use when evaluating generated strings", type=str, required=False, default="f1_score")
    parser.add_argument("--universal_set", help="Flag for whether to evaluate using the univeral set of heads", action="store_true", required=False)
    parser.add_argument('--revision', help='Specify model checkpoints for pythia or olmo models', type=str, required=False, default=None)
    parser.add_argument('--head_selection_threshold', help='The different method of selecting intervention heads', type=str, required=False, default='above_mean_once', choices=['above_mean_once', 'above_mean_twice', 'above_mean_thrice', 'above_0.5', 'above_mean_forth', 'above_0.6', 'above_0.7', 'above_0.8', 'above_0.9'])
        
    args = parser.parse_args() 

    dataset_names = [name.strip() for name in args.dataset_names.strip('[]').split(',')] # [ac, ad, bc, bd]
    dataset_folder = args.dataset_folder

    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{dataset_names[-1]}"
    ie_path_root = f"{args.ie_path_root}/{dataset_names[-1]}" if args.ie_path_root else save_path_root
    seed = args.seed
    device = args.device
    mean_activations_path = args.mean_activations_path
    indirect_effect_path = args.indirect_effect_path
    n_top_heads = args.n_top_heads
    eval_edit_layer = args.edit_layer

    test_split = float(args.test_split)
    n_shots = args.n_shots
    n_mean_activations_trials = args.n_mean_activations_trials
    n_indirect_effect_trials = args.n_indirect_effect_trials

    prefixes = args.prefixes 
    separators = args.separators
    compute_baseline = args.compute_baseline

    generate_str = args.generate_str
    metric = args.metric
    universal_set = args.universal_set

     # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, revision=args.revision)

    eval_edit_layer = [0, model_config['n_layers']]

    vectors = []
    for i, dataset_name in enumerate(dataset_names):
        # Load the ac dataset
        print("Loading Dataset")
        set_seed(seed)
        dataset = load_dataset2(dataset_name, dataset_folder=dataset_folder, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)

        print(f"Filtering Dataset via {n_shots}-shot Eval")

        fs_results_file_name = f'{save_path_root}/{dataset_name}_fs_results_layer_sweep.json'
        print(fs_results_file_name)
        if os.path.exists(fs_results_file_name):
            with open(fs_results_file_name, 'r') as indata:
                fs_results = json.load(indata)
            key = 'score' if generate_str else 'clean_rank_list'
            target_val = 1 if generate_str else 0
            filter_set = np.where(np.array(fs_results[key]) == target_val)[0]
            filter_set_validation = None
        else:
            set_seed(seed+42)
            fs_results_validation = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=True, test_split='valid', prefixes=prefixes, separators=separators)
            filter_set_validation = np.where(np.array(fs_results_validation['clean_rank_list']) == 0)[0]
            set_seed(seed)
            fs_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=True, prefixes=prefixes, separators=separators)
            filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]

        args.fs_results_file_name = fs_results_file_name
        with open(fs_results_file_name, 'w') as results_file:
            json.dump(fs_results, results_file, indent=2)

        set_seed(seed)

        # Load or Re-Compute mean_head_activations
        mean_activations_path = f'{ie_path_root}/{dataset_name}_mean_head_activations.pt'
        if mean_activations_path is not None and os.path.exists(mean_activations_path):
            mean_activations = torch.load(mean_activations_path)
        # elif mean_activations_path is None and os.path.exists(f'{ie_path_root}/{dataset_name}_mean_head_activations.pt'):
        #     mean_activations_path = f'{ie_path_root}/{dataset_name}_mean_head_activations.pt'
        #     mean_activations = torch.load(mean_activations_path)        
        else:
            print("Computing Mean Activations")
            set_seed(seed)
            mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=n_shots,
                                                        N_TRIALS=n_mean_activations_trials, prefixes=prefixes, separators=separators, filter_set=filter_set_validation)
            args.mean_activations_path = f'{save_path_root}/{dataset_name}_mean_head_activations.pt'
            torch.save(mean_activations, args.mean_activations_path)
        
        bernoullis_path = f"{ie_path_root}/{dataset_name}_top_heads.pt" # 选出的层和头
        bernoullis_save_path = f"{ie_path_root}/{dataset_name}_bernoullis.pt" # 最后的概率

        if os.path.exists(bernoullis_save_path):
            bernoullis = torch.load(bernoullis_save_path)
        else:
            bernoullis_ori = reinforce(model=model, mean_activations=mean_activations, model_config=model_config, dataset=dataset, tokenizer=tokenizer, n_shots=n_shots, last_token_only=True, prefixes=prefixes, separators=separators, filter_set=filter_set_validation)
            bernoullis = torch.stack([torch.sigmoid(bernoulli).clamp(min=0, max=1) for bernoulli in bernoullis_ori]) # 概率
            torch.save(bernoullis, bernoullis_save_path)

        fv = compute_reinforce_function_vector(args, bernoullis_path, bernoullis, mean_activations, model, model_config, tokenizer, dataset)

        vectors.append(fv)

        if i == 3:
            print(f"Running sweep over layers {eval_edit_layer}")
            zs_results = {}
            fs_shuffled_results = {}
            for edit_layer in range(eval_edit_layer[0], eval_edit_layer[1]):
                set_seed(seed)

                zs_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=0, prefixes=prefixes, separators=separators,
                                                    model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set)
                zs_results_file_suffix = f'_layer_sweep.json'
        
            # Save results to files
            zs_results_file_name = make_valid_path_name(f'{save_path_root}/{dataset_name}_zs_results' + zs_results_file_suffix)
            args.zs_results_file_name = zs_results_file_name
            with open(zs_results_file_name, 'w') as results_file:
                json.dump(zs_results, results_file, indent=2)
    
    fv_bd = vectors[1] + vectors[2] - vectors[0] # bd* = ad + bc - ac

    print(f"Running bd* vector sweep over layers {eval_edit_layer}")
    zs_results = {}
    fs_shuffled_results = {}
    for edit_layer in range(eval_edit_layer[0], eval_edit_layer[1]):
        set_seed(seed)

        zs_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv_bd, edit_layer=edit_layer, n_shots=0, prefixes=prefixes, separators=separators,
                                            model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set)
        zs_results_file_suffix = f'_layer_sweep.json'
    
    # Save results to files
    zs_results_file_name = make_valid_path_name(f'{save_path_root}/BD_synthetic_zs_results' + zs_results_file_suffix)
    args.zs_results_file_name = zs_results_file_name
    with open(zs_results_file_name, 'w') as results_file:
        json.dump(zs_results, results_file, indent=2)


