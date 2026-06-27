import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.extract_utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, logging

def reinforce_activation_replacement(prompt_data, avg_activations, dummy_labels, sampled, model, model_config, tokenizer, last_token_only=True):

    """
    This function performs Reinforce to select the attentions that encodes ICL examples.

    Parameters:
    model_input: Input to the forward function. Refer to model.py
    avg_activations:get_last_mean_head_activations
    model_helper:
    sampeld:
    last_token_only:

    Returns: 
    output: The logit of the first output token
    """
    device = model.device

    query_target_pair = prompt_data['query_target']

    query = query_target_pair['input']
    token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query=query, prepend_bos=model_config['prepend_bos'])

    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)
    idx_map = update_idx_map(idx_map, idx_avg)
      
    sentences = [prompt_string]# * model.config.n_head # batch things by head

    inputs = tokenizer(sentences, return_tensors='pt').to(device)

    # Speed up computation by only computing causal effect at last token
    if last_token_only:
        token_classes = ['query_predictive']
        token_classes_regex = ['query_predictive_token']
    # Compute causal effect for all token classes (instead of just last token)
    else:
        token_classes = ['demonstration', 'label', 'separator', 'predictive', 'structural','end_of_example', 
                        'query_demonstration', 'query_structural', 'query_separator', 'query_predictive']
        token_classes_regex = ['demonstration_[\d]{1,}_token', 'demonstration_[\d]{1,}_label_token', 'separator_token', 'predictive_token', 'structural_token','end_of_example_token', 
                            'query_demonstration_token', 'query_structural_token', 'query_separator_token', 'query_predictive_token']
    
    ###This function returns a list of locations to perform intervention on based on sampled. List((layer, head, token_idx)). Token_idx is default to -1, meaning we always perform intervention on the generated token
    for i,(token_class, class_regex) in enumerate(zip(token_classes, token_classes_regex)):
        reg_class_match = re.compile(f"^{class_regex}$")
        class_token_inds = [x[0] for x in token_labels if reg_class_match.match(x[2])]
        intervention_locations = reinforce_intervention_location(sampled, token_idx=class_token_inds) # 返回sampled中所有非零的（层，头，-1）干预位置


        intervention_fn = replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                    model=model, model_config=model_config,
                                                    batched_input=False, idx_map=idx_map, last_token_only=last_token_only)

        with TraceDict(model, layers=model_config['attn_hook_names'], edit_output=intervention_fn, retain_grad=True) as td: # edit_output接收每一层的输出并对其进行操作
            output = model.forward(**inputs).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction

        return output # [1, 50400]

def reinforce(model, mean_activations, model_config, dataset, tokenizer, n_shots=10, last_token_only=True, prefixes=None, separators=None, filter_set=None):

    """
    This function performs Reinforce to select the attentions that encodes ICL examples.

    Parameters:
    mean_activations: From get_last_mean_head_activations
    model_helper:
    reinforce_data: Dataset used during reinforce optimization
    eval_data: Dataset used for Validation

    Returns: 
    bernoullis: A tensor of bernoullis variable. One variable for each attention heads. Each denote the probability of selecting this attention head.
    """

    num_layer = model_config["n_layers"]
    num_heads = model_config["n_heads"]
    lr = 0.1
    eps = 1e-3
    epoch = 600 # 600
    n_test_examples = 1

    #(num_layer, num_head)为每一层设置可学习的参数,初始化为-1
    bernoullis = [torch.neg(torch.ones(num_heads)).requires_grad_() for _ in range(num_layer)] 
    optim = torch.optim.Adam(bernoullis, lr=lr)

    if prefixes is not None and separators is not None:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, prefixes=prefixes, separators=separators, model_config=model_config)
    else:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, model_config=model_config)

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos = False if model_config['prepend_bos'] else True

    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))

    with torch.set_grad_enabled(True):
        progress_bar = tqdm(range(epoch), total=epoch, desc="Training", dynamic_ncols=True)
        for epoch in progress_bar:
            
            loss_list = []
            saved_log_probs = []
            saved_entropies = []

            word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]
            word_pairs_test = dataset['valid'][np.random.choice(filter_set,n_test_examples, replace=False)]

            if prefixes is not None and separators is not None:
                prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, shuffle_labels=True, 
                                                            prepend_bos_token=prepend_bos, prefixes=prefixes, separators=separators)
            else:
                prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, 
                                                           shuffle_labels=True, prepend_bos_token=prepend_bos)

            target_out = prompt_data_random['query_target']['output']

            target_token = tokenizer(target_out, return_tensors='pt')["input_ids"][0].to("cuda") # [1, 29871,...] 1:<s>,29871:''
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor) # 创建伯努利分布对象，sigmoid_tensor中每个元素表示成功的概率

            ###Sampling the distribution many times to reduce variance.
            for _ in range(50):

                ##Current sample
                sampled = prob_dist.sample() # 采样(28,16)
                saved_log_probs.append(prob_dist.log_prob(sampled)) # sampled 张量中每个样本的对数概率
                # saved_entropies.append(prob_dist.entropy())

                with torch.no_grad():
                    out_logit = reinforce_activation_replacement(prompt_data=prompt_data_random, avg_activations = mean_activations,dummy_labels=dummy_gt_labels, sampled=sampled, model=model, model_config=model_config, tokenizer=tokenizer, last_token_only=last_token_only)
                    if 'gpt-j' in model_config['name_or_path'].lower():
                        task_loss = torch.nn.functional.cross_entropy(out_logit, target_token[0].unsqueeze(0))
                    elif 'gpt-neox' in model_config['name_or_path'].lower():
                        task_loss = torch.nn.functional.cross_entropy(out_logit, target_token[0].unsqueeze(0))
                    elif 'llama' in model_config['name_or_path'].lower():
                        if '7b' in model_config['name_or_path'].lower():
                            task_loss = torch.nn.functional.cross_entropy(out_logit, target_token[2].unsqueeze(0))
                        elif '8b' in model_config['name_or_path'].lower():
                            task_loss = torch.nn.functional.cross_entropy(out_logit, target_token[1].unsqueeze(0))
                    elif 'qwen' in model_config['name_or_path'].lower():
                        task_loss = torch.nn.functional.cross_entropy(out_logit, target_token[0].unsqueeze(0))
                    elif 'deepseek' in model_config['name_or_path'].lower():
                        task_loss = torch.nn.functional.cross_entropy(out_logit, target_token[1].unsqueeze(0))
                    # task_loss = torch.nn.functional.cross_entropy(out_logit, target_token[2].unsqueeze(0))
                    loss_list.append(task_loss)

            #print(model_helper.tokenizer.decode(out_logit[0].argmax(dim=-1)), model_helper.tokenizer.decode(target_token[0]), flush=True)

            policy_loss = []
            loss_list = torch.tensor(loss_list) # [32]
            loss_list = (loss_list - loss_list.mean())/(loss_list.std() + eps) # 标准化

            beta = 0.01 # 熵的权重
            # for log_prob, entropy, R in zip(saved_log_probs, saved_entropies, loss_list):
            for log_prob, R in zip(saved_log_probs, loss_list):
                policy_loss.append(log_prob * R) # 策略梯度方法：loss = -log(p(a|s))*R,R为奖励值，现在的R是 loss
                # loss = (log_prob * R) - beta * entropy
                # policy_loss.append(loss)

            optim.zero_grad()
            policy_loss = torch.cat(policy_loss).sum() # cat后[1024,32] .sum->[1]
            policy_loss.backward()
            optim.step()
            torch.cuda.empty_cache()

            avg_loss = sum(loss_list) / len(loss_list)
            progress_bar.set_postfix(loss=avg_loss)
            # if epoch % 50 == 0:
            #     validate_reinforce(model=model, bernoullis=bernoullis, eps=eps, mean_activations=mean_activations, dataset=dataset, epoch=epoch)
    return bernoullis

def validate_reinforce(model, model_config, tokenizer, bernoullis, eps, mean_activations, dataset, epoch, n_shots=10, last_token_only=True, sampled=None, prefixes=None, separators=None, filter_set=None):

    if prefixes is not None and separators is not None:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, prefixes=prefixes, separators=separators, model_config=model_config)
    else:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, model_config=model_config)

    n_test_examples = 1
    prepend_bos = False if model_config['prepend_bos'] else True
    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))

    with torch.no_grad():
        if sampled is None:
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
            sampled = prob_dist.sample()

        loss_list = []
        for _ in range(50):
            word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]
            word_pairs_test = dataset['valid'][np.random.choice(filter_set,n_test_examples, replace=False)]

            if prefixes is not None and separators is not None:
                prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, shuffle_labels=True, 
                                                            prepend_bos_token=prepend_bos, prefixes=prefixes, separators=separators)
            else:
                prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, 
                                                           shuffle_labels=True, prepend_bos_token=prepend_bos)

            target_out = prompt_data_random['query_target']['output']
            target_token = tokenizer(target_out, return_tensors='pt')["input_ids"][0].to("cuda")


            out_logit = reinforce_activation_replacement(prompt_data=prompt_data_random, avg_activations = mean_activations,dummy_labels=dummy_gt_labels, sampled=sampled, model=model, model_config=model_config, tokenizer=tokenizer, last_token_only=last_token_only)
            if 'gpt-j' in model_config['name_or_path'].lower():
                task_loss = torch.nn.functional.cross_entropy(out_logit, target_token[0].unsqueeze(0))
            elif 'llama' in model_config['name_or_path'].lower():
                if '7b' in model_config['name_or_path'].lower():
                    task_loss = torch.nn.functional.cross_entropy(out_logit, target_token[2].unsqueeze(0))
                elif '8b' in model_config['name_or_path'].lower():
                    task_loss = torch.nn.functional.cross_entropy(out_logit, target_token[1].unsqueeze(0))
            elif 'deepseek' in model_config['name_or_path'].lower():
                task_loss = torch.nn.functional.cross_entropy(out_logit, target_token[1].unsqueeze(0))

            loss_list.append(task_loss)


        print(f"validation loss at {epoch} epoch:", torch.tensor(loss_list).mean())
    return torch.tensor(loss_list).mean().item()

def reinforce_intervention_location(sampled, token_idx = [-1]):
    intervention_locations = []
    #(layer, head)

    patch_idx = torch.nonzero(sampled) # [260,2]返回非零元素
    count = 0
    for _ in patch_idx:
        cur_layer = _[0]
        cur_head = _[1]
        intervention_locations.append((cur_layer, cur_head, token_idx[0]))

    return intervention_locations
