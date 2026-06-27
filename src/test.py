# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
# bernoulli_path = "/opt/data/private/hyl/function_vectors/src/results/gptj/antonym/antonym_bernoullis.pt"
# bernoulli_path = "/opt/data/private/hyl/function_vectors/src/results/llama2-7b/test/antonym/bernoullis.pt"
# bernoulli = torch.load(bernoulli_path)
# print(bernoulli)
# from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
# # model_name = "/opt/data/private/hyl/model/meta-llama/Llama-2-7b-hf"
# model_name = "/opt/data/private/hyl/model/meta-llama/Meta-Llama-3-8B" # 不加：31373，加空23748
# model_name = "/opt/data/private/zjx/ckpt/deepseek-llm-7b-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"
# texts = ["Hello world!", "This is a second sentence.", "Short."]
# out = tokenizer(texts, padding=True, return_tensors="pt")
# print(out)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# model = model.to("cuda:0")
# model_config = model.config # dict(model.named_modules())
# print(model_config)
# ans = " yes"

# ans_tok = tokenizer.encode(ans) # llama:[1, 29871, 22172] gpt
# print(ans_tok)

# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# bernoulli_path = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_reinforce/ag_news/bernoullis.pt"
# bernoulli_path2 = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_reinforce/ag_news/ag_news_bernoullis.pt"
# bernoulli = torch.load(bernoulli_path)
# bernoulli2 = torch.load(bernoulli_path2)

# top_heads = []

# for layer_idx in range(len(bernoulli)):
#     probs = bernoulli[layer_idx]  # shape: [16]
#     for head_idx in range(probs.size(0)):
#         layer_tensor = torch.tensor(layer_idx)
#         head_tensor = torch.tensor(head_idx)
#         prob = probs[head_idx].item() if isinstance(probs[head_idx], torch.Tensor) else probs[head_idx]
#         top_heads.append((layer_tensor, head_tensor, prob, -1))

# print(len(bernoulli))
# print(top_heads)

# save_path = "/opt/data/private/hyl/function_vectors/src/results/gpt_neox/reinforce/antonym/bernoullis_heatmap.png"
# plt.figure(figsize=(8, 6))
# sns.heatmap(bernoulli, annot=False, cmap='viridis', square=True,
#             cbar=True, xticklabels=False, yticklabels=False)

# plt.title("Bernoulli Head Probabilities (64x44)", fontsize=14)
# plt.xlabel("Head index")
# plt.ylabel("Layer index")
# plt.tight_layout()
# plt.savefig(save_path)
# plt.show()

# bernoulli_path = "/opt/data/private/hyl/function_vectors/src/results/llama2-7b/test/antonym/bernoullis.pt"
# bernoulli = torch.load(bernoulli_path) # (32, 32)
# 两次平均
# flat = bernoulli.flatten()
# mean1 = flat.mean()
# mask1 = flat > mean1
# filtered1 = flat[mask1] # 402个

# mean2 = filtered1.mean()
# mask2 = filtered1 > mean2
# filtered2 = filtered1[mask2] # 236个

# indices_lvl1 = torch.nonzero(mask1, as_tuple=False).squeeze()
# final_indices = indices_lvl1[mask2]

# rows = final_indices // bernoulli.size(1)
# cols = final_indices % bernoulli.size(1)

# 一次平均
# flat = bernoulli.flatten()
# mean = flat.mean()
# mask = flat > mean
# indices = torch.nonzero(mask, as_tuple=False).squeeze() # 402个

# rows = indices // bernoulli.size(1)
# cols = indices % bernoulli.size(1)

# results = [(rows[i], cols[i], -1) for i in range(len(rows))]

# print(results)

activation1 = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_all_layer/synonym/bernoullis.pt"
# activation2 = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_ori/ag_news/ag_news_mean_head_activations.pt"

a1 = torch.load(activation1)

print(a1.shape)