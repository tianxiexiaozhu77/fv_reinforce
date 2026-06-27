# Function Vectors with REINFORCE Head Selection

先从少样本上下文中提取各注意力头的平均激活，再选择对任务敏感的注意力头，将其输出投影后求和得到任务向量，并在下游推理时把该向量注入模型指定层。当前代码同时保留了原始 indirect effect 选头流程，并新增了基于 REINFORCE 的注意力头重要性学习流程。



## 项目结构

```text
.
├── dataset_files/                 # 任务数据
│   ├── abstractive/               # 答案不直接出现在 prompt 中的任务
│   ├── extractive/                # 答案可从 prompt 中抽取的任务
│   └── algebra/                   # 向量代数实验数据
├── notebooks/                     # 示例 notebook
├── src/
│   ├── evaluate_function_vector.py            # 原始 indirect-effect Function Vector 评测
│   ├── evaluate_function_vector_reinforce.py  # REINFORCE 选头版 Function Vector 评测
│   ├── run_reinforce.py                       # REINFORCE 训练注意力头选择概率
│   ├── compute_average_activations.py         # 单独计算平均注意力头激活
│   ├── compute_indirect_effect.py             # 单独计算 indirect effect
│   ├── vector_algebra.py                      # Function Vector 向量代数实验
│   ├── portability_eval.py                    # prompt 模板迁移性评测
│   ├── eval_scripts/                          # 常用运行脚本
│   └── utils/                                 # 数据、prompt、模型加载、干预与评测工具
├── requirements.txt
├── requirement_1.txt
└── fv_environment.yml
```

## 环境配置

推荐使用 Python 3.10，并在 GPU 环境下运行。大模型实验需要提前准备好 HuggingFace 模型权重或本地模型路径。

使用 conda：

```bash
conda env create -f fv_environment.yml
conda activate fv
pip install git+https://github.com/davidbau/baukit@main#egg=baukit
```

或使用 pip：

```bash
pip install -r requirements.txt
pip install git+https://github.com/davidbau/baukit@main#egg=baukit
```

如果只需要安装 PyTorch/torchvision，可参考 `requirement_1.txt`。实际完整运行仍需要 `transformers`、`datasets`、`baukit`、`scikit-learn` 等依赖。

## 支持模型

模型通过 `src/utils/model_utils.py` 中的 `load_gpt_model_and_tokenizer` 加载。目前代码中适配了以下模型族：

- `gpt2-xl`
- GPT-J，如 `EleutherAI/gpt-j-6b`
- GPT-NeoX / Pythia
- LLaMA / Llama-2 / Llama-3
- DeepSeek LLM
- Gemma
- OLMo
- Qwen

不同模型的目标 token 下标、hook 层名和输出投影层略有差异，新增模型时需要同步检查 `model_utils.py`、`run_reinforce.py` 和 `utils/extract_utils.py` 中的模型分支。

## 数据格式

任务数据位于 `dataset_files/`，每个任务通常是一个 JSON 文件。默认加载逻辑会根据任务名在 `abstractive/` 和 `extractive/` 下查找对应数据；向量代数实验使用 `dataset_files/algebra/` 下的数据。

常用任务示例：

```text
antonym
capitalize
country-capital
english-french
present-past
singular-plural
product-company
sentiment
```

## 快速运行：REINFORCE 选头版

这是当前仓库的主要实验入口。下面示例会在 `antonym` 任务上运行 REINFORCE 注意力头选择、构造 Function Vector，并对所有层做 zero-shot intervention 和 shuffled few-shot intervention 评测。

```bash
cd src

python evaluate_function_vector_reinforce.py \
  --dataset_name antonym \
  --model_name /path/to/EleutherAI/gpt-j-6b \
  --root_data_dir /storage/3090group2/hyl/function_vectors/dataset_files \
  --save_path_root /storage/3090group2/hyl/function_vectors/src/results/gptj/gptj_above0.8 \
  --head_selection_threshold above_0.8 \
  --n_shots 10 \
  --n_mean_activations_trials 100 \
  --edit_layer -1
```

也可以直接修改并运行已有脚本，例如：

```bash
cd src/eval_scripts/gptj
bash eval_fv_gptj6_0.8.sh
```

注意：脚本里包含本机路径和 `CUDA_VISIBLE_DEVICES`，在新机器上运行前需要改成自己的模型路径、结果目录和 GPU 编号。

## REINFORCE 流程说明

`src/run_reinforce.py` 中的流程如下：

1. 为每个注意力头维护一个可学习参数 `theta_lh`，初始化为 `-1`。
2. 使用 `sigmoid(theta_lh)` 得到伯努利采样概率。
3. 每轮训练从训练集采样 `n_shots` 个示例，从验证集采样查询样本。
4. 每轮采样 50 个二值注意力头掩码。
5. 对被选中的注意力头位置执行平均激活替换，计算目标 token 的交叉熵损失。
6. 对 50 个 loss 做标准化，然后用 `log_prob * normalized_loss` 构造 policy loss。
7. 只更新注意力头选择参数，不更新原始语言模型参数。

默认超参数写在 `reinforce()` 内部：

```text
learning rate = 0.1
epochs = 600
samples per update = 50
optimizer = Adam
```

训练完成后，脚本会保存伯努利概率矩阵，并根据 `--head_selection_threshold` 选择需要参与 Function Vector 构造的注意力头。

## 选头阈值策略

`evaluate_function_vector_reinforce.py` 支持以下阈值：

```text
above_mean_once
above_mean_twice
above_mean_thrice
above_mean_forth
above_0.5
above_0.6
above_0.7
above_0.8
above_0.9
```

其中 `above_mean_*` 会对伯努利概率递归执行“高于当前均值”的筛选；`above_0.x` 则直接选择概率大于指定阈值的注意力头。

## 原始 Function Vector 流程

如果要使用 indirect effect 排序选头，可以运行：

```bash
cd src

python evaluate_function_vector.py \
  --dataset_name antonym \
  --model_name /path/to/EleutherAI/gpt-j-6b \
  --root_data_dir /storage/3090group2/hyl/function_vectors/dataset_files \
  --save_path_root /storage/3090group2/hyl/function_vectors/src/results/gptj/original_fv \
  --n_top_heads 10 \
  --n_shots 10 \
  --n_mean_activations_trials 100 \
  --n_indirect_effect_trials 25 \
  --edit_layer -1
```

该流程会先计算平均注意力头激活，再计算每个注意力头的 indirect effect，并选择 top-k 注意力头构造 Function Vector。

## 单独计算中间结果

计算平均注意力头激活：

```bash
cd src

python compute_average_activations.py \
  --dataset_name antonym \
  --model_name /path/to/EleutherAI/gpt-j-6b \
  --root_data_dir ../dataset_files \
  --save_path_root ../results \
  --n_shots 10 \
  --n_trials 100
```

计算 indirect effect：

```bash
cd src

python compute_indirect_effect.py \
  --dataset_name antonym \
  --model_name /path/to/EleutherAI/gpt-j-6b \
  --root_data_dir ../dataset_files \
  --save_path_root ../results \
  --n_shots 10 \
  --n_trials 25
```

## 向量代数实验

`src/vector_algebra.py` 用于测试类似 `bd* = ad + bc - ac` 的 Function Vector 组合效果。示例脚本：

```bash
cd src/eval_scripts/ablation/vector_algebra
bash eval_vector_algebra.sh
```

对应数据位于：

```text
dataset_files/algebra/<task_name>/
```

运行前同样需要检查脚本中的模型路径、结果路径和 GPU 设置。

## 主要输出文件

默认结果会写入：

```text
<save_path_root>/<dataset_name>/
```

常见输出包括：

- `fs_results_layer_sweep.json`：无干预 few-shot 基线结果，用于筛选模型本来能答对的样本。
- `<dataset_name>_mean_head_activations.pt`：平均注意力头激活。
- `bernoullis.pt`：REINFORCE 学到的注意力头选择概率矩阵。
- `<dataset_name>_bernoullis.pt` 或 `<dataset_name>_top_heads.pt`：按阈值筛出的注意力头位置。
- `zs_results_layer_sweep.json`：zero-shot 注入 Function Vector 的逐层评测结果。
- `fs_shuffled_results_layer_sweep.json`：few-shot label shuffled 场景下注入 Function Vector 的逐层评测结果。
- `model_baseline.json`：模型 0-shot 到 n-shot 的基线结果。
- `fv_eval_args.txt`：本次运行参数。

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `--dataset_name` | 任务名，例如 `antonym`、`country-capital` |
| `--model_name` | HuggingFace 模型名或本地模型路径 |
| `--root_data_dir` | 数据根目录 |
| `--save_path_root` | 结果保存根目录 |
| `--ie_path_root` | 读取已有中间结果的根目录；为空时使用 `save_path_root` |
| `--n_shots` | prompt 中的 demonstration 数量，默认 10 |
| `--edit_layer` | 注入 Function Vector 的层；`-1` 表示遍历所有层 |
| `--n_mean_activations_trials` | 计算平均激活时采样的 prompt 数 |
| `--n_indirect_effect_trials` | 计算 indirect effect 时采样的 prompt 数 |
| `--head_selection_threshold` | REINFORCE 概率选头策略 |
| `--generate_str` | 使用生成式字符串指标评测，而不是 token rank/ppl |
| `--metric` | 生成式评测指标，默认 `f1_score` |
| `--revision` | 指定 HuggingFace checkpoint revision |

## 注意事项

- 大模型评测显存开销较高，建议先用单个任务和较小 trial 数检查流程。
- 当前部分脚本使用绝对路径，需要按自己的环境修改。
- `evaluate_function_vector_reinforce.py` 中 `bernoullis_save_path` 默认为 `<ie_path_root>/bernoullis.pt`，同一结果目录下不同任务可能会共享这个文件；如果希望每个任务独立保存概率，需要检查或修改保存路径。
- 代码中 `compute_baseline` 使用 `type=bool`，命令行传入字符串时可能不符合直觉；如需关闭基线计算，建议在代码或脚本中显式确认。
- REINFORCE 训练只优化注意力头选择参数，不会微调原始语言模型权重。
