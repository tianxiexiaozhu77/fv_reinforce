import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.patches import Rectangle

# 主目录路径（包含多个子文件夹，每个子文件夹中有一个 *_indirect_effect.pt 文件）
root_directory = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_ori"
# root_directory = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_all_layer"

# 遍历 root_directory 下的所有文件，找到以 '_indirect_effect.pt' 结尾的文件
for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.endswith("_indirect_effect.pt"):
            # 构建完整文件路径
            base_dir = os.path.join(dirpath, filename)
            
            # 提取任务名称（如 color_v_animal_3），用于保存文件名
            task_name = filename.replace("_indirect_effect.pt", "")
            
            # 输出目录即为当前文件所在目录
            directory_path = dirpath

            print(f"Processing: {base_dir}")

            # 加载数据
            indirect_effect = torch.load(base_dir)
            
            # 计算均值并转置
            mean_indirect_effect = indirect_effect.mean(dim=0)  # [28, 16]
            mean_indirect_effect = mean_indirect_effect.T       # [16, 28]

            # ------------------------ 第一张图：普通热力图 ------------------------
            plt.figure(figsize=(10, 6))
            sns.heatmap(mean_indirect_effect, annot=False, cmap="viridis")
            plt.title("Heatmap of Mean Indirect Effect (Shape [16, 28])")
            heatmap_path = os.path.join(directory_path, f"{task_name}_heatmap_mean_indirect_effect_zjx.pdf")
            plt.savefig(heatmap_path, bbox_inches='tight')
            plt.close()
            print(f"Saved: {heatmap_path}")

            # ------------------------ 第二张图：高亮 top 10 ------------------------
            plt.figure(figsize=(10, 6))
            sns.heatmap(mean_indirect_effect, annot=False, cmap="viridis")

            # 获取 top 10 值的位置
            top_k = 10
            flat_array = mean_indirect_effect.numpy().flatten()
            top_k_indices_flat = np.argsort(flat_array)[-top_k:]  # 最大的 10 个值的扁平索引
            top_k_indices = np.unravel_index(top_k_indices_flat, mean_indirect_effect.shape)  # 转为二维坐标

            ax = plt.gca()
            cell_width = 1
            cell_height = 1

            # 在热力图上画红框
            for idx in range(top_k):
                row, col = top_k_indices[0][idx], top_k_indices[1][idx]
                rect = Rectangle((col, row), cell_width, cell_height,
                                 linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                print(f"{task_name} - Top {idx+1} - Position: (Row: {row}, Col: {col}), Value: {mean_indirect_effect[row, col]:.4f}")

            plt.title("Heatmap of Mean Indirect Effect (Shape [16, 28]) with Top 10 Values Highlighted")
            highlight_path = os.path.join(directory_path, f"{task_name}_heatmap_mean_indirect_effect_with_highlight_zjx.pdf")
            plt.savefig(highlight_path, bbox_inches='tight')
            plt.close()
            print(f"Saved: {highlight_path}")

print("All files processed.")