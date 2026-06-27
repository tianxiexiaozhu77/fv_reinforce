import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import os

# 自定义 colormap：截取 viridis 的 10% ~ 100%，增强亮色部分
orig_cmap = cm.get_cmap("viridis")
new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "trunc_viridis", orig_cmap(np.linspace(0.1, 1, 256))
)

# 主目录路径，包含多个子文件夹（如 singular-plural, color_v_animal 等）
root_directory = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_all_layer"

# 遍历 root_directory 下的所有子文件夹
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)

    # 只处理文件夹
    if not os.path.isdir(folder_path):
        continue

    # 构建 bernoullis.pt 路径
    bernoullis_path = os.path.join(folder_path, "bernoullis.pt")
    
    if not os.path.exists(bernoullis_path):
        print(f"  File not found: {bernoullis_path}")
        continue

    print(f"Processing: {bernoullis_path}")

    try:
        # 加载数据
        bernoullis = torch.load(bernoullis_path)  # 假设 shape 是 [28, 16] 或 [25, 28, 16]
        
        # 转置为 [16, 28]
        mean_bernoullis = bernoullis.T  # 如果是 3D 张量，可能需要 .mean(dim=0).T；根据实际 shape 调整

    except Exception as e:
        print(f" Error loading or processing {bernoullis_path}: {e}")
        continue

    # 绘制热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(mean_bernoullis, annot=False, cmap=new_cmap)
    plt.title(f"Heatmap of bernoullis (Shape {mean_bernoullis.shape})")

    # 保存路径：{子文件夹名}_bernoullis_zjx.pdf
    save_path = os.path.join(folder_path, f"{folder_name}_bernoullis_zjx.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f" Saved: {save_path}")

print(" All 'bernoullis.pt' files processed.")
