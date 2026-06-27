###################⬇#####################
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import numpy as np

# # 设置绘图样式
# sns.set_style("whitegrid")
# plt.figure(figsize=(448, 10))

# # 主目录路径
# root_directory = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_all_layer"

# # 存储所有序列用于绘图
# all_data = []
# labels = []

# # 遍历所有子文件夹
# for folder_name in os.listdir(root_directory):
#     folder_path = os.path.join(root_directory, folder_name)
    
#     # 只处理文件夹
#     if not os.path.isdir(folder_path):
#         continue

#     bernoullis_path = os.path.join(folder_path, "bernoullis.pt")
#     if not os.path.exists(bernoullis_path):
#         print(f"File not found: {bernoullis_path}")
#         continue

#     print(f"Loading: {bernoullis_path}")
    
#     try:
#         # 加载数据
#         bernoullis = torch.load(bernoullis_path)
        
#         # 转置并展平 28为层数，16为头数
#         mean_bernoullis = bernoullis.T  # [28, 16] -> [16, 28]
#         flattened = mean_bernoullis.flatten().numpy()  # 转为 1D 数组，长度 28*16=448

#         all_data.append(flattened)
#         labels.append(folder_name)  # 用文件夹名作为图例标签

#     except Exception as e:
#         print(f"Error processing {bernoullis_path}: {e}")
#         continue

# # 检查是否有数据
# if len(all_data) == 0:
#     raise ValueError("No valid 'bernoullis.pt' files loaded.")

# # 转为 numpy 数组便于处理
# all_data = np.array(all_data)  # shape: [num_tasks, 448]

# # 绘制每条曲线
# for i in range(all_data.shape[0]):
#     plt.plot(all_data[i], label=labels[i], alpha=0.7, linewidth=1.2)

# # 设置标题和标签
# plt.title("Flattened Bernoullis Matrices (Transposed) Across Tasks", fontsize=16)
# plt.xlabel("Index in Flattened Matrix (Shape [16, 28] → 448)", fontsize=12)
# plt.ylabel("Value", fontsize=12)

# # 图例（如果任务太多，可考虑缩小或滚动）
# if len(labels) <= 20:
#     plt.legend(loc='upper right', fontsize=8)
# else:
#     plt.legend(loc='upper right', fontsize=6, ncol=2)  # 多列显示

# # 自动调整布局
# plt.tight_layout()

# # 保存图像到主目录或指定路径
# output_path = os.path.join(root_directory, "all_bernoullis_flattened_curves.pdf")
# plt.savefig(output_path, dpi=1000, bbox_inches='tight')
# plt.close()

# print(f"All curves plotted and saved to: {output_path}")
###################⬆#####################



###################⬇#####################
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import numpy as np

# # 设置绘图样式
# sns.set_style("whitegrid")
# plt.figure(figsize=(448, 10))  # 极宽图，x轴448点完全展开

# # 主目录路径
# root_directory = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_all_layer"

# # 存储所有序列用于绘图
# all_data = []
# labels = []

# # 遍历所有子文件夹
# for folder_name in os.listdir(root_directory):
#     folder_path = os.path.join(root_directory, folder_name)
    
#     # 只处理文件夹
#     if not os.path.isdir(folder_path):
#         continue

#     bernoullis_path = os.path.join(folder_path, "bernoullis.pt")
#     if not os.path.exists(bernoullis_path):
#         print(f"File not found: {bernoullis_path}")
#         continue

#     print(f"Loading: {bernoullis_path}")
    
#     try:
#         # 加载数据
#         bernoullis = torch.load(bernoullis_path)
        

#         trans_bernoullis = bernoullis.T  # bernoullis.shape:torch.Size([28, 16])   trans_bernoullis.shape:torch.Size([16, 28])
#         flattened = trans_bernoullis.flatten().numpy()  # 长度 16*28 = 448

#         all_data.append(flattened)
#         labels.append(folder_name)

#     except Exception as e:
#         print(f"Error processing {bernoullis_path}: {e}")
#         continue

# # 检查是否有数据
# if len(all_data) == 0:
#     raise ValueError("No valid 'bernoullis.pt' files loaded.")

# # 转为 numpy 数组
# all_data = np.array(all_data)  # shape: [num_tasks, 448]

# # ================================
# # ✅ 计算平均曲线
# # ================================
# mean_curve = np.mean(all_data, axis=0)  # 在所有任务上取均值，shape: [448]

# # ================================
# # ✅ 开始绘图
# # ================================
# # 先画所有个体曲线（浅色细线）
# for i in range(all_data.shape[0]):
#     plt.plot(all_data[i], color='gray', alpha=0.4, linewidth=0.8)

# # ✅ 再画平均线（最后画，确保在最上层）
# plt.plot(mean_curve, 
#          color='red', 
#          linewidth=3.0,        # 加粗
#          label='Mean Across Tasks', 
#          solid_capstyle='round')  # 线头更圆润

# # 设置标题和标签
# plt.title("Flattened Bernoullis Matrices (Transposed) Across Tasks", fontsize=20)
# plt.xlabel("Index in Flattened Matrix (Shape [16, 28] → 448)", fontsize=16)
# plt.ylabel("Value", fontsize=16)

# # 图例：只显示平均线 + 可选个体示例（太多时不显示个体标签）
# plt.legend(loc='upper right', fontsize=12)

# # ⚠️ 注意：figsize=(448,10) 非常大，保存时用 bbox_inches='tight' 很关键
# output_path = os.path.join(root_directory, "all_bernoullis_flattened_curves_with_mean_zjx.pdf")
# plt.savefig(output_path, bbox_inches='tight', format='pdf')  # 高 dpi + tight 裁剪
# plt.close()

# print(f"All curves and mean curve plotted and saved to: {output_path}")
###################⬆#####################

###################⬇#####################
# import torch
# import os
# import numpy as np

# # 设置主目录
# root_directory = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_all_layer"

# # 定义要排除的数据集名称（文件夹名）
# excluded_datasets = {
#     'antonym',
#     'capitalize',
#     'country-capital',
#     'english-french',
#     'present-past',
#     'singular-plural'
# }

# # 存储有效的 bernoullis 张量（未展平，保持 [28, 16] 形状）
# valid_bernoullis_list = []
# included_labels = []

# print(f"Excluding datasets: {sorted(excluded_datasets)}")
# print(f"Processing folders in: {root_directory}")

# # 遍历所有子文件夹
# for folder_name in os.listdir(root_directory):
#     folder_path = os.path.join(root_directory, folder_name)

#     if not os.path.isdir(folder_path):
#         continue

#     # 跳过指定数据集
#     if folder_name in excluded_datasets:
#         print(f"Skipping excluded dataset: {folder_name}")
#         continue

#     bernoullis_path = os.path.join(folder_path, "bernoullis.pt")
#     if not os.path.exists(bernoullis_path):
#         print(f"File not found: {bernoullis_path}")
#         continue

#     print(f"Loading: {bernoullis_path}")
#     try:
#         bernoullis = torch.load(bernoullis_path)  # shape: [28, 16]

#         # 确保是正确形状
#         if bernoullis.shape != (28, 16):
#             print(f"Unexpected shape in {folder_name}: {bernoullis.shape}, skipping.")
#             continue

#         valid_bernoullis_list.append(bernoullis)
#         included_labels.append(folder_name)

#     except Exception as e:
#         print(f"Error loading {bernoullis_path}: {e}")
#         continue

# # 检查是否收集到数据
# if len(valid_bernoullis_list) == 0:
#     raise ValueError("No valid bernoullis.pt files loaded after filtering. Check folder names and data.")

# # 堆叠并计算平均值: list of [28,16] -> mean over tasks
# stacked = torch.stack(valid_bernoullis_list)  # shape: [N, 28, 16]
# average_bernoullis = torch.mean(stacked, dim=0)  # shape: [28, 16]

# # 保存平均向量
# output_pt_path = os.path.join(root_directory, "average_bernoullis.pt")
# torch.save(average_bernoullis, output_pt_path)

# print(f"Successfully computed average over {len(valid_bernoullis_list)} datasets.")
# print(f"Included datasets: {sorted(included_labels)}")
# print(f"Saved average_bernoullis.pt to: {output_pt_path}")
# print(f"Shape of average_bernoullis: {average_bernoullis.shape}")
###################⬆#####################


import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# 设置绘图样式
sns.set_style("whitegrid")

# 主目录路径
root_directory = "/storage/3090group2/hyl/function_vectors/src/results/gptj/gptj_all_layer"

# 存储所有序列用于绘图
all_data = []
labels = []


# include 6 main data
subdirs = ['adjective_v_verb_3', 'adjective_v_verb_5', 'ag_news', 'animal_v_object_3', 'animal_v_object_5','antonym', 'capitalize', 'capitalize_first_letter', 'capitalize_last_letter', 'capitalize_second_letter', 'choose_first_of_3', 'choose_first_of_5', 'choose_last_of_3', 'choose_last_of_5', 'choose_middle_of_3', 'choose_middle_of_5', 'color_v_animal_3', 'color_v_animal_5', 'commonsense_qa', 'concept_v_object_3', 'concept_v_object_5', 'conll2003_location', 'conll2003_organization', 'conll2003_person','country-capital', 'country-currency', 'english-french', 'english-german', 'english-spanish', 'fruit_v_animal_3', 'fruit_v_animal_5', 'landmark-country', 'lowercase_first_letter', 'lowercase_last_letter', 'national_parks',  'next_item', 'object_v_concept_3', 'object_v_concept_5', 'park-country', 'person-instrument', 'person-occupation', 'person-sport', 'present-past', 'prev_item', 'product-company', 'singular-plural', 'synonym', 'verb_v_adjective_3', 'verb_v_adjective_5'] 

# no  main data
# subdirs = ['adjective_v_verb_3', 'adjective_v_verb_5', 'ag_news', 'animal_v_object_3', 'animal_v_object_5',  'capitalize_first_letter', 'capitalize_last_letter', 'capitalize_second_letter', 'choose_first_of_3', 'choose_first_of_5', 'choose_last_of_3', 'choose_last_of_5', 'choose_middle_of_3', 'choose_middle_of_5', 'color_v_animal_3', 'color_v_animal_5', 'commonsense_qa', 'concept_v_object_3', 'concept_v_object_5', 'conll2003_location', 'conll2003_organization', 'conll2003_person', 'country-currency', 'english-german', 'english-spanish', 'fruit_v_animal_3', 'fruit_v_animal_5', 'landmark-country', 'lowercase_first_letter', 'lowercase_last_letter', 'national_parks',  'next_item', 'object_v_concept_3', 'object_v_concept_5', 'park-country', 'person-instrument', 'person-occupation', 'person-sport',  'prev_item', 'product-company',  'synonym', 'verb_v_adjective_3', 'verb_v_adjective_5'] 

# 遍历所有子文件夹
for folder_name in subdirs:
    folder_path = os.path.join(root_directory, folder_name)
    
    if not os.path.isdir(folder_path):
        continue

    bernoullis_path = os.path.join(folder_path, "bernoullis.pt")
    if not os.path.exists(bernoullis_path):
        print(f"File not found: {bernoullis_path}")
        continue

    print(f"Loading: {bernoullis_path}")
    
    try:
        bernoullis = torch.load(bernoullis_path)  # shape [28, 16]
        mean_bernoullis = bernoullis.T  # [16, 28]
        flattened = mean_bernoullis.flatten().numpy()  # 448 维

        all_data.append(flattened)
        labels.append(folder_name)

    except Exception as e:
        print(f"Error processing {bernoullis_path}: {e}")
        continue

# 检查数据
if len(all_data) == 0:
    raise ValueError("No valid 'bernoullis.pt' files loaded.")

all_data = np.array(all_data)  # [num_tasks, 448]

# ============= 创建横坐标标签：0, 1, 2, ..., 447 =============
positions = np.arange(448)  # 0 到 447
xtick_labels = [str(i) for i in positions]  # 转为字符串

# ============= 绘图 =============
plt.figure(figsize=(448, 6))  # 合理宽度，避免过大

# 绘制数据曲线
for i in range(all_data.shape[0]):
    plt.plot(all_data[i], label=labels[i], alpha=0.7, linewidth=1.2, rasterized=False)

plt.title("Flattened Bernoullis Matrices with Index Labels (0 to 447)", fontsize=16)
plt.xlabel("Flattened Index (0 to 447)", fontsize=12)
plt.ylabel("Value", fontsize=12)

# 设置每个点的横坐标标签：0, 1, 2, ..., 447
plt.xticks(ticks=positions, labels=xtick_labels, rotation=90, fontsize=10)

# 图例：如果任务较多，显示在多个列中
if len(labels) <= 10:
    plt.legend(loc='upper right', fontsize=8)
else:
    plt.legend(loc='upper right', fontsize=6, ncol=2)

# 保存主图像
output_path = os.path.join(root_directory, "all_bernoullis_index_labeled_0_to_447.pdf")
plt.tight_layout()
plt.savefig(output_path, format='pdf', bbox_inches='tight')
plt.close()

print(f"Plot saved with index labels 0-447 to: {output_path}")


#################### 绘制图例单独生成一张图 ####################

# # 单独绘制图例部分
# plt.figure(figsize=(4, 2))  # 设置一个适合图例的小图形尺寸

# for label in labels:
#     plt.plot([], [], label=label)  

# # 生成空白的图形，只用于显示图例
# plt.legend(labels=labels, loc='upper left', fontsize=8, ncol=1)  # 设定合适的字体和列数

# # 不显示数据曲线，只显示图例
# plt.axis('off')  # 关闭坐标轴

# # 保存图例图为单独的 PDF 文件
# legend_output_path = os.path.join(root_directory, "legend_only.pdf")
# plt.savefig(legend_output_path, format='pdf', bbox_inches='tight')
# plt.close()

# print(f"Legend plot saved to: {legend_output_path}")

# plt.figure(figsize=(6, 2))  # 设置一个适合图例的小图形尺寸

# # 使用虚拟数据点来绘制图例
# for label in labels:
#     plt.plot([], [], label=label)  

# # 生成横向排列的图例，设置 ncol 为大于 1
# plt.legend(labels=labels, loc='upper left', fontsize=8, ncol=len(labels))  # ncol 为图例标签的数量

# # 不显示数据曲线，只显示图例
# plt.axis('off')  # 关闭坐标轴

# # 保存图例图为单独的 PDF 文件
# legend_output_path = os.path.join(root_directory, "legend_only_horizontal.pdf")
# plt.savefig(legend_output_path, format='pdf', bbox_inches='tight')
# plt.close()

# print(f"Legend plot saved to: {legend_output_path}")

# 创建一个小图用于图例

# 单独绘制图例部分
# 单独绘制图例部分
plt.figure(figsize=(6, 2))  # 设置一个适合图例的小图形尺寸（可以根据需要调整）

# 对 labels 列表按字母顺序排序
labels_sorted = sorted(labels)

# 使用虚拟数据点来绘制图例，按字母顺序
for label in labels_sorted:
    plt.plot([], [], label=label)  # 这里添加一个虚拟的空数据点来展示图例标签

# 生成横向排列的图例
plt.legend(loc='upper left', fontsize=8, ncol=5)  # 设置横向排列，ncol 设置列数

# 不显示数据曲线，只显示图例
plt.axis('off')  # 关闭坐标轴

# 保存图例图为单独的 PDF 文件
legend_output_path = os.path.join(root_directory, "legend_only_sorted_horizontal.pdf")
plt.savefig(legend_output_path, format='pdf', bbox_inches='tight')
plt.close()

print(f"Legend plot saved to: {legend_output_path}")
