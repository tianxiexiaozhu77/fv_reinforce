import os
import pandas as pd
import matplotlib.pyplot as plt
import math


## 实现把多个子图放到一张大图里（只针对6个主要数据集）

# Function to dynamically generate output directory based on the input file path
def get_output_directory_from_filename(input_file):
    base_dir = os.path.dirname(input_file)
    if 'zs' in input_file:
        return os.path.join(base_dir, 'figs', 'zs_rein')
    else:
        return os.path.join(base_dir, 'figs', 'fs_shuffled_rein')

# def plot_and_save(data_1, data_2, data_3, output_dir,prefix):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     num_datasets = len(data_1)
    
#     # 动态设置图形宽度：每张子图宽 6 英寸
#     fig_width = 4 * num_datasets
#     fig_height = 4  # 高度固定或根据需要调整

#     fig, axes = plt.subplots(
#         nrows=1, ncols=num_datasets,
#         figsize=(fig_width, fig_height),
#         sharex=False, sharey=True  # 可选择是否共享 x 轴
#     )

#     # 如果只有一个数据集，axes 不是数组，要转成列表
#     if num_datasets == 1:
#         axes = [axes]

#     for i, (index, row) in enumerate(data_1.iterrows()):
#         dataset_name = row['name']
#         ax = axes[i]



#         # Plot ours (data_1)
#         ax.plot(
#             data_1.columns[1:], row[1:],
#             label='ours', color='green', linewidth=3
#         )

#         # Match in data_2
#         matching_row2 = data_2[data_2['name'] == dataset_name]
#         if not matching_row2.empty:
#             ax.plot(
#                 data_2.columns[1:], matching_row2.iloc[0, 1:],
#                 label='fv', color='red', linewidth=3
#             )

#         # Match in data_3
#         matching_row3 = data_3[data_3['name'] == dataset_name]
#         if not matching_row3.empty:
#             ax.plot(
#                 data_3.columns[1:], matching_row3.iloc[0, 1:],
#                 label='ori', color='orange', linewidth=3, linestyle='--'
#             )

#         if 'avg' in dataset_name and prefix=='fs_shuffled':
#             dataset_name = 'avg_shf'
#         elif 'avg' in dataset_name and prefix=='zs':
#             dataset_name = 'avg_zs'

#         ax.set_title(dataset_name, fontsize=22)
        
#         # 设置 x 轴刻度：每 5 层一个标签
#         x_cols = data_1.columns[1:]
#         tick_indices = range(0, len(x_cols), 10)
#         ax.set_xticks(tick_indices)
#         ax.set_xticklabels([int(x_cols[i].split('_')[1]) for i in tick_indices], fontsize=16,rotation=0)

#         ax.set_yticks([0, 0.5, 1])
#         ax.set_yticklabels(
#             ['0.0', '0.5', '1.0'],
#             fontsize=16        # 增大 y 轴字号
#         )
#         # ax.grid(True, linestyle='--', alpha=0.3)

#     # 总标题
#     # fig.suptitle("Comparison across datasets", fontsize=16)

#     # 统一图例（只取第一个子图的图例内容）
#     handles, labels = axes[0].get_legend_handles_labels()
#     if prefix == 'fs_shuffled':
#     # fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=20, frameon=False)
#         fig.legend(
#         handles, labels,
#         loc='lower center',  # 实际位置由 bbox 控制，这个可随便设
#         bbox_to_anchor=(0.5, 0.85),  # (x, y): 把图例锚点放在图像上方
#         ncol=3,
#         fontsize=20,
#         frameon=False
#     )

#     # 调整布局，防止重叠
#     plt.tight_layout(rect=[0, 0, 1, 0.9])  # 留出上方空间给 suptitle 和 legend

#     # 保存图像
#     output_path = os.path.join(output_dir, f'{prefix}_all_datasets_combined_subplots_horizontal.pdf')
#     plt.savefig(output_path, bbox_inches='tight', dpi=150)
#     plt.close()

#     print(f"Saved combined horizontal subplot figure to: {output_path}")


def plot_and_save(data_1, data_2, data_3, output_dir, prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_datasets = len(data_1)
    
    # 每行放 7 张子图
    ncols = 7
    nrows = math.ceil(num_datasets / ncols)
    
    # 图形宽高：每张子图宽 4 英寸，高 4 英寸
    fig_width = ncols * 4
    fig_height = nrows * 4

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(fig_width, fig_height),
        sharex=False, sharey=True
    )

    # axes 转成二维数组
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    # 展平 axes 便于迭代
    axes_flat = [ax for row in axes for ax in row]

    for i, (index, row) in enumerate(data_1.iterrows()):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        dataset_name = row['name']

        # Plot ours (data_1)
        ax.plot(
            data_1.columns[1:], row[1:],
            label='ours', color='green', linewidth=3
        )

        # Match in data_2
        matching_row2 = data_2[data_2['name'] == dataset_name]
        if not matching_row2.empty:
            ax.plot(
                data_2.columns[1:], matching_row2.iloc[0, 1:],
                label='fv', color='red', linewidth=3
            )

        # Match in data_3
        matching_row3 = data_3[data_3['name'] == dataset_name]
        if not matching_row3.empty:
            ax.plot(
                data_3.columns[1:], matching_row3.iloc[0, 1:],
                label='ori', color='orange', linewidth=3, linestyle='--'
            )

        if 'avg' in dataset_name and prefix=='fs_shuffled':
            dataset_name = 'avg_shf'
        elif 'avg' in dataset_name and prefix=='zs':
            dataset_name = 'avg_zs'

        ax.set_title(dataset_name, fontsize=22)
        
        # x 轴刻度
        x_cols = data_1.columns[1:]
        tick_indices = range(0, len(x_cols), 10)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([int(x_cols[i].split('_')[1]) for i in tick_indices], fontsize=16, rotation=0)

        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=16)
        # ax.grid(True, linestyle='--', alpha=0.3)

    # 删除多余的空白子图
    for j in range(i+1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # 统一图例
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.6, 0.05),
        ncol=3,
        fontsize=30,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    # 保存
    output_path = os.path.join(output_dir, f'{prefix}_all_datasets_combined_subplots.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved combined subplot figure to: {output_path}")


# GPT-J-6b shuffled fs
# input_file_1 = "/storage/3090group2/hyl/function_vectors/src/results/gptj/gptj_reinforce/gptj_reinforce_fs_shuffled_intervention_topk_rank1_values.xlsx"
# input_file_2 = "/storage/3090group2/hyl/function_vectors/src/results/gptj/gptj_ori/gptj_ori_fs_shuffled_intervention_topk_rank1_values.xlsx"
# input_file_3 = "/storage/3090group2/hyl/function_vectors/src/results/gptj/gptj_reinforce/gptj_reinforce_fs_shuffled_clean_topk_rank1_values.xlsx"
# GPT-J-6b zs
input_file_1 = "/storage/3090group2/hyl/function_vectors/src/results/gptj/gptj_reinforce/gptj_reinforce_zs_intervention_topk_rank1_values.xlsx"
input_file_2 = "/storage/3090group2/hyl/function_vectors/src/results/gptj/gptj_ori/gptj_ori_zs_intervention_topk_rank1_values.xlsx"
input_file_3 = "/storage/3090group2/hyl/function_vectors/src/results/gptj/gptj_reinforce/gptj_reinforce_zs_clean_topk_rank1_values.xlsx"






data_1 = pd.read_excel(input_file_1)
data_2 = pd.read_excel(input_file_2)
data_3 = pd.read_excel(input_file_3)

output_dir_1 = get_output_directory_from_filename(input_file_1)
print(f"Output Directory for file 1: {output_dir_1}")

if "fs_shuffled" in input_file_1:
    prefix="fs_shuffled"
else:
    prefix="zs"

plot_and_save(data_1, data_2, data_3, output_dir_1,prefix)
