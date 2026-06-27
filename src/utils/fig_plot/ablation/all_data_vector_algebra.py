import os
import json
import pandas as pd

base_dir = "/storage/3090group2/hyl/function_vectors/src/results/ablation/vector_algebra/gptj_average_3"
# base_dir = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_ori"
# base_dir = "/opt/data/private/hyl/function_vectors/src/results/gptj/gptj_reinforce"
last_dir = os.path.basename(os.path.normpath(base_dir))


subdirs = ['last_antonym', 'last_capitalize', 'last_capitalize_first_letter','last_country-capital', 'last_english-french', 'last_present-past', 'last_product-company','last_singular-plural']
# subdirs = ['capitalize_first_letter', 'country-currency', 'english-german', 'english-spanish', 'landmark-country', 'lowercase_first_letter', 'national_parks', 'park-country', 'person-sport', 'product-company', 'sentiment', 'synonym', 'adjective_v_verb_3', 'adjective_v_verb_5', 'animal_v_object_3', 'animal_v_object_5', 'choose_first_of_3', 'choose_first_of_5', 'choose_middle_of_3', 'choose_last_of_3', 'choose_last_of_5', 'color_v_animal_3', 'color_v_animal_5', 'concept_v_object_3', 'concept_v_object_5', 'conll2003_location', 'conll2003_organization', 'fruit_v_animal_3', 'fruit_v_animal_5', 'object_v_concept_3', 'object_v_concept_5', 'verb_v_adjective_3', 'verb_v_adjective_5']

subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
# 存储结果
all_results = []
for subdir in subdirs:
    file_path = os.path.join(base_dir, subdir, 'BD_synthetic_zs_results_layer_sweep.json')
    # file_path = os.path.join(base_dir, subdir, 'zs_results_layer_sweep.json')

    if not os.path.isfile(file_path):
        print(f"Warning: File not found, skipping {file_path}")
        continue  # 跳过当前 subdir

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    row = [subdir]  # 第一列是文件夹名

    # 遍历每个 key（"0", "1", ..., "28"）
    for key in sorted(data.keys(), key=lambda x: int(x)):
        topk = data[key].get('intervention_topk', [])
        # topk = data[key].get('clean_topk', [])
        value_for_rank_1 = None
        for item in topk:
            if item[0] == 1:
                value_for_rank_1 = item[1]
                break
        row.append(value_for_rank_1)

    all_results.append(row)


columns = ['name'] + [f'rank1_{i}' for i in range(len(all_results[0]) - 1)]
df = pd.DataFrame(all_results, columns=columns)

# 计算每一列的平均值（排除第一列 name），乘以 100 并保留两位小数
mean_values = df.iloc[:, 1:].mean().apply(lambda x: round(x * 1, 4))
mean_row = ['avg'] + mean_values.tolist()

# 将平均值作为新行追加
df.loc[len(df)] = mean_row

# output_file = os.path.join(base_dir, f"{last_dir}_zs_intervention_topk_rank1_values.xlsx")
output_file = os.path.join(base_dir, f"{last_dir}_BD_star_synthetic_zs_results_layer_sweep.xlsx")
# output_file = os.path.join(base_dir, f"{last_dir}_fs_shuffled_clean_topk_rank1_values_new.xlsx")
df.to_excel(output_file, index=False)

print(f"结果已保存到 {output_file}")



# 存储结果
all_results = []
for subdir in subdirs:
    # file_path = os.path.join(base_dir, subdir, 'fs_shuffled_results_layer_sweep.json')
    file_path = os.path.join(base_dir, subdir, f'{subdir}_zs_results_layer_sweep.json')

    if not os.path.isfile(file_path):
        print(f"Warning: File not found, skipping {file_path}")
        continue  # 跳过当前 subdir

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    row = [subdir]  # 第一列是文件夹名

    # 遍历每个 key（"0", "1", ..., "28"）
    for key in sorted(data.keys(), key=lambda x: int(x)):
        topk = data[key].get('intervention_topk', [])
        # topk = data[key].get('clean_topk', [])
        value_for_rank_1 = None
        for item in topk:
            if item[0] == 1:
                value_for_rank_1 = item[1]
                break
        row.append(value_for_rank_1)

    all_results.append(row)


columns = ['name'] + [f'rank1_{i}' for i in range(len(all_results[0]) - 1)]
df = pd.DataFrame(all_results, columns=columns)

# 计算每一列的平均值（排除第一列 name），乘以 100 并保留两位小数
mean_values = df.iloc[:, 1:].mean().apply(lambda x: round(x * 1, 4))
mean_row = ['avg'] + mean_values.tolist()

# 将平均值作为新行追加
df.loc[len(df)] = mean_row

output_file = os.path.join(base_dir, f"{last_dir}_BD_zs_results_layer_sweep.xlsx")
# output_file = os.path.join(base_dir, f"{last_dir}_zs_clean_topk_rank1_values_all_new.xlsx")
# output_file = os.path.join(base_dir, f"{last_dir}_fs_shuffled_intervention_topk_rank1_values.xlsx")
df.to_excel(output_file, index=False)

print(f"结果已保存到 {output_file}")



# 存储结果
all_results = []
for subdir in subdirs:
    # file_path = os.path.join(base_dir, subdir, 'fs_shuffled_results_layer_sweep.json')
    file_path = os.path.join(base_dir, subdir, f'{subdir}_fs_results_layer_sweep.json')

    if not os.path.isfile(file_path):
        print(f"Warning: File not found, skipping {file_path}")
        continue  # 跳过当前 subdir

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    row = [subdir]  # 第一列是文件夹名

    # 遍历每个 key（"0", "1", ..., "28"）
    # for key in sorted(data.keys(), key=lambda x: int(x)):
        # topk = data[key].get('intervention_topk', [])
    topk = data.get('clean_topk', [])
    value_for_rank_1 = None
    for item in topk:
        if item[0] == 1:
            value_for_rank_1 = item[1]
            break
    row.append(value_for_rank_1)

    all_results.append(row)


columns = ['name'] + [f'rank1_{i}' for i in range(len(all_results[0]) - 1)]
df = pd.DataFrame(all_results, columns=columns)

# 计算每一列的平均值（排除第一列 name），乘以 100 并保留两位小数
mean_values = df.iloc[:, 1:].mean().apply(lambda x: round(x * 1, 4))
mean_row = ['avg'] + mean_values.tolist()

# 将平均值作为新行追加
df.loc[len(df)] = mean_row

output_file = os.path.join(base_dir, f"{last_dir}_fs_results_layer_sweep.xlsx")
# output_file = os.path.join(base_dir, f"{last_dir}_zs_clean_topk_rank1_values_all_new.xlsx")
# output_file = os.path.join(base_dir, f"{last_dir}_fs_shuffled_intervention_topk_rank1_values.xlsx")
df.to_excel(output_file, index=False)

print(f"结果已保存到 {output_file}")


# all_results = []
# for subdir in subdirs:
#     file_path = os.path.join(base_dir, subdir, 'fs_shuffled_results_layer_sweep.json')
#     # file_path = os.path.join(base_dir, subdir, 'zs_results_layer_sweep.json')

#     if not os.path.isfile(file_path):
#         print(f"Warning: File not found, skipping {file_path}")
#         continue  # 跳过当前 subdir

#     with open(file_path, 'r') as f:
#         data = json.load(f)
    
#     row = [subdir]  # 第一列是文件夹名

#     # 遍历每个 key（"0", "1", ..., "28"）
#     for key in sorted(data.keys(), key=lambda x: int(x)):
#         # topk = data[key].get('intervention_topk', [])
#         topk = data[key].get('clean_topk', [])
#         value_for_rank_1 = None
#         for item in topk:
#             if item[0] == 1:
#                 value_for_rank_1 = item[1]
#                 break
#         row.append(value_for_rank_1)

#     all_results.append(row)


# columns = ['name'] + [f'rank1_{i}' for i in range(len(all_results[0]) - 1)]
# df = pd.DataFrame(all_results, columns=columns)

# # 计算每一列的平均值（排除第一列 name），乘以 100 并保留两位小数
# mean_values = df.iloc[:, 1:].mean().apply(lambda x: round(x * 1, 4))
# mean_row = ['avg'] + mean_values.tolist()

# # 将平均值作为新行追加
# df.loc[len(df)] = mean_row

# # output_file = os.path.join(base_dir, f"{last_dir}_zs_intervention_topk_rank1_values.xlsx")
# # output_file = os.path.join(base_dir, f"{last_dir}_fs_shuffled_intervention_topk_rank1_values_all_new.xlsx")
# output_file = os.path.join(base_dir, f"{last_dir}_fs_shuffled_clean_topk_rank1_values_new.xlsx")
# df.to_excel(output_file, index=False)

# print(f"结果已保存到 {output_file}")



# # 存储结果
# all_results = []
# for subdir in subdirs:
#     # file_path = os.path.join(base_dir, subdir, 'fs_shuffled_results_layer_sweep.json')
#     file_path = os.path.join(base_dir, subdir, 'zs_results_layer_sweep.json')

#     if not os.path.isfile(file_path):
#         print(f"Warning: File not found, skipping {file_path}")
#         continue  # 跳过当前 subdir

#     with open(file_path, 'r') as f:
#         data = json.load(f)
    
#     row = [subdir]  # 第一列是文件夹名

#     # 遍历每个 key（"0", "1", ..., "28"）
#     for key in sorted(data.keys(), key=lambda x: int(x)):
#         # topk = data[key].get('intervention_topk', [])
#         topk = data[key].get('clean_topk', [])
#         value_for_rank_1 = None
#         for item in topk:
#             if item[0] == 1:
#                 value_for_rank_1 = item[1]
#                 break
#         row.append(value_for_rank_1)

#     all_results.append(row)


# columns = ['name'] + [f'rank1_{i}' for i in range(len(all_results[0]) - 1)]
# df = pd.DataFrame(all_results, columns=columns)

# # 计算每一列的平均值（排除第一列 name），乘以 100 并保留两位小数
# mean_values = df.iloc[:, 1:].mean().apply(lambda x: round(x * 1, 4))
# mean_row = ['avg'] + mean_values.tolist()

# # 将平均值作为新行追加
# df.loc[len(df)] = mean_row

# # output_file = os.path.join(base_dir, f"{last_dir}_zs_intervention_topk_rank1_values_all_new.xlsx")
# output_file = os.path.join(base_dir, f"{last_dir}_zs_clean_topk_rank1_values_all_new.xlsx")
# # output_file = os.path.join(base_dir, f"{last_dir}_fs_shuffled_intervention_topk_rank1_values.xlsx")
# df.to_excel(output_file, index=False)

# print(f"结果已保存到 {output_file}")
