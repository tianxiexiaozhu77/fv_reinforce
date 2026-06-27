import json
import random
import os
# 1. 加载原始国家-首都映射
input_file = '/storage/3090group2/hyl/function_vectors/dataset_files/abstractive/product-company.json' 
# output_dir = '/data/zjx/hyl/function_vectors/dataset_files/algebra/' 
# input_file = '/data/zjx/hyl/function_vectors/dataset_files/abstractive/product-company.json'  
# antonym.json english-french.json capitalize.json present-past.json singular-plural.json capitalize_first_letter.json product-company.json country-capital.json
output_dir = '/storage/3090group2/hyl/function_vectors/dataset_files/algebra/'
file_name_without_extension = os.path.splitext(os.path.basename(input_file))[0]
output_file = os.path.join(output_dir, f'{file_name_without_extension}_algebra_data_2500.json')

with open(input_file, 'r', encoding='utf-8') as f:
    country_data = json.load(f)

# 构建字典方便查询
country_to_capital = {item['input']: item['output'] for item in country_data}
countries = list(country_to_capital.keys())

# 2. 生成指定数量的全局不重复样本
def generate_unique_test_samples(num_samples=2500, countries_per_sample=4):
    if len(countries) < countries_per_sample:
        raise ValueError(f"国家数量不足 {countries_per_sample} 个")

    # 总理论组合数：C(197, 4) ≈ 2465万，2500个非常安全
    test_data = []
    used_combinations = set()  # 用于记录已使用的组合（frozenset 无序去重）

    attempts = 0
    max_attempts = num_samples * 10  # 防止死循环，但2500样本基本不会触发

    while len(test_data) < num_samples and attempts < max_attempts:
        selected = random.sample(countries, countries_per_sample)
        selected_tuple = tuple(sorted(selected))  # 排序后转 tuple，用于去重比较

        if selected_tuple not in used_combinations:
            used_combinations.add(selected_tuple)

            first_country = selected[0]
            last_country = selected[-1]

            sample = {
                "output_first_copy": first_country,
                "output_last_copy": last_country,
                "output_first_captial": country_to_capital[first_country],
                "output_last_captial": country_to_capital[last_country],
                "input": ", ".join(selected)
            }
            test_data.append(sample)

        attempts += 1

    if len(test_data) < num_samples:
        raise RuntimeError(f"无法生成 {num_samples} 个不重复样本，只生成了 {len(test_data)} 个")

    return test_data

# 3. 生成并保存
if __name__ == "__main__":
    print("正在生成 2500 个不重复测试样本...")
    test_samples = generate_unique_test_samples(num_samples=2500, countries_per_sample=4)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)

    print(f"成功生成 {len(test_samples)} 个不重复样本，已保存到 test_data_2500.json")

    # 预览前3个样本
    print("预览前 3 个样本：")
    for i, sample in enumerate(test_samples[:3]):
        print(f"样本 {i+1}:")
        for key, value in sample.items():
            print(f"  {key}: {value}")