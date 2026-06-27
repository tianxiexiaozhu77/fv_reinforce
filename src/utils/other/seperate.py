import json
import os
tasks = ['capitalize_first_letter', 'country-capital', 'english-french', 'present-past', 'product-company', 'singular-plural']
def seperate(task):
    input_file = f"/storage/3090group2/hyl/function_vectors/dataset_files/algebra/{task}_algebra_data_2500.json"
    output_dir = f'/storage/3090group2/hyl/function_vectors/dataset_files/algebra/{task}'
    output_file = {
        'AC': f'{task}_first_copy.json',
        'BC': f'{task}_last_copy.json',
        'AD': f'first_{task}.json',
        'BD': f'last_{task}.json'
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {
        'AC': [],
        'BC': [],
        'AD': [],
        'BD': []
    }

    for item in data:
        input_val = item['input']

        results['AC'].append({
            'input': input_val,
            'output': item['output_first_copy']
        })

        results['BC'].append({
            'input': input_val,
            'output': item['output_last_copy']
        })

        results['AD'].append({
            'input': input_val,
            'output': item['output_first_captial']
        })

        results['BD'].append({
            'input': input_val,
            'output': item['output_last_captial']
        })

    for key, filename in output_file.items():
        with open(f'{output_dir}/{filename}', 'w', encoding='utf-8') as f:
            json.dump(results[key], f, indent=2, ensure_ascii=False)

for task in tasks:
    seperate(task)