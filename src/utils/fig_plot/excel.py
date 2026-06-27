import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to dynamically generate output directory based on the input file path
def get_output_directory_from_filename(input_file):
    # Extract the directory part from the input file path
    base_dir = os.path.dirname(input_file)
    
    # Check if 'ori' is in the file path to determine the subdirectory
    if 'zs' in input_file:
        return os.path.join(base_dir, 'figs', 'zs')  # If 'ori' is in the file name, save in 'figs/ori'
    else:
        return os.path.join(base_dir, 'figs', 'fs')  # Otherwise, save in 'fig/rein'

# Function to process the data, generate line plots, and save them
def plot_and_save(data, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each row (representing datasets)
    for index, row in data.iterrows():
        dataset_name = row['name']
        # if dataset_name == "avg":

        plt.figure()
        plt.plot(data.columns[1:], row[1:], label=dataset_name, color='c')  # plot the data
        # plt.title(dataset_name)  # set the title as dataset name
        # plt.xlabel('Rank')
        # plt.ylabel('Accuracy')
        # plt.grid(True)
        # plt.xticks(ticks=range(0, len(data.columns[1:]), 5), labels=data.columns[1:][::5])
        plt.xticks(ticks=range(0, len(data.columns[1:]), 5), labels=[int(col.split('_')[1]) for col in data.columns[1:][::5]])
        # Save the figure
        plt.yticks([0, 0.5, 1])
        plt.savefig(os.path.join(output_dir, f'{dataset_name}.png'))
        plt.close()

# Test file paths
input_file_1 = '/opt/data/private/zjx/ICL/hyl/function_vectors/src/results/gptj_ori/gptj_ori_intervention_topk_rank1_values.xlsx'
input_file_2 = '/opt/data/private/zjx/ICL/hyl/function_vectors/src/results/gptj_ori/gptj_ori_zs_intervention_topk_rank1_values.xlsx'

# Read the Excel files to load the data
data_1 = pd.read_excel(input_file_1)
data_2 = pd.read_excel(input_file_2)

# Dynamically determine output directories
output_dir_1 = get_output_directory_from_filename(input_file_1)
output_dir_2 = get_output_directory_from_filename(input_file_2)

# Print the generated directories to verify
print(f"Output Directory for file 1: {output_dir_1}")
print(f"Output Directory for file 2: {output_dir_2}")

# Call the function to plot and save for both datasets
plot_and_save(data_1, output_dir_1)
plot_and_save(data_2, output_dir_2)

"Plots generated successfully."

# colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'c']
