import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to dynamically generate output directory based on the input file path
def get_output_directory_from_filename(input_file):
    # Extract the directory part from the input file path
    base_dir = os.path.dirname(input_file)
    
    # Check if 'zs' is in the file path to determine the subdirectory
    if 'zs' in input_file:
        return os.path.join(base_dir, 'figs', 'zs_rein')  # If 'zs' is in the file name, save in 'figs/zs'
    else:
        return os.path.join(base_dir, 'figs', 'fs_shuffled_rein')  # Otherwise, save in 'figs/fs'

# Function to process the data, generate line plots, and save them
def plot_and_save(data_1, data_2, data_3, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each row (representing datasets)
    for index, row in data_1.iterrows():
        dataset_name = row['name']
        
        # Create a new figure for each dataset
        plt.figure()

        # Plot data_1 in green
        plt.plot(data_1.columns[1:], row[1:], label=f'{dataset_name} (BD)', color='green', linewidth=4)  # plot with green color
        
        # Find matching dataset in data_2
        matching_row2 = data_2[data_2['name'] == dataset_name]
        
        if not matching_row2.empty:
            # Plot data_2 in orange for the same dataset name
            plt.plot(data_2.columns[1:], matching_row2.iloc[0, 1:], label=f'{dataset_name} (BD✳️)', color='red', linewidth=4)  # plot with orange color

        matching_row3 = data_3[data_3['name'] == dataset_name]

        if not matching_row3.empty:
            # Plot data_2 in orange for the same dataset name
            plt.plot(data_3.columns[1:], matching_row3.iloc[0, 1:], label=f'{dataset_name} (few-shot)', color='orange', linewidth=4,linestyle='--')  # plot with orange color

        # plt.axvline(x=10, color='red', linestyle='--', label='Vertical Line at 11th Position')

        # Customize the plot
        plt.xticks(ticks=range(0, len(data_1.columns[1:]), 5), labels=[int(col.split('_')[1]) for col in data_1.columns[1:][::5]])
        plt.yticks([0, 0.5, 1])
        # Add legend
        plt.legend()

        # Save the figure
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_combined_plot.pdf'), bbox_inches='tight')
        plt.close()


# zs
input_file_1 = '/storage/3090group2/hyl/function_vectors/src/results/ablation/vector_algebra/gptj_average_3/gptj_average_3_BD_zs_results_layer_sweep.xlsx'
input_file_2 = '/storage/3090group2/hyl/function_vectors/src/results/ablation/vector_algebra/gptj_average_3/gptj_average_3_BD_star_synthetic_zs_results_layer_sweep.xlsx'
input_file_3 = '/storage/3090group2/hyl/function_vectors/src/results/ablation/vector_algebra/gptj_average_3/gptj_average_3_fs_results_layer_sweep.xlsx'
# Read the Excel files to load the data



data_1 = pd.read_excel(input_file_1)
data_2 = pd.read_excel(input_file_2)
data_3 = pd.read_excel(input_file_3)

#-----------------
rank_values = data_3['rank1_0'].values  # 转为 numpy 数组便于操作
new_columns = pd.DataFrame({f'rank1_{i}': rank_values for i in range(28)})
result = pd.concat([data_3[['name']].reset_index(drop=True), new_columns], axis=1)
data_3 = result 
#-----------------

# Dynamically determine output directory
output_dir_1 = get_output_directory_from_filename(input_file_1)

# Print the generated directories to verify
print(f"Output Directory for file 1: {output_dir_1}")

# Call the function to plot and save the combined plot for both datasets
plot_and_save(data_1, data_2, data_3,output_dir_1)

# "Plots generated successfully."
