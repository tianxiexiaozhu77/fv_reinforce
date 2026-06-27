# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# base_dir = '/opt/data/private/hyl/function_vectors/src/results/gptj_ori/antonym/antonym_indirect_effect.pt'
# directory_path = os.path.dirname(base_dir)
# indirect_effect = torch.load(base_dir)
# mean_indirect_effect = indirect_effect.mean(dim=0) # 25，28，16 -> 28,16
# mean_indirect_effect = mean_indirect_effect.T 

# plt.figure(figsize=(10, 6))
# sns.heatmap(mean_indirect_effect, annot=False, cmap="viridis")
# plt.title("Heatmap of Mean Indirect Effect (Shape [16, 28])")
# plt.savefig(f"{directory_path}/heatmap_mean_indirect_effect.png")
# #---------------------------------⬇️---------------------------------------
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import numpy as np
# from matplotlib.patches import Rectangle

# # Load the indirect effect data
# base_dir = '/opt/data/private/hyl/function_vectors/src/results/gptj_ori/singular-plural/singular-plural_indirect_effect.pt'
# directory_path = os.path.dirname(base_dir)
# indirect_effect = torch.load(base_dir)

# # Take the mean along the 0th dimension (dim=0)
# mean_indirect_effect = indirect_effect.mean(dim=0)  # Shape: [28, 16]

# # Transpose the tensor to [16, 28]
# mean_indirect_effect = mean_indirect_effect.T  # Shape [16, 28]

# # Set up the plot
# plt.figure(figsize=(10, 6))

# # Create the heatmap
# sns.heatmap(mean_indirect_effect, annot=False, cmap="viridis")

# # Highlight the top 10 values
# top_k = 10
# # Flatten the array and get the indices of the top k values
# top_k_indices = np.unravel_index(np.argsort(mean_indirect_effect.numpy().flatten())[-top_k:], mean_indirect_effect.shape)

# # Get the axis for plotting
# ax = plt.gca()

# # Define the size of each cell in the heatmap
# cell_width = 1
# cell_height = 1

# # Draw rectangles around the top k values
# for idx in range(top_k):
#     row, col = top_k_indices[0][idx], top_k_indices[1][idx]
#     rect = Rectangle((col, row), cell_width, cell_height, linewidth=2, edgecolor='red', facecolor='none')
#     ax.add_patch(rect)
#     print(f"Top {idx+1} - Position: (Row: {row}, Col: {col}), Value: {mean_indirect_effect[row, col]}")

# # Add title
# plt.title("Heatmap of Mean Indirect Effect (Shape [16, 28]) with Top 10 Values Highlighted")

# # Save the heatmap to the directory
# plt.savefig(f"{directory_path}/heatmap_mean_indirect_effect_with_highlight.png")
# #------------------------------------⬆️------------------------------------
# # Show the plot
# plt.show()

# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import numpy as np
# from matplotlib.patches import Rectangle

# # Load the indirect effect data
# # base_dir = '/opt/data/private/hyl/function_vectors/src/results/llama3-8b/ori/singular-plural/singular-plural_indirect_effect.pt'
# base_dir = '/opt/data/private/hyl/function_vectors/src/results/llama2-7b/test/antonym_10/bernoullis.pt'
# directory_path = os.path.dirname(base_dir)
# indirect_effect = torch.load(base_dir)

# # Take the mean along the 0th dimension (dim=0)
# mean_indirect_effect = indirect_effect.mean(dim=0)  # Shape: [28, 16]

# # Transpose the tensor to [16, 28]
# mean_indirect_effect = mean_indirect_effect.T  # Shape [16, 28]

# # Compute the average of the positive values
# positive_values = mean_indirect_effect[mean_indirect_effect > 0]
# positive_average = positive_values.mean() if positive_values.numel() > 0 else 0  # Avoid division by zero

# # Set up the plot
# plt.figure(figsize=(10, 6))

# # Create the heatmap
# sns.heatmap(mean_indirect_effect, annot=False, cmap="viridis")

# # Highlight values greater than the positive average
# # Get the indices of values greater than the positive average
# highlight_indices = np.where(mean_indirect_effect.numpy() > positive_average.item())

# # Get the axis for plotting
# ax = plt.gca()

# # Define the size of each cell in the heatmap
# cell_width = 1
# cell_height = 1

# # Draw rectangles around the values greater than the positive average
# for row, col in zip(*highlight_indices):
#     rect = Rectangle((col, row), cell_width, cell_height, linewidth=2, edgecolor='red', facecolor='none')
#     ax.add_patch(rect)
#     print(f"Position: (Row: {row}, Col: {col}), Value: {mean_indirect_effect[row, col]}")

# # Add title
# plt.title(f"Heatmap of Mean Indirect Effect with Values Greater than Positive Average ({positive_average:.2f}) Highlighted")

# # Save the heatmap to the directory
# plt.savefig(f"{directory_path}/heatmap_mean_indirect_effect_highlighted.png")

# # Show the plot
# plt.show()
# # Output the total number of highlighted values
# total_highlighted = len(highlight_indices[0])  # The first element in highlight_indices contains the row indices
# print(f"Total number of values greater than the positive average: {total_highlighted}")



# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import numpy as np
# from matplotlib.patches import Rectangle

# # Load the indirect effect data
# base_dir = '/opt/data/private/hyl/function_vectors/src/results/gptj_ori/antonym/antonym_indirect_effect.pt'
# directory_path = os.path.dirname(base_dir)
# indirect_effect = torch.load(base_dir)

# # Take the mean along the 0th dimension (dim=0)
# mean_indirect_effect = indirect_effect.mean(dim=0)  # Shape: [28, 16]

# # Transpose the tensor to [16, 28]
# mean_indirect_effect = mean_indirect_effect.T  # Shape [16, 28]

# # Set up the plot
# plt.figure(figsize=(12, 6))  # Increased width to make room for text

# # Create the heatmap
# sns.heatmap(mean_indirect_effect, annot=False, cmap="viridis")

# # Highlight the top 10 values
# top_k = 10
# # Flatten the array and get the indices of the top k values
# top_k_indices = np.unravel_index(np.argsort(mean_indirect_effect.numpy().flatten())[-top_k:], mean_indirect_effect.shape)

# # Get the axis for plotting
# ax = plt.gca()

# # Define the size of each cell in the heatmap
# cell_width = 1
# cell_height = 1

# # Draw rectangles around the top k values and display text on the right side
# for idx in range(top_k):
#     row, col = top_k_indices[0][idx], top_k_indices[1][idx]
#     rect = Rectangle((col, row), cell_width, cell_height, linewidth=2, edgecolor='red', facecolor='none')
#     ax.add_patch(rect)
    
#     # Display text with position and value
#     text = f"({row}, {col}):     {mean_indirect_effect[row, col]:.2f}"
#     ax.text(mean_indirect_effect.shape[1] + 5, row, text, ha='left', va='center', fontsize=10, color='black')

# # Add title
# plt.title("Heatmap of Mean Indirect Effect (Shape [16, 28]) with Top 10 Values Highlighted")

# # Save the heatmap to the directory
# plt.savefig(f"{directory_path}/heatmap_mean_indirect_effect_with_highlight_and_text.png")

# # Show the plot
# plt.show()


# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import numpy as np
# from matplotlib.patches import Rectangle

# # Load the indirect effect data
# base_dir = '/opt/data/private/hyl/function_vectors/src/results/gptj_reinforce/antonym/antonym_bernoullis.pt'
# directory_path = os.path.dirname(base_dir)
# indirect_effect = torch.load(base_dir)

# # Take the mean along the 0th dimension (dim=0)
# mean_indirect_effect = indirect_effect.mean(dim=0)  # Shape: [28, 16]

# # Transpose the tensor to [16, 28]
# mean_indirect_effect = mean_indirect_effect.T  # Shape [16, 28]

# # Set up the plot with more space on the right
# plt.figure(figsize=(14, 6))  # Increased width to make more room for text

# # Create the heatmap
# sns.heatmap(mean_indirect_effect, annot=False, cmap="viridis", cbar_kws={"shrink": 0.8})

# # Highlight the top 10 values
# top_k = 10
# # Flatten the array and get the indices of the top k values
# top_k_indices = np.unravel_index(np.argsort(mean_indirect_effect.numpy().flatten())[-top_k:], mean_indirect_effect.shape)

# # Get the axis for plotting
# ax = plt.gca()

# # Define the size of each cell in the heatmap
# cell_width = 1
# cell_height = 1

# # Initialize a dictionary to keep track of the vertical offset for each row
# row_offsets = {row: 0 for row in range(mean_indirect_effect.shape[0])}

# # Draw rectangles around the top k values and display text further to the right
# for idx in range(top_k):
#     row, col = top_k_indices[0][idx], top_k_indices[1][idx]
#     rect = Rectangle((col, row), cell_width, cell_height, linewidth=2, edgecolor='red', facecolor='none')
#     ax.add_patch(rect)
    
#     # Stagger the vertical position of the text to avoid overlap
#     vertical_offset = row_offsets[row] * 0.5  # Adjust this value to control spacing
#     row_offsets[row] += 1  # Increment the offset for the row

#     # Display text with position and value
#     text = f"({row}, {col}): {mean_indirect_effect[row, col]:.2f}"
#     ax.text(mean_indirect_effect.shape[1] + 5, row + vertical_offset, text, ha='left', va='center', fontsize=10, color='black')

# # Add title
# plt.title("Heatmap of Mean Indirect Effect (Shape [16, 28]) with Top 10 Values Highlighted")

# # Adjust layout to add more space on the right
# plt.subplots_adjust(right=0.85)  # Adjust the plot to add space on the right side

# # Save the heatmap to the directory
# plt.savefig(f"{directory_path}/heatmap_mean_indirect_effect_with_highlight_and_text.png")

# # Show the plot
# plt.show()



# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import numpy as np
# from matplotlib.patches import Rectangle

# # Load the indirect effect data
# # base_dir = '/opt/data/private/hyl/function_vectors/src/results/llama3-8b/ori/singular-plural/singular-plural_indirect_effect.pt'
# base_dir = '/opt/data/private/hyl/function_vectors/src/results/llama2-7b/test/antonym_10/bernoullis.pt'
# directory_path = os.path.dirname(base_dir)
# indirect_effect = torch.load(base_dir)
# mean_indirect_effect = indirect_effect
# # Take the mean along the 0th dimension (dim=0)
# # mean_indirect_effect = indirect_effect.mean(dim=0)  # Shape: [28, 16]

# # # Transpose the tensor to [16, 28]
# # mean_indirect_effect = mean_indirect_effect.T  # Shape [16, 28]

# # Compute the average of the positive values
# positive_values = mean_indirect_effect[mean_indirect_effect > 0]
# positive_average = positive_values.mean() if positive_values.numel() > 0 else 0  # Avoid division by zero

# # Set up the plot
# plt.figure(figsize=(10, 10))

# # Create the heatmap
# sns.heatmap(mean_indirect_effect, annot=False, cmap="viridis")

# # Highlight values greater than the positive average
# # Get the indices of values greater than the positive average
# highlight_indices = np.where(mean_indirect_effect.numpy() > positive_average.item())

# # Get the axis for plotting
# ax = plt.gca()

# # Define the size of each cell in the heatmap
# cell_width = 1
# cell_height = 1

# # Draw rectangles around the values greater than the positive average
# for row, col in zip(*highlight_indices):
#     rect = Rectangle((col, row), cell_width, cell_height, linewidth=2, edgecolor='red', facecolor='none')
#     ax.add_patch(rect)
#     print(f"Position: (Row: {row}, Col: {col}), Value: {mean_indirect_effect[row, col]}")

# # Add title
# plt.title(f"Heatmap of Mean Indirect Effect with Values Greater than Positive Average ({positive_average:.2f}) Highlighted")

# # Save the heatmap to the directory
# plt.savefig(f"{directory_path}/heatmap_mean_indirect_effect_highlighted.png")

# # Show the plot
# plt.show()
# # Output the total number of highlighted values
# total_highlighted = len(highlight_indices[0])  # The first element in highlight_indices contains the row indices
# print(f"Total number of values greater than the positive average: {total_highlighted}")

