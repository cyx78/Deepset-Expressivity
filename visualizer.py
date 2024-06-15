import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the data from pickle files
with open('data/param.pkl', 'rb') as file:
    point_list = pickle.load(file)

with open('data/acc.pkl', 'rb') as file:
    acc_list = pickle.load(file)

with open('data/loss.pkl', 'rb') as file:
    loss_list = pickle.load(file)

# Define the unique values of hidden_dimension and sum_dimension
hidden_dimensions = [16, 24, 32, 64, 96]
sum_dimensions = [1, 2, 4, 8]

# Initialize numpy arrays to hold the accuracy and loss values
train_acc_grid = np.zeros((len(hidden_dimensions), len(sum_dimensions)))
test_acc_grid = np.zeros((len(hidden_dimensions), len(sum_dimensions)))
train_loss_grid = np.zeros((len(hidden_dimensions), len(sum_dimensions)))
test_loss_grid = np.zeros((len(hidden_dimensions), len(sum_dimensions)))

# Fill the grids with the corresponding values from the lists
for idx, ((train_acc, test_acc), (train_loss, test_loss)) in enumerate(zip(acc_list, loss_list)):
    hidden_idx = idx // len(sum_dimensions)
    sum_idx = idx % len(sum_dimensions)
    
    train_acc_grid[hidden_idx, sum_idx] = train_acc
    test_acc_grid[hidden_idx, sum_idx] = test_acc
    train_loss_grid[hidden_idx, sum_idx] = train_loss
    test_loss_grid[hidden_idx, sum_idx] = test_loss

# Function to create a heatmap
def plot_heatmap(data, title, xlabel, ylabel, xticklabels, yticklabels):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(len(xticklabels)), xticklabels)
    plt.yticks(np.arange(len(yticklabels)), yticklabels)
    plt.savefig(f'figs/{title}.png')

# Plot the heatmaps
plot_heatmap(train_acc_grid, 'Train Accuracy', 'Sum Dimension', 'Hidden Dimension', sum_dimensions, hidden_dimensions)
plot_heatmap(test_acc_grid, 'Test Accuracy', 'Sum Dimension', 'Hidden Dimension', sum_dimensions, hidden_dimensions)
plot_heatmap(train_loss_grid, 'Train Loss', 'Sum Dimension', 'Hidden Dimension', sum_dimensions, hidden_dimensions)
plot_heatmap(test_loss_grid, 'Test Loss', 'Sum Dimension', 'Hidden Dimension', sum_dimensions, hidden_dimensions)
