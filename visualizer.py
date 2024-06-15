import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('data/param.pkl', 'rb') as file:
    point_list = pickle.load(file)

with open('data/acc.pkl', 'rb') as file:
    acc_list = pickle.load(file)

with open('data/loss.pkl', 'rb') as file:
    loss_list = pickle.load(file)

# Extract unique values of hidden_dimension and sum_dimension
hidden_dimensions = sorted(list(set([pt[0] for pt in point_list])))
sum_dimensions = sorted(list(set([pt[1] for pt in point_list])))

# Initialize numpy arrays to hold the accuracy and loss values
train_acc_grid = np.zeros((len(hidden_dimensions), len(sum_dimensions)))
test_acc_grid = np.zeros((len(hidden_dimensions), len(sum_dimensions)))
train_loss_grid = np.zeros((len(hidden_dimensions), len(sum_dimensions)))
test_loss_grid = np.zeros((len(hidden_dimensions), len(sum_dimensions)))

for idx, ((hidden_dim, sum_dim), (train_acc, test_acc), (train_loss, test_loss)) in enumerate(zip(point_list, acc_list, loss_list)):
    hidden_idx = hidden_dimensions.index(hidden_dim)
    sum_idx = sum_dimensions.index(sum_dim)
    
    train_acc_grid[hidden_idx, sum_idx] = train_acc
    test_acc_grid[hidden_idx, sum_idx] = test_acc
    train_loss_grid[hidden_idx, sum_idx] = train_loss
    test_loss_grid[hidden_idx, sum_idx] = test_loss

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

plot_heatmap(train_acc_grid, 'Train Accuracy', 'Sum Dimension', 'Hidden Dimension', sum_dimensions, hidden_dimensions)
plot_heatmap(test_acc_grid, 'Test Accuracy', 'Sum Dimension', 'Hidden Dimension', sum_dimensions, hidden_dimensions)
plot_heatmap(train_loss_grid, 'Train Loss', 'Sum Dimension', 'Hidden Dimension', sum_dimensions, hidden_dimensions)
plot_heatmap(test_loss_grid, 'Test Loss', 'Sum Dimension', 'Hidden Dimension', sum_dimensions, hidden_dimensions)
