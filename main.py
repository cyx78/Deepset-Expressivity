import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from trainer import *
import pickle

torch.manual_seed(0)

point_list=[]
acc_list=[]
loss_list=[]

for hidden_dimension in [16,24,32,64]:
    for sum_dimension in [1,2,4,8]:
        config = {
            'device': 'cuda',
            'seq_len': 6,
            'target_function': lambda x: sum_of_combinations(x,2,1),#lambda x: torch.mean(x ** torch.mean(x)).item(),
            'filepath': 'data.pth',
            'generate_data': False,
            'num_samples': 256,
            'batch_size': 24,
            'trainratio': 0.5,
            'num_epochs': 2000,
            'hidden_dimension': hidden_dimension,
            'sum_dimension': sum_dimension,
            'lr': 0.003,
            'acc_tolerance': 0.1,
            'manual_build_model': False
        }

        print(f'-----n={hidden_dimension},k={sum_dimension}-------')
        trainer = Trainer(config)
        trainer.train()
        # trainer.Deepset_prepare_graph()
        # trainer.plot_scatter_graph()
        # trainer.plot_target_function()
        # trainer.plot_model_output()
        # trainer.diag_single_sample()

        point_list.append((hidden_dimension,sum_dimension))
        acc_list.append(trainer.diag_max_acc())
        loss_list.append(trainer.diag_min_loss())


with open('data/param.pkl', 'wb') as file:
    pickle.dump(point_list, file)

with open('data/acc.pkl', 'wb') as file:
    pickle.dump(acc_list, file)

with open('data/loss.pkl', 'wb') as file:
    pickle.dump(loss_list, file)