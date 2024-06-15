import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt

#Model input shape: (seq_len, batch_size)
#Model output shape: (batch_size,)
#Target Function: tensor to scaler
#Create folder "figs/" before running visualize

class Trainer:
    def __init__(self, config):
        self.device = config['device']
        self.seq_len = config['seq_len']
        self.target_function = config['target_function']
        self.filepath = config['filepath']
        self.generate_data = config['generate_data']
        self.num_samples = config['num_samples']
        self.batch_size = config['batch_size']
        self.trainratio = config['trainratio']
        self.num_epochs = config['num_epochs']
        self.hidden_dimension = config['hidden_dimension']
        self.sum_dimension = config['sum_dimension']
        self.lr = config['lr']
        self.manual_build_model = config['manual_build_model']

        self.info = f'n={self.hidden_dimension}_k={self.sum_dimension}_lr={self.lr}'

        self.train_loss_list = []
        self.test_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        self._prepare_data()
        if not self.manual_build_model:
            self._build_model()

    def _prepare_data(self):
        if self.generate_data:
            Gen_Rand_Dataset(self.seq_len, self.target_function, num_samples=self.num_samples, filepath=self.filepath)
        self.trainloader, self.testloader = Load_Dataset(self.filepath, self.batch_size, self.trainratio)

    def _build_model(self):
        self.model = DeepSet(hidden_dim_n=self.hidden_dimension, hidden_dim_k=self.sum_dimension)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            train_loss, train_acc = self._train_one_epoch()
            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)

            test_loss, test_acc = self._evaluate()
            self.test_loss_list.append(test_loss)
            self.test_acc_list.append(test_acc)

        self._plot_results()

    def _train_one_epoch(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for src, target in iter(self.trainloader):
            src, target = src.to(self.device), target.to(self.device)
            output = self.model(src.t())
            self.optimizer.zero_grad()
            loss = torch.mean((output - target) ** 2)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            correct += torch.sum((output - target) ** 2 < 0.01).item()
            total += target.size(0)

        avg_train_loss = train_loss / len(self.trainloader)
        train_acc = correct / total
        return avg_train_loss, train_acc

    def _evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for src, target in iter(self.testloader):
                src, target = src.to(self.device), target.to(self.device)
                output = self.model(src.t())
                loss = torch.mean((output - target) ** 2)
                test_loss += loss.item()
                correct += torch.sum((output - target) ** 2 < 0.01).item()
                total += target.size(0)

        avg_test_loss = test_loss / len(self.testloader)
        test_acc = correct / total
        return avg_test_loss, test_acc

    def _plot_results(self):
        #Training curves
        plt.figure()
        plt.plot(self.train_loss_list, label='train_loss')
        plt.plot(self.test_loss_list, label='test_loss')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.title('Loss')
        plt.savefig(f'figs/{self.info}_loss.png')

        plt.figure()
        plt.plot(self.train_acc_list, label='train_acc')
        plt.plot(self.test_acc_list, label='test_acc')
        plt.xscale('log')
        plt.legend()
        plt.title('Acc')
        plt.savefig(f'figs/{self.info}_acc.png')

    def Transformer_prepare_graph(self):
        self.x = []
        self.y = []
        self.z = []
        self.w = []
        self.k = []

        for _ in range(100):
            src = 2 * torch.rand(self.seq_len, 1)
            srcsave = src.clone()
            src = src.unsqueeze(2)
            q = self.model.query(src).transpose(0, 1)
            key = self.model.key(src).transpose(0, 1)
            v = self.model.value(src).transpose(0, 1)
            attnscores = torch.bmm(q, key.transpose(1, 2)) / (self.model.hidden_size ** 0.5)
            attnscores = attnscores + self.model.bias
            if self.has_softmax:
                attnscores = nn.functional.softmax(attnscores, dim=-1)
            src = torch.bmm(attnscores, v).transpose(0, 1)

            self.x.append(src[0, 0, 0].item())
            if self.hidden_dimension > 1:
                self.y.append(src[0, 0, 1].item())
            else:
                self.y.append(src[0, 0, 0].item())
            self.z.append(self.model(srcsave)[0].item())
            self.w.append(self.target_function(srcsave.reshape(-1)))
            self.k.append(sum(src.reshape(-1)).item())

    def Deepset_prepare_graph(self):
        self.x = []
        self.y = []
        self.z = []
        self.w = []
        self.k = []

        for _ in range(100):
            src = 2 * torch.rand(self.seq_len, 1).to(self.device)
            srcsave = src.clone()
            src = src.unsqueeze(2)
            seq_len, batch_size, _ = src.size()

            # Apply MLP1 to each item in the sequence
            src = src.reshape(seq_len * batch_size, -1)  # (seq_len * batch_size, 1)
            src = self.model.mlp1(src)  # (seq_len * batch_size, k)
            src = src.reshape(seq_len, batch_size, -1)  # (seq_len, batch_size, k)

            # Calculate mean over the sequence dimension
            src = src.mean(dim=0)

            self.x.append(src[0, 0].item())
            if src.shape[1] > 1:
                self.y.append(src[0, 1].item())
            else:
                self.y.append(src[0, 0].item())
            self.z.append(self.model(srcsave).item())
            self.w.append(self.target_function(srcsave.reshape(-1)))
            self.k.append(sum(src.reshape(-1)).item())

    def plot_scatter_graph(self):
        plt.figure()
        plt.scatter(self.x, self.y, c=self.w, s=20)
        plt.colorbar(label='Sum of Input')
        plt.xlabel('MLP input dim-1')
        plt.ylabel('MLP input dim-2')
        plt.savefig(f'figs/{self.info}_scatter.png')

    def plot_target_function(self):
        plt.figure()
        sorted_data = sorted(zip(self.x, self.w))
        sorted_x, sorted_y = zip(*sorted_data)
        plt.scatter(sorted_x, sorted_y, label='Scatter')
        plt.plot(sorted_x, sorted_y, color='red', label='Line')
        plt.title('Target Function')
        plt.savefig(f'figs/{self.info}_target.png')

    def plot_model_output(self):
        plt.figure()
        sorted_data = sorted(zip(self.x, self.z))
        sorted_x, sorted_y = zip(*sorted_data)
        plt.scatter(sorted_x, sorted_y, label='Scatter')
        plt.plot(sorted_x, sorted_y, color='red', label='Line')
        plt.title('Model Output')
        plt.savefig(f'figs/{self.info}_output.png')

    def diag_single_sample(self):
        src = 2 * torch.rand(self.seq_len, 1).to(self.device)
        print(f"srcshape: {src.shape}")
        print(f"outputshape: {self.model(src).shape}")
        print(f"Input: {src}")
        print(f"Target: {self.target_function(src)}")
        print(f"Output: {self.model(src)[0].item()}")
    
    def diag_final_loss(self,show=True):
        #return tuple (train_loss, test_loss)
        if show:
            print(f'Final train loss: {self.train_loss_list[-1]}')
            print(f'Final test loss: {self.test_loss_list[-1]}')
        return self.train_loss_list[-1],self.test_loss_list[-1]
    
    def diag_final_acc(self,show=True):
        #return tuple (train_acc, test_acc)
        if show:
            print(f'Final train acc: {self.train_acc_list[-1]}')
            print(f'Final test acc: {self.test_acc_list[-1]}')
        return self.train_acc_list[-1],self.test_acc_list[-1]

if __name__ == "__main__":
    
    config = {
        'device': 'cuda',
        'seq_len': 6,
        'target_function': lambda x: torch.mean(x ** torch.mean(x)).item(),
        'filepath': 'data.pth', #For data save&load, data is automatically saved if generate_data=True
        'generate_data': True,
        'num_samples': 256,
        'batch_size': 24,
        'trainratio': 0.75, #train set percent
        'num_epochs': 1000,
        'hidden_dimension': 96,
        'sum_dimension': 1,
        'lr': 0.003,
        'manual_build_model': False #Manually define trainer.model and trainer.optimizer after __init__
    }

    trainer = Trainer(config)
    trainer.train()
    trainer.Deepset_prepare_graph()
    trainer.plot_scatter_graph()
    trainer.plot_target_function()
    trainer.plot_model_output()
    trainer.diag_single_sample()
    trainer.diag_final_acc()
    trainer.diag_final_loss()
