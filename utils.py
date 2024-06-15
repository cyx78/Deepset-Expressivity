import torch
import torch.nn as nn
import torch.optim as optim

class Transformer2Layer(nn.Module):
    def __init__(self,seq_len,hidden_size,output_dim,MLP_hidden_dimension,has_softmax
                 ):
        super().__init__()
        #self.embedding_layer = nn.Embedding(2, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.hidden_size=hidden_size
        self.bias = nn.Parameter(torch.randn(1,seq_len,seq_len))
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, MLP_hidden_dimension),
            nn.ReLU(),
            nn.Linear(MLP_hidden_dimension, output_dim)
        )

        self.query2 = nn.Linear(hidden_size, hidden_size)
        self.key2 = nn.Linear(hidden_size, hidden_size)
        self.value2 = nn.Linear(hidden_size, hidden_size)
        self.bias2 = nn.Parameter(torch.randn(1,seq_len,seq_len))
        self.feedforward2 = nn.Sequential(
            nn.Linear(hidden_size, MLP_hidden_dimension),
            nn.ReLU(),
            nn.Linear(MLP_hidden_dimension, output_dim)
        )
        self.has_softmax=has_softmax

    def forward(self, src):
        # src shape: (seq_len, batch_size, hidden_size)
        #src=self.embedding_layer(src)
        src=src.unsqueeze(2)
        #print("First Layer src shape",src.shape)

        q = self.query(src).transpose(0,1)
        k = self.key(src).transpose(0,1)
        v = self.value(src).transpose(0,1)
        attnscores = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attnscores = attnscores+self.bias

        if self.has_softmax:
            attnscores = nn.functional.softmax(attnscores, dim=-1)
        src = torch.bmm(attnscores, v).transpose(0,1)

        src = self.feedforward(src)

        q = self.query2(src).transpose(0,1)
        k = self.key2(src).transpose(0,1)
        v = self.value2(src).transpose(0,1)
        attnscores = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attnscores = attnscores+self.bias2

        if self.has_softmax:
            attnscores = nn.functional.softmax(attnscores, dim=-1)
        src = torch.bmm(attnscores, v).transpose(0,1)

        src = self.feedforward2(src)
        

        return src[0,:,0]
import torch

def Gen_Rand_Dataset(seq_len, function, num_samples=1024, filepath='dataset.pth'):
    data = []
    for _ in range(num_samples):
        x = 2 * torch.rand(seq_len)
        data.append((x, function(x)))
    # Save the dataset
    torch.save(data, filepath)

def Load_Dataset(filepath, batch_size, trainratio):
    # Load the dataset
    data = torch.load(filepath)
    # Split into training and testing sets
    train_data = data[:int(len(data) * trainratio)]
    test_data = data[int(len(data) * trainratio):]
    # Create DataLoader objects
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return trainloader, testloader

class DeepSet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim_n=128, hidden_dim_k=64):
        super(DeepSet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_n = hidden_dim_n
        self.hidden_dim_k = hidden_dim_k
        
        # MLP 1: 1 -> n -> k
        self.mlp1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim_n),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_n, self.hidden_dim_k),
            nn.ReLU()
        )
        
        # MLP 2: k -> n -> 1
        self.mlp2 = nn.Sequential(
            nn.Linear(self.hidden_dim_k, self.hidden_dim_n),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_n, 1)
        )

    def forward(self, x):
        x=x.unsqueeze(2)
        # x is of shape (seq_len, batch_size, 1)
        seq_len, batch_size, _ = x.size()
        
        # Apply MLP1 to each item in the sequence
        x = x.reshape(seq_len * batch_size, -1)  # (seq_len * batch_size, 1)
        x = self.mlp1(x)  # (seq_len * batch_size, k)
        x = x.reshape(seq_len, batch_size, -1)  # (seq_len, batch_size, k)
        
        # Calculate mean over the sequence dimension
        x = x.mean(dim=0)  # (batch_size, k)
        
        # Apply MLP2
        x = self.mlp2(x)  # (batch_size, 1)
        
        # Reshape to (batch_size,)
        x = x.reshape(batch_size)
        
        return x
class TransformerLayer(nn.Module):
    def __init__(self,seq_len,hidden_size,output_dim,MLP_hidden_dimension,has_softmax
                 ):
        super().__init__()
        #self.embedding_layer = nn.Embedding(2, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.hidden_size=hidden_size
        self.bias = nn.Parameter(torch.randn(1,seq_len,seq_len))
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, MLP_hidden_dimension),
            nn.ReLU(),
            nn.Linear(MLP_hidden_dimension, output_dim)
        )
        self.has_softmax=has_softmax

    def forward(self, src):
        # src shape: (seq_len, batch_size, hidden_size)
        #src=self.embedding_layer(src)
        src=src.unsqueeze(2)

        q = self.query(src).transpose(0,1)
        k = self.key(src).transpose(0,1)
        v = self.value(src).transpose(0,1)
        attnscores = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_size ** 0.5)
        attnscores = attnscores+self.bias

        if self.has_softmax:
            attnscores = nn.functional.softmax(attnscores, dim=-1)
        src = torch.bmm(attnscores, v).transpose(0,1)

        src = self.feedforward(src)

        return src[0,:,0]

def sum_of_and_pairs(binary_list):
    n = len(binary_list) // 2
    result = 0
    for i in range(n):
        result += binary_list[i] & binary_list[i + n]
    return result%2