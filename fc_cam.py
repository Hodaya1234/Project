import torch
from torch import nn
import torchvision
import csv
import numpy as np
from torch import optim

class Net(nn.Module):
    def __init__(self, D_in, H1, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, D_in)
        self.linear2 = nn.Linear(D_in, H1)
        self.linear3 = nn.Linear(H1, D_out)

    def forward(self, x):
        input_vec = x
        out = self.linear1(x)
        out += input_vec
        out = self.linear2(out)
        out = torch.sigmoid(self.linear3(out))
        return out


def read_data():
    # read data
    data = np.load('temp_outputs/mnist_test.npz')
    return data['X_train'], data['Y_train'], data['X_test'], data['Y_test']


X_train, Y_train, X_test, Y_test = read_data()
D_in = X_train.shape[1]
H1 = 100
D_out = len(np.unique(Y_test))

net = Net(D_in, H1, D_out)
optimizer = optim.Adam(net.parameters(), lr=1e-2)

# train
n_epoch = 100
# for e in range(n_epoch):
    
