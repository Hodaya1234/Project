import torch.nn as nn
import torch


class DenseNet(nn.Module):
    def __init__(self, D_in, H1, H2, D_out, p=0.5):
        super(DenseNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=p)
        self.batchnorm = nn.BatchNorm1d(D_in)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.relu(self.drop(self.linear1(x)))
        x = self.relu(self.drop(self.linear2(x)))
        x = torch.sigmoid(self.linear3(x))
        return x

    def init_weight(self):
        self.apply(weights_init)


def get_model(D_in):
    H1 = 150
    H2 = 10
    D_out = 1
    p = 0.5
    return DenseNet(D_in, H1, H2, D_out, p)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)


def init_weights(net):
    for modul in net._modules.values():
        if type(modul) == nn.Linear:
            nn.init.normal_(modul.weight, mean=0, std=1)
            prev_size = modul.weight.shape[1]
            modul.weight.data = modul.weight * torch.sqrt(torch.tensor(2, dtype=torch.float32)/prev_size)
    return net
