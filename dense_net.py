import torch.nn as nn
import torch


class DenseNet(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out, p=0.5):
        super(DenseNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H2)
        # self.linear2 = nn.Linear(H1, H2)
        # self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H2, D_out)
        self.relu = nn.ReLU()
        # self.drop = nn.Dropout(p=p)
        self.batchnorm = nn.BatchNorm1d(D_in)

    def forward(self, x):
        x = self.batchnorm(x)
        x = torch.relu(self.linear1(x))
        # x = self.relu(self.linear2(x))
        # x = self.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        return x


def get_model(D_in):
    H1 = 500
    H2 = 100
    H3 = 50
    D_out = 1
    p = 0.5
    return DenseNet(D_in, H1, H2, H3, D_out, p)


def init_weights(net):
    for modul in net._modules.values():
        if type(modul) == nn.Linear:
            nn.init.normal_(modul.weight, mean=0, std=1)
            prev_size = modul.weight.shape[1]
            modul.weight.data = modul.weight * torch.sqrt(torch.tensor(2, dtype=torch.float32)/prev_size)
    return net