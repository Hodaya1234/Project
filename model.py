import torch
import numpy as np
import torch.utils.data as data_utils
from torch import optim


class SimpleNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(SimpleNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        return x


class DataSet(data_utils.Dataset):
    def __init__(self, x, y):
        super(DataSet, self).__init__()
        self.all_x = x
        self.all_y = y

    def __getitem__(self, index):
        x = self.all_x[index]
        y = self.all_y[index]
        return x, y

    def __len__(self):
        return len(self.all_y)


def train(data_sets):
    train_x, train_y, valid_x, valid_y, test_x, test_y = data_sets
    # create the dataset class and data loader:
    train_dataset = DataSet(train_x, train_y)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=8, shuffle=True)

    # create a basic FC network:
    D_in = train_x.shape[1]
    H1 = 500
    H2 = 200
    H3 = 50
    D_out = 1

    model = SimpleNet(D_in, H1, H2, H3, D_out).double()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()

    n_epochs = 100
    for e in range(n_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred = y_pred.view(y_pred.numel())
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

        # if e % 5 == 0:
        #     # accuracy on the real data:
        #     correct = 0
        #     outputs = model(test_x_real)
        #     predictions = torch.round(outputs).int().view(outputs.numel())
        #     correct += (predictions == test_y_real).sum().item()
        #     print('accuracy on the real data: {0:.2f}'.format(correct / n_real))
        #
        #     # accuracy on the augmented data:
        #     correct = 0
        #     outputs = model(test_x)
        #     predictions = torch.round(outputs).int().view(outputs.numel())
        #     correct += (predictions == test_y).sum().item()
        #     print('accuracy on the augmented data: {0:.2f}'.format(correct / n_test_examples))
