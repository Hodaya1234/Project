import torch
import numpy as np
import torch.utils.data as data_utils
from torch import optim
from torch import nn

class SimpleNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)
        self.lrelu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.drop(self.lrelu(self.linear1(x)))
        x = self.drop(self.lrelu(self.linear2(x)))
        x = self.lrelu(self.linear3(x))
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


def data_to_cude(data_sets, device):
    new_data = []
    for s in data_sets:
        new_data.append(s.to(device))
    return new_data


def train(data_sets):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_x, train_y, valid_x, valid_y, test_x, test_y = data_to_cude(data_sets, device)
    # create the dataset class and data loader:
    train_dataset = DataSet(train_x, train_y)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_dataset = DataSet(test_x, test_y)
    # test_loader = data_utils.DataLoader(test_dataset, batch_size=8, shuffle=False)

    # create a basic FC network:
    D_in = train_x.shape[1]
    H1 = 500
    H2 = 200
    H3 = 50
    D_out = 1

    # model = SimpleNet(D_in, H1, H2, H3, D_out).double().to(device)
    model = nn.Sequential(nn.Linear(D_in, D_out), nn.Sigmoid()).double().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10000)
    loss_fn = nn.BCELoss()

    n_epochs = 150000
    train_losses = []
    test_losses = []
    test_y = test_y
    test_y_int = test_y.int()

    for e in range(n_epochs):
        scheduler.step()
        epoch_train_loss = []
        # epoch_test_loss = []
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred = y_pred.view(y_pred.numel())
            train_loss = loss_fn(y_pred, y)
            train_loss.backward()
            optimizer.step()
            epoch_train_loss.append(train_loss.item())
        train_loss_mean = sum(epoch_train_loss)/len(epoch_train_loss)
        train_losses.append(train_loss_mean)
        model.eval()
        outputs = model(test_x)
        outputs = outputs.view(outputs.numel())
        test_loss_mean = loss_fn(outputs, test_y).item()
        test_losses.append(test_loss_mean)
        if e % 100 == 0:
            print("{}. train loss: {}   test_loss: {}".format(e, train_loss_mean, test_loss_mean))
            correct = 0
            predictions = torch.round(outputs).int().view(outputs.numel())
            correct += (predictions == test_y_int).sum().item()
            print('mean pred of y=0: {}'.format(torch.mean(outputs[test_y == 0]).item()))
            print('mean pred of y=1: {}'.format(torch.mean(outputs[test_y == 1]).item()))
            print('accuracy on the augmented data: {0:.2f}'.format(correct / len(test_y)))

    return train_losses, test_losses
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
