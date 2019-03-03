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
        self.lrelu = torch.nn.LeakyReLU()
        self.drop = torch.nn.Dropout(p=0.5)

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


def train(data_sets):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_x, train_y, valid_x, valid_y, test_x, test_y = data_sets
    # create the dataset class and data loader:
    train_dataset = DataSet(train_x, train_y)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_dataset = DataSet(test_x, test_y)
    # test_loader = data_utils.DataLoader(test_dataset, batch_size=8, shuffle=False)

    # create a basic FC network:
    D_in = train_x.shape[1]
    H1 = 500
    H2 = 200
    H3 = 50
    D_out = 1

    model = SimpleNet(D_in, H1, H2, H3, D_out).double()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()

    n_epochs = 200
    train_losses = []
    test_losses = []
    for e in range(n_epochs):
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
        # for x, y in test_loader:
        #     outputs = model(x)
        #     outputs = outputs.view(outputs.numel())
        #     epoch_test_loss.append(loss_fn(outputs, y).item())
        outputs = model(test_x)
        outputs = outputs.view(outputs.numel())
        test_loss_mean = loss_fn(outputs, test_y).item()
        test_losses.append(test_loss_mean)
        print("{}. train loss: {}   test_loss: {}".format(e, train_loss_mean, test_loss_mean))

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
