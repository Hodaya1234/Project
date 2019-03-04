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


def train_model(model, train_dataset, valid_dataset, test_dataset, optimizer, scheduler, loss_fn, n_epochs, cv=True):
    train_losses = []
    valid_losses = []
    test_losses = []
    test_accuracies = []
    if cv:
        train_loaders = []
        for train_d in train_dataset:
            train_loaders.append(data_utils.DataLoader(train_d, batch_size=2, shuffle=True))
        for e in range(n_epochs):
            for one_train_loader, one_valid_set, one_test_set in zip(train_loaders, valid_dataset, test_dataset):
                scheduler.step()
                epoch_train_loss = []
                model.train()
                for x, y in one_train_loader:
                    optimizer.zero_grad()
                    y_pred = model(x)
                    y_pred = y_pred.view(y_pred.numel())
                    train_loss = loss_fn(y_pred, y)
                    train_loss.backward()
                    optimizer.step()
                    epoch_train_loss.append(train_loss.item())
                train_loss_mean = sum(epoch_train_loss) / len(epoch_train_loss)
                train_losses.append(train_loss_mean)
                model.eval()
                outputs = model(one_valid_set.all_x)
                outputs = outputs.view(outputs.numel())
                valid_losses.append(loss_fn(outputs, one_valid_set.all_y).item())

                outputs = model(one_test_set.all_x)
                outputs = outputs.view(outputs.numel())
                test_losses.append(loss_fn(outputs, one_test_set.all_y).item())

                if e % 100 == 0:
                    test_y_int = one_test_set.all_y.int()
                    outputs = model(one_test_set.all_x)
                    outputs = outputs.view(outputs.numel())
                    predictions = torch.round(outputs).int().view(outputs.numel())
                    correct = (predictions == test_y_int).sum().item()
                    test_accuracies.append(correct)
                    # print("{}. train loss: {}   test_loss: {}".format(e, train_loss_mean, test_loss_mean))
                    print('mean pred of y=0: {}'.format(torch.mean(outputs[test_y_int == 0]).item()))
                    print('mean pred of y=1: {}'.format(torch.mean(outputs[test_y_int == 1]).item()))
                    print('accuracy on the augmented data: {0:.2f}'.format(correct / len(test_y_int)))

    else:
        train_loader = data_utils.DataLoader(train_dataset, batch_size=2, shuffle=True)
        valid_y_int = valid_dataset.all_y.int()
        for e in range(n_epochs):
            scheduler.step()
            epoch_train_loss = []
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = model(x)
                y_pred = y_pred.view(y_pred.numel())
                train_loss = loss_fn(y_pred, y)
                train_loss.backward()
                optimizer.step()
                epoch_train_loss.append(train_loss.item())
            train_loss_mean = sum(epoch_train_loss) / len(epoch_train_loss)
            train_losses.append(train_loss_mean)
            model.eval()
            outputs = model(valid_dataset.all_x)
            outputs = outputs.view(outputs.numel())
            test_loss_mean = loss_fn(outputs, valid_dataset.all_y).item()
            valid_losses.append(test_loss_mean)
            if e % 100 == 0:
                print("{}. train loss: {}   test_loss: {}".format(e, train_loss_mean, test_loss_mean))
                correct = 0
                predictions = torch.round(outputs).int().view(outputs.numel())
                correct += (predictions == valid_y_int).sum().item()
                print('mean pred of y=0: {}'.format(torch.mean(outputs[valid_y_int == 0]).item()))
                print('mean pred of y=1: {}'.format(torch.mean(outputs[valid_y_int == 1]).item()))
                print('accuracy on the augmented data: {0:.2f}'.format(correct / len(valid_y_int)))
    return model, train_losses, valid_losses, test_losses


def run_model(data_sets, cv=True):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    train_dataset, valid_dataset, test_dataset = data_sets  # DataSet objects
    if cv:
        D_in = train_dataset[0].all_x.shape[1]
    else:
        D_in = train_dataset[0].shape[1]
    H1 = 500
    H2 = 200
    H3 = 50
    D_out = 1

    model = SimpleNet(D_in, H1, H2, H3, D_out).double().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10000)
    loss_fn = nn.BCELoss()
    n_epochs = 100

    model, train_losses, validation_losses, test_losses = train_model(model, train_dataset, valid_dataset, train_dataset,
                                                         optimizer, scheduler,
                                                         loss_fn, n_epochs, cv=cv)

    return model, train_losses, validation_losses, test_losses

