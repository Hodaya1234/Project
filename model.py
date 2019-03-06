import torch
import numpy as np
import torch.utils.data as data_utils
from torch import optim
from torch import nn


class SimpleNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(SimpleNet, self).__init__()
        # self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(D_in, H2)
        # self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H2, D_out)
        self.relu = nn.ReLU()
        # self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        # x = self.lrelu(self.linear1(x))
        x = self.relu(self.linear2(x))
        # x = self.lrelu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        return x


class DataSet(data_utils.Dataset):
    def __init__(self, x, y):
        super(DataSet, self).__init__()
        self.all_x = x
        self.all_y = y
        self.n_data = len(y)

    def __getitem__(self, index):
        x = self.all_x[index]
        y = self.all_y[index]
        return x, y

    def __len__(self):
        return len(self.all_y)

    def change_device(self, device):
        return DataSet(self.all_x.to(device), self.all_y.to(device))

    def normalize(self):
        mean_x = self.all_x.mean().item()
        std_x = self.all_x.std().item()
        return DataSet(torch.div(torch.sub(self.all_x, mean_x), std_x), self.all_y)


def data_to_cuda(data_sets, device, cv=True):
    new_data = []
    if cv:
        for set_type in data_sets:
            set_type_list = []
            for one_dataset in set_type:
                set_type_list.append(one_dataset.change_device(device))
            new_data.append(set_type_list)
    else:
        for s in data_sets:
            new_data.append(s.change_device(device))
    return new_data


def normalize_datasets(data_sets, cv=True):
    new_data = []
    if cv:
        for set_type in data_sets:
            set_type_list = []
            for one_dataset in set_type:
                set_type_list.append(one_dataset.normalize())
            new_data.append(set_type_list)
    else:
        for s in data_sets:
            new_data.append(s.normalize())
    return new_data


def train_model(model, train_dataset, valid_dataset, test_dataset, optimizer, scheduler, loss_fn, n_epochs, cv):
    train_losses = []
    valid_losses = []
    test_losses = []
    test_accuracies = []
    if cv:
        train_loaders = []
        for train_d in train_dataset:
            train_loaders.append(data_utils.DataLoader(train_d, batch_size=32, shuffle=True))
        for e in range(n_epochs + 1):
            scheduler.step()
            for one_train_loader, one_valid_set, one_test_set in zip(train_loaders, valid_dataset, test_dataset):
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
                valid_loss_mean = loss_fn(outputs, one_valid_set.all_y).item()
                valid_losses.append(valid_loss_mean)

                outputs = model(one_test_set.all_x)
                outputs = outputs.view(outputs.numel())
                test_loss_mean = loss_fn(outputs, one_test_set.all_y).item()
                test_losses.append(test_loss_mean)
            if e % 2 == 0:
                print("{}. train loss: {}   valid_loss: {}  test_loss: {}".format(
                    e, train_loss_mean, valid_loss_mean, test_loss_mean))
                test_y_int = one_test_set.all_y.int()
                outputs = model(one_test_set.all_x)
                outputs = outputs.view(outputs.numel())
                predictions = torch.round(outputs).int().view(outputs.numel())
                correct = (predictions == test_y_int).sum().item()
                test_accuracies.append(correct)
                print('mean pred of y=0: {}\tmean pred of y=1: {}'.format(
                    torch.mean(outputs[test_y_int == 0]).item(), torch.mean(outputs[test_y_int == 1]).item()))
                print('accuracy on the augmented data: {0:.2f}'.format(correct / len(test_y_int)))

    else:
        train_loader = data_utils.DataLoader(train_dataset, batch_size=32, shuffle=True)
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
            if e % 10 == 0:
                print("{}. train loss: {}   test_loss: {}".format(e, train_loss_mean, test_loss_mean))
                correct = 0
                predictions = torch.round(outputs).int().view(outputs.numel())
                correct += (predictions == valid_y_int).sum().item()
                print('mean pred of y=0: {}\tmean pred of y=1: {}\taccuracy on the augmented data: {}'.format(
                    torch.mean(outputs[valid_y_int == 0]).item(), torch.mean(outputs[valid_y_int == 1]).item(),
                    correct / len(valid_y_int)))

    return model, train_losses, valid_losses, test_losses


def run_model(data_sets, cv=True, norm=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("available devices: {}".format(torch.cuda.device_count()))
    train_dataset, valid_dataset, test_dataset = data_to_cuda(data_sets, device, cv)  # DataSet objects
    if norm:
        train_dataset, valid_dataset, test_dataset = normalize_datasets([train_dataset, valid_dataset, test_dataset], cv)
    if cv:
        D_in = train_dataset[0].all_x.shape[1]
    else:
        D_in = train_dataset.all_x.shape[1]
    H1 = 500
    H2 = 100
    H3 = 50
    D_out = 1

    model = SimpleNet(D_in, H1, H2, H3, D_out).double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150])
    loss_fn = nn.BCELoss()
    n_epochs = 10

    model, train_losses, validation_losses, test_losses = train_model(model, train_dataset,
                                                                      valid_dataset, train_dataset, optimizer,
                                                                      scheduler, loss_fn, n_epochs, cv=cv)

    return model, train_losses, validation_losses, test_losses

