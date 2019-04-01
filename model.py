import torch
import numpy as np
from torch import optim
from torch import nn
from dense_net import DenseNet
from data_set import DataSet
import torch.utils.data as data_utils
import data_set


def train_model(model, train_dataset, valid_dataset, test_dataset, optimizer, scheduler, loss_fn, n_epochs, cv):
    train_losses = []
    valid_losses = []
    test_losses = []
    test_accuracies = []
    if cv:
        train_loaders = []
        for train_d in train_dataset:
            train_loaders.append(data_utils.DataLoader(train_d, batch_size=8, shuffle=True))
        for e in range(n_epochs + 1):
            scheduler.step()
            for one_train_loader, one_valid_set, one_test_set in zip(train_loaders, valid_dataset, test_dataset):
                epoch_train_loss = []
                epoch_valid_loss = []
                epoch_test_loss = []
                model.train()
                for x, y in one_train_loader:
                    optimizer.zero_grad()
                    y_pred = model(x)
                    y_pred = y_pred.view(y_pred.numel())
                    loss = loss_fn(y_pred, y)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss.append(loss.item())

                model.eval()
                outputs = model(one_valid_set.all_x)
                outputs = outputs.view(outputs.numel())
                loss = loss_fn(outputs, one_valid_set.all_y).item()
                epoch_valid_loss.append(loss)

                outputs = model(one_test_set.all_x)
                outputs = outputs.view(outputs.numel())
                loss = loss_fn(outputs, one_test_set.all_y).item()
                epoch_test_loss.append(loss)

                train_losses.append(sum(epoch_train_loss) / len(epoch_train_loss))
                valid_losses.append(sum(epoch_valid_loss) / len(epoch_valid_loss))
                test_losses.append(sum(epoch_test_loss) / len(epoch_test_loss))

            if e % 2 == 0:
                print("{}. train loss: {}   valid_loss: {}  test_loss: {}".format(
                    e, train_losses[-1], valid_losses[-1], test_losses[-1]))
                test_y_int = test_dataset[0].all_y.int()
                outputs = model(test_dataset[0].all_x)
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
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())
            train_loss_mean = sum(epoch_train_loss) / len(epoch_train_loss)
            train_losses.append(train_loss_mean)
            model.eval()
            outputs = model(valid_dataset.all_x)
            outputs = outputs.view(outputs.numel())
            loss = loss_fn(outputs, valid_dataset.all_y).item()
            valid_losses.append(loss)
            if e % 10 == 0:
                print("{}. train loss: {}   test_loss: {}".format(e, train_loss_mean, loss))
                correct = 0
                predictions = torch.round(outputs).int().view(outputs.numel())
                correct += (predictions == valid_y_int).sum().item()
                print('mean pred of y=0: {}\tmean pred of y=1: {}\taccuracy on the augmented data: {}'.format(
                    torch.mean(outputs[valid_y_int == 0]).item(), torch.mean(outputs[valid_y_int == 1]).item(),
                    correct / len(valid_y_int)))

    return model, train_losses, valid_losses, test_losses


def run_model(model, data_sets, cv=True, norm=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("available devices: {}".format(torch.cuda.device_count()))
    train_dataset, valid_dataset, test_dataset = data_set.data_to_cuda(data_sets, device, cv)  # DataSet objects
    if norm:
        train_dataset, valid_dataset, test_dataset = data_set.normalize_datasets([train_dataset, valid_dataset, test_dataset], cv)

    model = model.double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5])
    loss_fn = nn.BCELoss()
    n_epochs = 10

    model, train_losses, validation_losses, test_losses = train_model(model, train_dataset,
                                                                      valid_dataset, train_dataset, optimizer,
                                                                      scheduler, loss_fn, n_epochs, cv=cv)

    return model, train_losses, validation_losses, test_losses


def test_model(model, test_x, test_y, loss_fn=nn.BCELoss()):
    outputs = model(test_x)
    outputs = outputs.view(outputs.numel())
    mean_loss = loss_fn(outputs, test_y.double()).item()

    predictions = torch.round(outputs).int().view(outputs.numel())
    accuracy = (predictions == test_y.int()).sum().item()
    return accuracy, mean_loss


def run_with_missing_segments(model, segments_map, test_set, cv):
    seg_numbers = np.unique(segments_map)
    if seg_numbers[0] == 0:
        seg_numbers = seg_numbers[1:]
    seg_acc_map = np.zeros(len(seg_numbers))
    seg_loss_map = np.copy(seg_acc_map)

    if cv:
        # test_x: n_examples X n_segments X n_frames

        for one_test_set in test_set:
            for idx, num in enumerate(seg_numbers):
                new_test_x = one_test_set.all_x.clone()
                new_test_x[:, idx, ] = 0  # is 0 the right choice here?
                curr_acc, curr_loss = test_model(model, new_test_x, one_test_set.all_y)
                seg_acc_map[idx] += curr_acc
                seg_loss_map[idx] += curr_loss
        seg_acc_map = np.divide(seg_acc_map, len(test_set))
        seg_loss_map = np.divide(seg_loss_map, len(test_set))
    else:
        for idx, num in enumerate(seg_numbers):
            new_test_x = test_set.all_x.clone()
            new_test_x[:, idx, ] = 0  # is 0 the right choice here?
            seg_acc_map[idx], seg_loss_map[idx] = test_model(model, new_test_x, test_set.all_y)

    return seg_acc_map, seg_loss_map
