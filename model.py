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
            train_loaders.append(data_utils.DataLoader(train_d, batch_size=32, shuffle=True))
        for e in range(n_epochs):
            scheduler.step()
            for one_train_loader, one_valid_set, one_test_set in zip(train_loaders, valid_dataset, test_dataset):
                epoch_train_loss = []

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
                epoch_valid_loss = loss_fn(outputs, one_valid_set.all_y).item()

                outputs = model(one_test_set.all_x)
                outputs = outputs.view(outputs.numel())
                epoch_test_loss = loss_fn(outputs, one_test_set.all_y).item()

                predictions = torch.round(outputs).int().view(outputs.numel())
                epoch_test_acc = (predictions == one_test_set.all_y.int()).sum().item()

                train_losses.append(sum(epoch_train_loss) / len(epoch_train_loss))
                valid_losses.append(epoch_valid_loss)
                test_losses.append(epoch_test_loss)
                test_accuracies.append(epoch_test_acc)

            if e % 2 == 0:
                print("{}. train loss: {}   valid_loss: {}  test_loss: {}".format(
                    e, train_losses[-1], valid_losses[-1], test_losses[-1]))


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


# def tune_parameters(model, data_sets, cv=True)


def run_model(model, data_sets, cv=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("available devices: {}".format(torch.cuda.device_count()))
    train_dataset, valid_dataset, test_dataset = data_set.data_to_cuda(data_sets, device, cv)  # DataSet objects

    model = model.double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [4])
    loss_fn = nn.BCELoss()
    n_epochs = 15

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


def run_with_missing_parts(model, segments_map, test_set, cv, n_frames, part_type='segments', zero_all=True, value_type='loss'):
    seg_numbers = np.unique(segments_map)
    if seg_numbers[0] == 0:
        seg_numbers = seg_numbers[1:]
    n_seg = len(seg_numbers)
    n_points = n_frames * n_seg
    if part_type == 'segments':
        loss_map = np.zeros(n_seg)
    else:
        loss_map = np.zeros(n_frames)

    if cv:
        # test_x: n_examples X n_segments * n_frames
        for one_test_set in test_set:
            test_x = one_test_set.all_x
            test_x = np.reshape(test_x, (-1, n_seg, n_frames))
            for idx in range(len(loss_map)):
                if zero_all:
                    new_test_x = torch.zeros_like(test_x)
                    if part_type == 'segments':
                        new_test_x[:, idx, :] = test_x[:, idx, :]
                    else:
                        new_test_x[:, :, idx] = test_x[:, :, idx]
                else:
                    new_test_x = test_x.clone()
                    if part_type == 'segments':
                        new_test_x[:, idx, :] = 0
                    else:
                        new_test_x[:, :, idx] = 0
                new_test_x = np.reshape(new_test_x, (-1, n_points))
                curr_acc, curr_loss = test_model(model, new_test_x, one_test_set.all_y)
                if value_type == 'loss':
                    loss_map[idx] += curr_loss
                else:
                    loss_map[idx] += curr_acc
    else:
        n_points = test_set.all_x.shape[1]
        n_frames = int(n_points / n_seg)
        test_x = np.reshape(test_set.all_x, (-1, n_seg, n_frames))
        for idx, num in enumerate(seg_numbers):
            new_test_x = test_x.clone()
            new_test_x = np.reshape(new_test_x, (-1, n_points))
            new_test_x[:, idx, :] = 0
            loss_map[idx], loss_map[idx] = test_model(model, new_test_x, test_set.all_y)

    return loss_map
