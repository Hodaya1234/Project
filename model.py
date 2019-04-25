import torch
import numpy as np
from torch import optim
from torch import nn
from dense_net import DenseNet
from data_set import DataSet
import torch.utils.data as data_utils
import data_set


def train_single_data_set(model, train_dataset, valid_dataset, test_dataset, optimizer, scheduler, loss_fn, n_epochs):
    train_losses = []
    valid_losses = []
    test_losses = []
    test_accuracies = []
    train_loader = data_utils.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_y_int = test_dataset.all_y.int()
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

        outputs = model(test_dataset.all_x)
        outputs = outputs.view(outputs.numel())
        loss = loss_fn(outputs, test_dataset.all_y).item()
        test_losses.append(loss)

        predictions = torch.round(outputs).int().view(outputs.numel())
        accuracy = (predictions == test_y_int).sum().item() / len(test_y_int)
        test_accuracies.append(accuracy)

        if e % 5 == 0:
            print("{}. train loss: {}   valid_loss: {}  test_loss: {}".format(
                e, train_losses[-1], valid_losses[-1], test_losses[-1]))

    return model, train_losses, valid_losses, test_losses, test_accuracies


def train_model(model, train_dataset, valid_dataset, test_dataset, optimizer, scheduler, loss_fn, n_epochs, cv):
    # TODO reorganize this
    n_sets = len(train_dataset)
    train_losses = np.zeros([n_sets, n_epochs])
    valid_losses = np.zeros_like(train_losses)
    test_losses = np.zeros_like(train_losses)
    test_accuracies = np.zeros_like(train_losses)
    train_loaders = []
    for train_d in train_dataset:
        train_loaders.append(data_utils.DataLoader(train_d, batch_size=32, shuffle=True))
    for idx, one_train_loader, one_valid_set, one_test_set in zip(range(n_sets), train_loaders, valid_dataset,
                                                                  test_dataset):
        for e in range(n_epochs):
            scheduler.step()
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

            train_losses[idx, e] = sum(epoch_train_loss) / len(epoch_train_loss)
            valid_losses[idx, e] = epoch_valid_loss
            test_losses[idx, e] = epoch_test_loss
            test_accuracies[idx, e] = epoch_test_acc

            if e % 2 == 0:
                print("{}. train loss: {}   valid_loss: {}  test_loss: {}".format(
                    e, sum(epoch_train_loss) / len(epoch_train_loss), epoch_valid_loss, epoch_test_loss))

    return model, train_losses, valid_losses, test_losses, test_accuracies


def run_model(model, data_sets, cv=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("available devices: {}".format(torch.cuda.device_count()))
    train_dataset, valid_dataset, test_dataset = data_set.data_to_cuda(data_sets, device, cv)  # DataSet objects

    model = model.double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [4])
    loss_fn = nn.BCELoss()
    n_epochs = 50
    if not cv:
        model, train_losses, valid_losses, test_losses, test_accuracies = train_single_data_set(model, train_dataset,
                                                                      valid_dataset, train_dataset, optimizer,
                                                                      scheduler, loss_fn, n_epochs)
    else:
        n_sets = len(train_dataset)
        train_losses = np.zeros([n_sets, n_epochs])
        valid_losses = np.zeros_like(train_losses)
        test_losses = np.zeros_like(train_losses)
        test_accuracies = np.zeros_like(train_losses)
        train_loaders = []
        for train_d in train_dataset:
            train_loaders.append(data_utils.DataLoader(train_d, batch_size=32, shuffle=True))
        for idx, one_train_loader, one_valid_set, one_test_set in zip(range(n_sets), train_loaders, valid_dataset,
                                                                      test_dataset):
            model.init_weight()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [4])
            _, one_train_losses, one_valid_losses, one_test_losses, one_test_accuracies = train_single_data_set(model, train_dataset[idx],
                                                                                        valid_dataset[idx], train_dataset[idx],
                                                                                        optimizer,
                                                                                        scheduler, loss_fn, n_epochs)
            train_losses[idx, :] = one_train_losses
            valid_losses[idx, :] = one_valid_losses
            test_losses[idx, :] = one_test_losses
            test_accuracies[idx, :] = one_test_accuracies

    return model, train_losses, valid_losses, test_losses, test_accuracies


def test_model(model, test_x, test_y, loss_fn=nn.BCELoss()):
    outputs = model(test_x)
    outputs = outputs.view(outputs.numel())
    mean_loss = loss_fn(outputs, test_y.double()).item()

    predictions = torch.round(outputs).int().view(outputs.numel())
    accuracy = (predictions == test_y.int()).sum().item() / len(test_y)
    return accuracy, mean_loss


def get_all_missing_indexes(n_ponits, n_seg, n_frames, part_type):
    index_array = np.asarray(list(range(n_ponits)))
    if part_type == 'both':
        return index_array
    index_array = np.reshape(index_array, [n_seg, n_frames])
    if part_type == 'segments':
        return np.asarray([index_array[i, :] for i in range(n_seg)])
    if part_type == 'frames':
        return np.asarray([index_array[:, i] for i in range(n_frames)])


def run_with_missing_parts(model, segments_map, test_set, cv, n_frames, part_type='segments', zero_all=True, value_type='loss'):
    seg_numbers = np.unique(segments_map)
    if seg_numbers[0] == 0:
        seg_numbers = seg_numbers[1:]
    n_seg = len(seg_numbers)
    n_points = n_frames * n_seg
    if part_type == 'segments':
        loss_map = np.zeros(n_seg)
    elif part_type == 'frames':
        loss_map = np.zeros(n_frames)
    elif part_type == 'both':
        loss_map = np.zeros(n_points)

    if cv:
        # test_x: n_examples X n_segments * n_frames
        for one_test_set in test_set:
            test_x = one_test_set.all_x
            all_indexes = get_all_missing_indexes(n_points, n_seg, n_frames, part_type)
            for iter, indexes in enumerate(all_indexes):
                if zero_all:
                    new_test_x = torch.zeros_like(test_x)
                    new_test_x[:, indexes] = test_x[:, indexes]
                else:
                    new_test_x = test_x.clone()
                    new_test_x[:, indexes] = 0

                curr_acc, curr_loss = test_model(model, new_test_x, one_test_set.all_y)
                if value_type == 'loss':
                    loss_map[iter] += curr_loss
                else:
                    loss_map[iter] += curr_acc
        loss_map = np.divide(loss_map, len(test_set))

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
