import torch
import numpy as np
from torch import optim
from torch import nn
from dense_net import DenseNet
from data_set import DataSet
import torch.utils.data as data_utils
import data_set


def train(model, datasets, parameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.double().to(device)

    train_dataset, valid_dataset = datasets
    loss_fn, n_epochs, optimizer, scheduler = parameters

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    train_loader = data_utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
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
        with torch.no_grad():
            outputs = model(valid_dataset.all_x)
            outputs = outputs.view(outputs.numel())
            loss = loss_fn(outputs, valid_dataset.all_y).item()
            valid_losses.append(loss)

            predictions = torch.round(outputs).int().view(outputs.numel())
            accuracy = torch.sum((predictions == valid_dataset.all_y.int())).item() / valid_dataset.all_y.shape[0]
            valid_accuracies.append(accuracy)
        if valid_losses[-1] < 0.01:
            print('finished at epoch {}'.format(e))
            break
        if e > 60 and np.mean(valid_losses[-5:]) > np.mean(valid_losses[-10:-5]):
            print('finished at epoch {}'.format(e))
            break
        if e % 10 == 0:
            print("{}. train loss: {}   valid_loss: {}  valid-acc:{}".format(
                e, train_losses[-1], valid_losses[-1], valid_accuracies[-1]))

    return model, train_losses, valid_losses, valid_accuracies


def get_train_params(model, loss_fn=nn.BCELoss(), n_epochs=80, lr=0.001, optimizer_type=optim.Adam, scheduler_type=optim.lr_scheduler.MultiStepLR, schedule_epochs=5):
    optimizer = optimizer_type(model.parameters(), lr=lr, weight_decay=0.5)
    scheduler = scheduler_type(optimizer, [30], gamma=0.1)
    return [loss_fn, n_epochs, optimizer, scheduler]


def test_model(model, test_x, test_y, loss_fn=nn.BCELoss()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.double().to(device)
    outputs = model(test_x.to(device))
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
    model.eval()
    seg_numbers = np.unique(segments_map)
    if seg_numbers[0] == 0:
        seg_numbers = seg_numbers[1:]
    n_seg = len(seg_numbers)
    n_points = n_frames * n_seg
    if part_type == 'segments':
        loss_map = np.zeros(n_seg)
    elif part_type == 'frames':
        loss_map = np.zeros(n_frames)
    else: # part_type == 'both':
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
        test_x = test_set.all_x
        all_indexes = get_all_missing_indexes(n_points, n_seg, n_frames, part_type)
        for iter, indexes in enumerate(all_indexes):
            if zero_all:
                new_test_x = torch.zeros_like(test_x)
                new_test_x[:, indexes] = test_x[:, indexes]
            else:
                new_test_x = test_x.clone()
                new_test_x[:, indexes] = 0

            curr_acc, curr_loss = test_model(model, new_test_x, test_set.all_y)
            if value_type == 'loss':
                loss_map[iter] += curr_loss
            else:
                loss_map[iter] += curr_acc

    return loss_map


def train_validation_and_test(model, datasets, parameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.double().to(device)

    train_dataset, valid_dataset, test_dataset = datasets
    loss_fn, n_epochs, optimizer, scheduler = parameters

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    test_losses = []
    test_accuracies = []
    train_loader = data_utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_y_int = valid_dataset.all_y.int()
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

        predictions = torch.round(outputs).int().view(outputs.numel())
        accuracy = (predictions == valid_y_int).sum().item() / len(valid_y_int)
        valid_accuracies.append(accuracy)

        #
        outputs = model(test_dataset.all_x)
        outputs = outputs.view(outputs.numel())
        loss = loss_fn(outputs, test_dataset.all_y).item()
        test_losses.append(loss)

        predictions = torch.round(outputs).int().view(outputs.numel())
        accuracy = (predictions == test_y_int).sum().item() / len(test_y_int)
        test_accuracies.append(accuracy)

        if valid_losses[-1] < 0.01:
            return model, train_losses, valid_losses, valid_accuracies
        if e > 18 and np.mean(valid_losses[-5:]) > np.mean(valid_losses[-10:-5]):
            print('finished at epoch {}'.format(e))
            return model, train_losses, valid_losses, valid_accuracies
        if e % 2 == 0:
            print("{}. train loss: {}   valid_loss: {}  valid-acc:{}".format(
                e, train_losses[-1], valid_losses[-1], valid_accuracies[-1]))

    return model, train_losses, valid_losses, valid_accuracies