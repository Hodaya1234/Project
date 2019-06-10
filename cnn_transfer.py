from __future__ import print_function, division
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch
from torchvision import datasets, models, transforms
import time
import os
import copy
from data_set import DataSet
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
import sys
from sklearn import linear_model


use_cuda = torch.cuda.is_available()

def load_data(raw_path, frames):
    raw = sio.loadmat(raw_path)
    vert = raw['clean_vert']
    horiz = raw['clean_horiz']
    tran_v = np.transpose(vert[:,frames,:], [2, 0, 1])
    tran_h = np.transpose(horiz[:,frames,:], [2, 0, 1])
    v_data = tran_v.reshape([-1,100,100,len(frames)])
    h_data = tran_h.reshape([-1,100,100,len(frames)])
    x = np.concatenate([v_data, h_data], axis=0)
    y = np.concatenate([np.ones(len(v_data)), np.zeros(len(h_data))])
    return x, y


def normalize_data(data, zscore=False):
    data[data == 0] = np.nan
    if zscore:
        m = np.nanmean(data)
        s = np.nanstd(data)
        data[np.isnan(data)] = 0
        data = np.divide(np.subtract(data, m), s)
    else:
        min_point = np.nanmin(data)
        max_point = np.nanmax(data)
        data = (data - min_point) / (max_point - min_point)
        data[np.isnan(data)] = 0
    np.save('norm_data', data)
    return data

"""
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
"""

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fine_tune_cnn():
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = Identity()
    # model_conv = model_conv.to(device)
    criterion = nn.BCELoss()
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


def transform(x, y, n_frames):
    pad = int((224 - 100) / 2)
    data = np.zeros([x.shape[0], 3, 224, 224, n_frames])
    for i in range(3):
        data[:, i, pad:224 - pad, pad:224 - pad, :] = x
    return data


def to_torchvision_norm(data):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for i in range(3):
        channel = data[i,:,:]
        m = np.mean(channel)
        s = np.std(channel)
        channel = (channel - m) / s
        channel *= stds[i]
        channel += means[i]
        data[i,:,:] = channel
    return data


def data_through_cnn(model, x_data, n_frames, single_frame_length):
    new_x = np.empty([len(x_data), n_frames*single_frame_length])
    for idx, x in enumerate(x_data):
        all_frame_vec = np.empty([n_frames, single_frame_length])
        for i in range(n_frames):
            after_torch_norm = to_torchvision_norm(x[:,:,:,i])
            input = torch.from_numpy(after_torch_norm[np.newaxis, :, :, :])
            if use_cuda:
                input = input.cuda()
            vec = model(input)
            if use_cuda:
                all_frame_vec[i, :] = vec.cpu()
            else:
                all_frame_vec[i, :] = vec
        new_x[idx, :] = all_frame_vec.flatten()
    return new_x


def create_cnn_data(raw_data_path):
    single_length = 512
    model_conv = models.resnet18(pretrained=True).double()
    for param in model_conv.parameters():
        param.requires_grad = False
    model_conv.fc = Identity()
    if use_cuda:
        model_conv = model_conv.cuda()
    frames = np.arange(27,68)
    all_x, all_y = load_data(raw_data_path, frames)
    n_data = len(all_y)
    train_x_sets = []
    val_x_sets = []
    train_y_sets = []
    val_y_sets = []
    for val_index in range(n_data):
        print(val_index)
        train_indices = np.delete(np.arange(n_data), val_index)
        train_y = all_y[train_indices]
        val_y = all_y[val_index]

        all_x[all_x == 0] = np.nan

        val_x = all_x[val_index, :, :, :]
        val_x = val_x[np.newaxis, :, :, :]
        train_x = all_x[train_indices, :, :, :]

        m = np.nanmean(train_x, axis=0)
        s = np.nanstd(train_x, axis=0)
        train_x = np.divide(np.subtract(train_x, m), s)
        val_x = np.divide(np.subtract(val_x, m), s)

        train_x = np.nan_to_num(train_x)
        val_x = np.nan_to_num(val_x)

        train_x = transform(train_x, train_y, len(frames))
        val_x = transform(val_x, val_y, len(frames))

        new_train_x = data_through_cnn(model_conv, train_x, len(frames), single_length)
        new_val_x = data_through_cnn(model_conv, val_x, len(frames), single_length)

        train_x_sets.append(new_train_x)
        train_y_sets.append(train_y)
        val_x_sets.append(new_val_x)
        val_y_sets.append(val_y)
    return np.asarray(train_x_sets), np.asarray(train_y_sets), np.asarray(val_x_sets), np.asarray(val_y_sets)


# def combine_days(file_paths):
    # open files from all days. go over all sets in a loop. for each set in the loop: normalize all the other days separately.
    # go over all the examples in that


def plot_pca(x, y):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = PCA(n_components=3)
    pca.fit(x)
    X = pca.transform(x)

    # Reorder the labels to have colors matching the cluster results
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')

    plt.show()


def main():
    args = sys.argv
    train_x, train_y, val_x, val_y = create_cnn_data(args[1])
    np.savez(args[2], train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)


def test_with_svm():
    new_data = np.load('new_data_b.npz')
    train_x = new_data['train_x']
    train_y = new_data['train_y']
    val_x = new_data['val_x']
    val_y = new_data['val_y']
    corrects = 0
    for i in range(len(train_x)):
        clf = svm.LinearSVC(class_weight='balanced')
        x = train_x[i,:,:]
        y = train_y[i,:]
        x_val = val_x[i,:,:]
        m = np.mean(x, axis=0)
        s = np.std(x, axis=0)
        x = np.divide(np.subtract(x, m), s)
        x_val = (x_val - m) / s
        clf.fit(x, y)
        y_hat = clf.predict(x_val)
        corrects += (y_hat == val_y[i])
    print(corrects / len(train_x))


def visualize_coef():
    coef = np.load('b0212_coef.npy')
    n_frames = int(coef.shape[1] / 512)
    maxi = np.zeros([n_frames, ])
    mini = np.zeros([n_frames, ])
    mean_abs = np.zeros([n_frames, ])
    for i in range(n_frames):
        frame_coef = coef[0, i * 512:(i + 1) * 512]
        # maxi[i] = np.mean(frame_coef[frame_coef > 0])
        # mini[i] = np.mean(frame_coef[frame_coef < 0])
        mean_abs[i] = np.mean(np.abs(frame_coef))
    plt.plot(np.arange(27,68), mean_abs)
    plt.title('Mean Absolute Value of SVM Coefficients per Frame Output from CNN')
    plt.show()


test_with_svm()