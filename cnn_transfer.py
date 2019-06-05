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
    data = np.zeros([len(y), 3, 224, 224, n_frames])
    for i in range(3):
        data[:, i, pad:224 - pad, pad:224 - pad, :] = x
    return data


def data_through_cnn(model, x_data, n_frames, single_frame_length):
    new_x = np.empty([len(x_data), n_frames*single_frame_length])
    for idx, x in enumerate(x_data):
        all_frame_vec = np.empty([n_frames, single_frame_length])
        for i in range(n_frames):
            vec = model(torch.from_numpy(x[np.newaxis, :, :, :, i]))
            all_frame_vec[i, :] = vec
        new_x[idx, :] = all_frame_vec.flatten()
    return new_x


frames = np.arange(27,68)
all_x, all_y = load_data('Data/02.12/a/clean.mat', frames)
all_x[all_x == 0] = np.nan

val_indices = [0,7]
train_indices = [i for i in range(14) if i not in val_indices]

train_y = all_y[train_indices]
val_y = all_y[val_indices]

val_x = all_x[val_indices,:,:,:]
train_x = all_x[train_indices,:,:,:]
m = np.nanmean(train_x, axis=0)
s = np.nanstd(train_x, axis=0)
train_x = np.divide(np.subtract(train_x, m), s)
val_x = np.divide(np.subtract(val_x, m), s)

train_x = np.nan_to_num(train_x)
val_x = np.nan_to_num(val_x)

train_x = transform(train_x, train_y, len(frames))
val_x = transform(val_x, val_y, len(frames))

# go over all examples: for each frame get a vector output from the cnn, concatenate them. insert to NN
single_length = 512
model_conv = models.resnet18(pretrained=True).double()
for param in model_conv.parameters():
    param.requires_grad = False
model_conv.fc = Identity()

new_train_x = data_through_cnn(model_conv, train_x, len(frames), single_length)
new_val_x = data_through_cnn(model_conv, val_x, len(frames), single_length)

print('done')
