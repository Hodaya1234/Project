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

plt.ion()


def load_data():
    raw = sio.loadmat('C:\\Users\\H\\Desktop\\project\\data\\clean.mat')
    vert = raw['clean_vert']
    horiz = raw['clean_horiz']
    tran_v = np.transpose(vert, [2, 0, 1])
    tran_h = np.transpose(horiz, [2, 0, 1])
    v_data = tran_v.reshape([-1,100,100,256])
    h_data = tran_h.reshape([-1,100,100,256])
    comb_data = np.concatenate([v_data, h_data], axis=0)
    np.save('data', comb_data)


def transform_data(raw_data=np.load('data.npy')):
    frame_first = 27
    frame_last = 67
    frames = list(range(frame_first, frame_last))
    raw_data = raw_data[:,:,:,frames].transpose([0,3,1,2])
    pad = int((224 - 100) / 2)
    data = np.zeros([14, (frame_last - frame_first), 3, 224, 224])
    for i in range(3):
        data[:, :, 0, pad:224 - pad, pad:224 - pad] = raw_data
    np.save('tran_data', data)


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


def create_data_set(x):
    n = x.shape[0]
    y = np.concatenate([np.ones(n / 2), np.zeros(n / 2)])
    return DataSet(torch.from_numpy(x), torch.from_numpy(y))


raw = np.load('norm_data.npy')
val_indices = [0,7]
train_indices = [i for i in range(14) if i not in val_indices]
val = raw[val_indices,:,:,:,:]
train = raw[train_indices,:,:,:,:]

frames_val_x = train.transpose([-1, 3, 100, 100])
frames_train_x = train.transpose([-1, 3, 100, 100])

train_dataset = create_data_set(frames_train_x)