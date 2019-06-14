"""
Can read several types of files here:
1. A mat file of the raw data
2. An npz file containing numpy arrays
    a. The raw data in numpy arrays (filename - 'raw_data.npz', containing 'horiz', 'vert')
    b. The segmented data (filename - 'segment_data.npz', containing 'segments', 'segmented_vert', 'segmented_horiz')
    c. The data set for the training (filename - 'data_set.npz')
"""

from scipy import io as sio
import numpy as np
import torch


def read_from_file(folder_name, flag):
    """
    Read input data, either as raw in the beginning, or as variables from the middle of the process.
    This can save double calculation when it is needed to redo only a part of the program.
    :param folder_name: A relative path to the file of the data.
    :param flag: Denotes which part of the script the following data is relevant to.
    types:
    a. raw:
        The "raw" data of the experiments. It is not actually raw because this is the data after
        dividing each trial by it's baseline, dividing by the mean blank condition, and cleaning the irrelevant
        data points (outside the chamber or on top of blood vessels).
        For this program this data after pre-processing is considered raw.
    b. seg:
        The data after dividing it to segments, and the mask of segment.
    c. set:
        The data set which is comprised of training, validation and test sets. The majority of the examples
        are augmented, and there are some from the original data.
    d. res:
        The results of the model.
    :return: numpy ndarrays with the extracted data from the file.
    """
    if flag == 'raw':
        file = sio.loadmat(folder_name)
        return file['clean_vert'], file['clean_horiz']
    if flag == 'mask':
        return np.load(folder_name)
    if flag == 'seg':
        with np.load(folder_name) as file:
            data = file['seg_v'], file['seg_h']
        return data
    if flag == 'set':
        with np.load(folder_name) as file:
            data = [file[i] for i in file]
        return data
    if flag == 'los':
        with np.load(folder_name) as file:
            data = file['train_losses'], file['validation_losses'], file['test_losses'], file['test_accuracies'], file['n_data_sets']
        return data
    if flag == 'net':
        return torch.load(folder_name)
    if flag == 'vis':
        return np.load(folder_name)


def save_to(data, folder_name, flag):
    """
    Save the data in .npy or .npz format in order to save calculations when only a part of the process need to be
    redone.
    :param data: An array or list of arrays to be saved.
    :param folder_name: A relative path to save the data to.
    :param flag: Indicates what's the type of data that needs to be saved.
    types:
    a. raw:
        The "raw" data of the experiments, after tranferring it from .mat to .npz.
        For this program this data after pre-processing is considered raw although it went through several stages of
        pre processing.
    b. seg:
        The data after dividing it to segments, and the mask of segment.
    c. set:
        The data set which is comprised of training, validation and test sets. The majority of the examples
        are augmented, and there are some from the original data.
    d. res:
        The results of the model.
    :return:
    """
    # if flag == 'raw':
    #     np.savez(folder_name, clean_horiz=data['clean_horiz'], clean_vert=data['clean_vert'],
    #              clean_blank=data['clean_blank'])
    if flag == 'mask':
        np.save(folder_name, data)

    if flag == 'seg':
        seg_v, seg_h = data
        np.savez(folder_name, seg_v=seg_v, seg_h=seg_h)
    elif flag == 'set':
        train_x, train_y, test_x, test_y = data
        np.savez(
            folder_name, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    elif flag == 'los':
        train_losses, validation_losses, test_losses, test_accuracies, n_data_sets = data
        np.savez(folder_name, train_losses=train_losses, validation_losses=validation_losses, test_losses=test_losses, test_accuracies=test_accuracies, n_data_sets=n_data_sets)
        return
    elif flag == 'net':
        net = data
        torch.save(net, folder_name)
    elif flag == 'vis':
        np.save(arr=data, file=folder_name)
