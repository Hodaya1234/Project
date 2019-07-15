# Convert some of the data set to mat files in order to view them in matlab
import numpy as np
import segment
import scipy.io as sio
import run_nn
import matplotlib.pyplot as plt
import create_data_set
import scipy.io as sio
import segment
from sklearn import svm
import visualize_res
import augment
import sys
import dense_net


def save_segmented(mask, v, h, name):
    v_image = segment.recreate_image(mask, v)
    h_image = segment.recreate_image(mask, h)
    sio.savemat(name, {'v': v_image, 'h': h_image})


def save_set(set_x, mask, name):
    n_seg = len(np.unique(mask)) - 1
    set_x = set_x.reshape([set_x.shape[0], n_seg, -1])
    x1 = segment.recreate_image(mask, set_x[0,:,:])
    x2 = segment.recreate_image(mask, set_x[1,:,:])
    x3 = segment.recreate_image(mask, set_x[-1,:,:])
    x4 = segment.recreate_image(mask, set_x[-2, :, :])
    sio.savemat(name, {'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4})



def leave_one_out_augmentation(seg_v, seg_h, n_train=50, normalize=False):
    seg_v, seg_h = np.transpose(seg_v, [2, 0, 1]), np.transpose(seg_h, [2, 0, 1])
    num_v = seg_v.shape[0]
    num_h = seg_h.shape[0]
    seg_v = seg_v.reshape([num_v, -1])
    seg_h = seg_h.reshape([num_h, -1])
    if not normalize:
        seg_v = seg_v - 1
        seg_h = seg_h - 1
    v_indices = np.arange(num_v)
    h_indices = np.arange(num_h)
    all_augmented_v = []
    all_augmented_h = []
    for i in range(num_v):
        print(i)
        train_v_original = seg_v[np.delete(v_indices, i), :]
        params = augment.get_parameters(train_v_original)
        augmented_v = augment.get_new_data(params, n_train)
        all_augmented_v.append(augmented_v)
    all_augmented_v = np.asarray(all_augmented_v)
    for i in range(num_h):
        print(i)
        train_h_original = seg_h[np.delete(h_indices, i), :]
        params = augment.get_parameters(train_h_original)
        augmented_h = augment.get_new_data(params, n_train)
        all_augmented_h.append(augmented_h)
    all_augmented_h = np.asarray(all_augmented_h)
    if not normalize:
        all_augmented_v = all_augmented_v + 1
        all_augmented_h = all_augmented_h + 1
    return all_augmented_v, all_augmented_h


def create_lou_sets():
    segmented = np.load('Data/Segmented/c_sq.npz')
    augmented = np.load('Data/Res/lou_c.npz')
    seg_v = segmented['seg_v']
    seg_h = segmented['seg_h']
    n_v = seg_v.shape[2]
    n_h = seg_h.shape[2]
    aug_v = augmented['v']
    aug_h = augmented['h']

    seg_v = seg_v.reshape([-1, n_v]).T
    seg_h = seg_h.reshape([-1, n_h]).T

    params = augment.get_parameters(seg_v)
    augmented_all_v = augment.get_new_data(params, 2)
    params = augment.get_parameters(seg_h)
    augmented_all_h = augment.get_new_data(params, 2)

    v_ind = np.arange(n_v)
    h_ind = np.arange(n_h)
    train_v = np.concatenate([augmented_all_v, seg_v], axis=0)
    train_h = np.concatenate([augmented_all_h, seg_h], axis=0)

    train_sets_x = []
    validation_sets_x = []
    train_sets_y = []
    validation_sets_y = []
    for i in range(n_v):
        print(i)
        validation_x = seg_v[i, :]
        validation_y = 1
        train_v_i = np.concatenate([aug_v[i,:,:], seg_v[np.delete(v_ind, i), :]], axis=0)
        train_x = np.concatenate([train_v_i, train_h])
        train_y = np.concatenate([np.ones([len(train_v_i),]), np.zeros([len(train_h),])])
        train_sets_x.append(train_x)
        train_sets_y.append(train_y)
        validation_sets_x.append(validation_x)
        validation_sets_y.append(validation_y)

    for i in range(n_h):
        print(i)
        validation_x = seg_h[i, :]
        validation_y = 0
        train_h_i = np.concatenate([aug_h[i,:,:], seg_h[np.delete(h_ind, i), :]], axis=0)
        train_x = np.concatenate([train_v, train_h_i])
        train_y = np.concatenate([np.ones([len(train_v),]), np.zeros([len(train_h_i),])])
        train_sets_x.append(train_x)
        train_sets_y.append(train_y)
        validation_sets_x.append(validation_x)
        validation_sets_y.append(validation_y)

    train_sets_x = np.asarray(train_sets_x)
    validation_sets_x = np.asarray(validation_sets_x)
    train_sets_y = np.asarray(train_sets_y)
    validation_sets_y = np.asarray(validation_sets_y)
    np.savez('Data/Sets/lou_c_no_norm', tx=train_sets_x, ty=train_sets_y, vx=validation_sets_x, vy=validation_sets_y)


new_set = np.load('Data/Sets/lou_b_no_norm.npz')
sets = [new_set[i] for i in new_set]
tx, ty, vx, vy = sets
mask = np.load('Data/Masks/a_sq.npy')
temp_svm(sets, mask)
