import numpy as np
import augment
import torch
from data_set import DataSet
import matplotlib.pyplot as plt


def concat_augment_orig(orig, n_aug):
    params = augment.get_parameters(orig)
    aug_cond1 = augment.get_new_data(params, n_aug)
    train = np.concatenate([orig, aug_cond1], axis=0)
    return train


def transform_seg(segmented):
    new_data = np.transpose(segmented, [2, 0, 1])
    num = new_data.shape[0]
    new_data = new_data.reshape([num, -1])
    return new_data, num


def get_data_no_aug(seg_v, seg_h):
    seg_v, num_v = transform_seg(seg_v)
    seg_h, num_h = transform_seg(seg_h)
    indices_v = np.arange(num_v)
    indices_h = np.arange(num_h)

    tx = []
    ty = []
    vx = []
    vy = []


def get_data(seg_v, seg_h, n_new_train, normalize=False):
    print('creating data sets')
    seg_v, num_v = transform_seg(seg_v)
    seg_h, num_h = transform_seg(seg_h)
    indices_v = np.arange(num_v)
    indices_h = np.arange(num_h)
    n_new_train_h = n_new_train_v = n_new_train
    if num_v > num_h:
        n_new_train_h += num_v - num_h
    elif num_v < num_h:
        n_new_train_v += num_h - num_v

    if not normalize:
        seg_v = seg_v - 1
        seg_h = seg_h - 1
    aug_all_v = concat_augment_orig(seg_v, n_new_train_v)
    aug_all_h = concat_augment_orig(seg_h, n_new_train_h)

    tx = []
    ty = []
    vx = []
    vy = []
    for i in range(num_v):
        print(i)
        vx.append(seg_v[i,:])
        vy.append(1)

        tv = concat_augment_orig(seg_v[np.delete(indices_v, i), :], n_new_train_v)
        tx.append(np.concatenate([tv, aug_all_h], axis=0))
        ty.append(np.concatenate([np.ones([len(tv),]), np.zeros([len(aug_all_h),])]))

    for i in range(num_h):
        print(i)
        vx.append(seg_h[i, :])
        vy.append(0)

        th = concat_augment_orig(seg_h[np.delete(indices_h, i), :], n_new_train_h)
        tx.append(np.concatenate([aug_all_v, th], axis=0))
        ty.append(np.concatenate([np.ones([len(aug_all_v), ]), np.zeros([len(th), ])]))

    tx = np.asarray(tx)
    ty = np.asarray(ty)
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    if not normalize:
        tx, vx = tx + 1, vx + 1
    return tx, ty, vx, vy



