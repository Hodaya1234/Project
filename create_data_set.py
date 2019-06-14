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


def get_data(seg_v, seg_h, n_new_train, normalize=False):
    print('creating data sets')
    seg_v, seg_h = np.transpose(seg_v, [2, 0, 1]), np.transpose(seg_h, [2, 0, 1])
    num_v = seg_v.shape[0]
    num_h = seg_h.shape[0]
    indices_v = np.arange(num_v)
    indices_h = np.arange(num_h)
    seg_v = seg_v.reshape([num_v, -1])
    seg_h = seg_h.reshape([num_h, -1])
    if not normalize:
        seg_v = seg_v - 1
        seg_h = seg_h - 1
    aug_all_v = concat_augment_orig(seg_v, n_new_train)
    aug_all_h = concat_augment_orig(seg_h, n_new_train)

    tx = []
    ty = []
    vx = []
    vy = []
    for i in range(num_v):
        print(i)
        vx.append(seg_v[i,:])
        vy.append(1)

        tv = concat_augment_orig(seg_v[np.delete(indices_v, i), :], n_new_train)
        tx.append(np.concatenate([tv, aug_all_h], axis=0))
        ty.append(np.concatenate([np.ones([len(tv),]), np.zeros([len(aug_all_h),])]))

    for i in range(num_h):
        print(i)
        vx.append(seg_h[i, :])
        vy.append(0)

        th = concat_augment_orig(seg_h[np.delete(indices_h, i), :], n_new_train)
        tx.append(np.concatenate([aug_all_v, th], axis=0))
        ty.append(np.concatenate([np.ones([len(aug_all_v), ]), np.zeros([len(th), ])]))

    tx = np.asarray(tx)
    ty = np.asarray(ty)
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    if not normalize:
        tx, ty, vx, vy = tx + 1, ty + 1, vx + 1, vy + 1
    return tx, ty, vx, vy



