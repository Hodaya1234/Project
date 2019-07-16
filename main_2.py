# need clean data from a certain session
# leave one out - each training data - train on single frames and classify single frames.
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import hinge_loss
import segment

def svm_classify(all_x, all_y, n_pixels):
    # read the original data
    accuracies = np.zeros([len(all_y), len(frames)])
    losses = np.zeros([len(all_y), len(frames)])
    coef = np.zeros([len(all_y), len(frames), n_pixels])

    for idx, validation_y in enumerate(all_y):
        train_indx = np.delete(np.arange(len(all_y)), idx)
        for f in range(n_frames):
            train_x = all_x[train_indx, f, :]
            validation_x = all_x[idx, f, :].reshape([1, -1])
            m = np.mean(train_x, axis=0)
            s = np.std(train_x, axis=0)
            train_x = (train_x - m) / s
            validation_x = (validation_x - m) / s

            train_y = all_y[train_indx]
            clf = svm.SVC(kernel='linear', class_weight='balanced')
            clf.fit(train_x, train_y)

            coef[idx, f, :] = clf.coef_

            pred = clf.predict(validation_x)
            acc = pred == validation_y
            accuracies[idx, f] = acc
            est = clf.decision_function(validation_x)
            loss = hinge_loss([validation_y], est)
            losses[idx, f] = loss
    return accuracies, losses, coef


def raw_classify(paths):
    accuracies = []
    losses = []
    coeffs_frames = []
    for path in paths:
        data = sio.loadmat(path)
        clean_v = data['clean_vert']
        clean_h = data['clean_horiz']
        n_v = clean_v.shape[2]
        n_h = clean_h.shape[2]

        down_v = np.transpose(clean_v[:, frames, :], [2, 1, 0])
        down_h = np.transpose(clean_h[:, frames, :], [2, 1, 0])

        non_zero = down_v[0, 0, :] != 0

        v = down_v[:, :, non_zero]
        h = down_h[:, :, non_zero]

        all_x = np.concatenate([v, h], 0)
        all_y = np.concatenate([np.ones([n_v]), np.zeros([n_h])])
        sess_acc, sess_loss, sess_coef = svm_classify(all_x, all_y, np.count_nonzero(non_zero))
        accuracies.append(np.mean(sess_acc, 0))
        losses.append(np.mean(sess_loss, 0))
        coefficients = np.mean(np.mean(sess_coef, 0), 0)
        coeff_frame = np.zeros([10000])
        coeff_frame[non_zero] = coefficients
        coeffs_frames.append(coeff_frame)

    plt.figure()
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(np.reshape(coeffs_frames[i], [100, 100]))
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Accuracy')
    for i in range(3):
        plt.plot([i * 10 - 270 for i in frames], accuracies[i])
    plt.legend(labels)
    plt.subplot(1, 2, 2)
    plt.title('Loss')
    for i in range(3):
        plt.plot([i * 10 - 270 for i in frames], losses[i])
    plt.legend(labels)
    plt.show()


def segment_classify(paths):
    accuracies = []
    losses = []
    coeffs_frames = []
    for path in paths:
        data = sio.loadmat(path)
        clean_v = data['clean_vert']
        clean_h = data['clean_horiz']
        n_v = clean_v.shape[2]
        n_h = clean_h.shape[2]

        # down_v = np.transpose(clean_v[:, frames, :], [2, 1, 0])
        # down_h = np.transpose(clean_h[:, frames, :], [2, 1, 0])
        seg_v = segment.divide_data_to_segments(mask, clean_v[:, frames, :])
        seg_h = segment.divide_data_to_segments(mask, clean_h[:, frames, :])
        seg_v = np.transpose(seg_v, [2, 1, 0])
        seg_h = np.transpose(seg_h, [2, 1, 0])

        all_x = np.concatenate([seg_v, seg_h], 0)
        all_y = np.concatenate([np.ones([n_v]), np.zeros([n_h])])
        sess_acc, sess_loss, sess_coef = svm_classify(all_x, all_y, n_seg)
        accuracies.append(np.mean(sess_acc, 0))
        losses.append(np.mean(sess_loss, 0))
        coeff = segment.recreate_image(mask, np.mean(np.mean(sess_coef, 0), 0))
        coeffs_frames.append(coeff)

    plt.figure()
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(np.reshape(coeffs_frames[i], [100, 100]))
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Accuracy')
    for i in range(3):
        plt.plot([i * 10 - 270 for i in frames], accuracies[i])
    plt.legend(labels)
    plt.subplot(1, 2, 2)
    plt.title('Loss')
    for i in range(3):
        plt.plot([i * 10 - 270 for i in frames], losses[i])
    plt.legend(labels)
    plt.show()

frames = np.arange(35, 55)
n_frames = len(frames)
raw_paths = ['Data/02.12/a/clean.mat', 'Data/02.12/b/clean.mat', 'Data/02.12/c/clean.mat']
mask = np.load('Data/Masks/a_sq.npy')
labels = ['a', 'b', 'c']
n_seg = len(np.unique(mask)) - 1
raw_classify(raw_paths)
