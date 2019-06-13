# Convert some of the data set to mat files in order to view them in matlab
import numpy as np
import segment
import scipy.io as sio
import train_model
import matplotlib.pyplot as plt
import create_data_set
import scipy.io as sio
import segment
from sklearn import svm
import visualize_res
import augment
import sys


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


def svm_classify():

    # read the original data
    data = sio.loadmat('Data/02.12/b/clean.mat')
    clean_v = data['clean_vert']
    clean_h = data['clean_horiz']
    n_v = clean_v.shape[2]
    n_h = clean_h.shape[2]
    min_num = min(n_v, n_h) - 2
    frames = list(range(27,68))
    n_frames = len(frames)
    down_v = np.transpose(clean_v[:,frames,:], [2, 0, 1])
    down_h = np.transpose(clean_h[:,frames,:], [2, 0, 1])


    non_zero = down_v[0,:,0] != 0

    v = down_v[:,non_zero,:]
    h = down_h[:,non_zero,:]


    train_y = np.concatenate((np.ones(min_num), np.zeros(min_num)))
    test_y = np.asarray([1,1, 0,0])
    frame_accuracies = np.zeros([len(frames),1])
    for iv in range(n_v - 1):
        for ih in range(n_h - 1):
            print('{}, {}'.format(iv, ih))
            v_indexes = np.random.choice(np.delete(np.arange(n_v), [iv, iv+1]), min_num)
            h_indexes = np.random.choice(np.delete(np.arange(n_h), [ih, ih+1]), min_num)
            current_v = v[v_indexes,:,:]
            current_h = h[h_indexes, :, :]
            current_all = np.concatenate([current_v, current_h], axis=0)
            m = np.mean(current_all, axis=0)
            s = np.std(current_all, axis=0)
            current_all = np.divide(np.subtract(current_all, m), s)
            train_x = []
            for f in range(n_frames):
                for i in range(len(current_all)):
                    train_x.append(current_all[i,:,f])
            train_x = np.asarray(train_x)
            train_y = np.concatenate([np.ones(n_frames*min_num), np.zeros(n_frames*min_num)], axis=0)

            clf = svm.SVC(gamma='auto')
            clf.fit(train_x, train_y)

            test_v = np.divide(np.subtract(v[iv:iv+1,:,:], m), s)
            test_h = np.divide(np.subtract(h[ih:ih+1,:,:], m), s)
            for f in range(len(frames)):
                test_x = np.concatenate([test_v[:,f], test_h[:,f]], axis=0)
                pred = clf.predict(test_x)
                acc = np.mean(pred == test_y)
                frame_accuracies[f] += acc
    frame_accuracies = frame_accuracies / (n_v * n_h)
    np.save('frame_accuracies', frame_accuracies)
    plt.figure()
    plt.title('Cross validation Accuracy for Frames')
    plt.plot([i+1 for i in frames],frame_accuracies)
    plt.show()
    # loop over train and test and run svm

    # plot accuracy


def temp_svm(data_sets, mask):
    train_sets_x, train_y, validation_sets_x, valid_y, test_sets_x, test_y = data_sets
    n_data_sets = len(train_sets_x)
    mask_nubmers = np.unique(mask)
    n_seg = len(mask_nubmers) - 1 if mask_nubmers[0] == 0 else len(mask_nubmers)
    n_frames = int(train_sets_x.shape[2] / n_seg)
    valid_accuracies = []
    frames_loss_maps = np.zeros([n_data_sets, n_frames])
    seg_loss_maps = np.zeros([n_data_sets, n_seg])
    all_indexes = np.asarray(list(range(n_seg*n_frames))).reshape([n_seg, n_frames])
    for idx, (one_train, one_test) in enumerate(zip(train_sets_x, test_sets_x)):
        m = np.mean(one_train, axis=0)
        s = np.std(one_train, axis=0)
        one_train = (one_train - m) / s
        one_test = (one_test - m) / s
        clf = svm.SVC(C=1, gamma='auto')
        clf.fit(one_train, train_y)
        prediction = clf.predict(one_test)
        real_validation_acc = np.mean(prediction == test_y)
        valid_accuracies.append(real_validation_acc)
        # for f in range(n_frames):
        #     new_test = np.zeros_like(one_test)
        #     indices = all_indexes[:,f].ravel()
        #     new_test[:,indices] = one_test[:,indices]
        #     prediction = clf.predict(new_test)
        #     frames_loss_maps[idx,f] += np.mean(prediction == test_y)
        # for s in range(n_seg):
        #     new_test = np.zeros_like(one_test)
        #     indices = all_indexes[s,:].ravel()
        #     new_test[:,indices] = one_test[:,indices]
        #     prediction = clf.predict(new_test)
        #     seg_loss_maps[idx,s] += np.mean(prediction == test_y)
    frame_loss = np.mean(frames_loss_maps, axis=0)
    seg_loss = np.mean(seg_loss_maps, axis=0)
    image = segment.recreate_image(mask, seg_loss)
    print(np.mean(valid_accuracies))

    # plt.plot(np.arange(41)*10, frame_loss)
    # plt.xlabel('Time from target onset (ms)')
    # plt.ylabel('Accuracy')
    # plt.show()
    # # sio.savemat('b_svm_seg_acc', {'b':image})
    # visualize_res.plot_spatial(image, title='accuracy for present segment')




def leave_one_out_sets(seg_v, seg_h, n_train=50, normalize=False):
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


# seg_path = sys.argv[1]
# save_path = sys.argv[2]
# orig_data = np.load(seg_path)
# v = orig_data['seg_v']
# h = orig_data['seg_h']
# all_augmented_v, all_augmented_h = leave_one_out_sets(v, h, 50)
# np.savez(save_path, v=all_augmented_v, h=all_augmented_h)

# def create_sets()


segmented = np.load('Data/Segmented/b_sq.npz')
augmented = np.load('Data/Res/lou_b.npz')
seg_v = segmented['seg_v']
seg_h = segmented['seg_h']
n_v = seg_v.shape[2]
n_h = seg_h.shape[2]
aug_v = augmented['v']
aug_h = augmented['h']

seg_v = seg_v.reshape([-1, n_v]).T
seg_h = seg_h.reshape([-1, n_h]).T

v_ind = np.arange(n_v)
h_ind = np.arange(n_h)
train_v = np.concatenate([aug_v, seg_v], axis=0)
train_h = np.concatenate([aug_h, seg_h])

train_sets_x = []
validation_sets_x = []
train_sets_y = []
validation_sets_y = []
for i in range(n_v):
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
np.savez('Data/Sets/lou_b_no_norm', tx=train_sets_x, ty = train_sets_y, vx=validation_sets_x, vy=validation_sets_y)