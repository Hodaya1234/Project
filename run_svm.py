import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm
import segment
import visualize_res

def svm_classify():
    # read the original data
    data = sio.loadmat('C:\\Users\\H\\Desktop\\project\\data\\clean\\b.mat')
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


def temp_svm(data_sets, mask, frames):
    train_sets_x, train_y, test_sets_x, test_y = data_sets
    n_data_sets = len(train_sets_x)
    mask_nubmers = np.unique(mask)
    n_seg = len(mask_nubmers) - 1 if mask_nubmers[0] == 0 else len(mask_nubmers)
    n_frames = int(train_sets_x.shape[2] / n_seg)
    valid_accuracies = []
    frames_loss_maps = np.zeros([n_data_sets, n_frames])
    seg_loss_maps = np.zeros([n_data_sets, n_seg])
    all_indexes = np.asarray(list(range(n_seg*n_frames))).reshape([n_seg, n_frames])
    for idx, (tx, ty, vx, vy) in enumerate(zip(train_sets_x,train_y, test_sets_x, test_y)):
        m = np.mean(tx, axis=0)
        s = np.std(tx, axis=0)
        tx = (tx - m) / s
        vx = (vx - m) / s
        vx = vx.reshape([1,-1])
        clf = svm.SVC()
        clf.fit(tx, ty)
        prediction = clf.predict(vx)
        real_validation_acc = np.mean(prediction == vy)
        valid_accuracies.append(real_validation_acc)
        for f in range(n_frames):
            new_test = np.zeros_like(vx)
            indices = all_indexes[:,f].ravel()
            new_test[:,indices] = vx[:,indices]
            prediction = clf.predict(new_test)
            frames_loss_maps[idx,f] += np.mean(prediction == vy)
        for s in range(n_seg):
            new_test = np.zeros_like(vx)
            indices = all_indexes[s,:].ravel()
            new_test[:,indices] = vx[:,indices]
            prediction = clf.predict(new_test)
            seg_loss_maps[idx,s] += np.mean(prediction == vy)
    frame_loss = np.mean(frames_loss_maps, axis=0)
    seg_loss = np.mean(seg_loss_maps, axis=0)
    image = segment.recreate_image(mask, seg_loss)
    print(np.mean(valid_accuracies))

    # TODO: check where is frame zero on frames
    plt.plot(np.arange(len(frames))*10, frame_loss)
    plt.xlabel('Time from target onset (ms)')
    plt.ylabel('Accuracy')
    plt.show()
    # sio.savemat('b_svm_seg_acc', {'b':image})
    visualize_res.plot_spatial(image, title='accuracy for present segment')

