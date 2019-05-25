import data_io
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from dense_net import DenseNet
import train_model
from data_set import DataSet

def normalize(data, m=None, s=None):
    if m is None:
        m = np.mean(data, axis=0)
    if s is None:
        s = np.std(data, axis=0)
    return np.divide(np.subtract(data, m), s)


def svm_train_test(train_x, train_y, orig_x, orig_y, new_x, new_y, frames):
    n_frames = len(frames)
    n_sets = len(train_x)
    orig_acc = np.zeros([n_sets, n_frames])
    new_acc = np.zeros([n_sets, n_frames])
    # train_y = np.concatenate([np.ones(63*41), np.zeros(63*41)])
    new_x = normalize(new_x)
    for idx, one_train_x, one_orig_test_x in zip(range(n_sets), train_x, orig_x):
        one_train_x = normalize(one_train_x)
        one_orig_test_x = normalize(one_orig_test_x)
        for f in range(n_frames):
            train_frame = one_train_x[:,:,f]
            test_frame_orig = one_orig_test_x[:,:,f]
            test_frame_new = new_x[:,:,f]

            clf = svm.SVC(gamma='auto')
            clf.fit(np.asarray(train_frame), train_y)

            orig_pred = clf.predict(test_frame_orig)
            acc = np.mean(orig_pred == orig_y)
            orig_acc[idx,f] = acc

            new_pred = clf.predict(test_frame_new)
            acc = np.mean(new_pred == new_y)
            new_acc[idx,f] = acc
        print(idx)

    np.save('orig', orig_acc)
    np.save('new', new_acc)
    orig_acc = np.mean(orig_acc, axis=0)
    new_acc = np.mean(new_acc, axis=0)
    plt.figure()
    plt.plot(frames, orig_acc, label='long SOA test')
    plt.plot(frames, new_acc, label='short SOA test')
    plt.legend()
    plt.hlines(0.5, frames[0], frames[-1], linestyles='dashed')
    plt.ylabel('accuracy')
    plt.xlabel('frame')
    plt.title('SVM trained on long SOA separate frames')
    plt.show()


def nn_train_test(train_x, train_y, orig_x, orig_y, new_x, new_y, frames):
    new_x = normalize(new_x)
    new_data = DataSet(new_x, new_y)

    n_frames = len(frames)
    n_sets = len(train_x)
    orig_acc = np.zeros([n_sets, n_frames])
    new_acc = np.zeros([n_sets, n_frames])
    D_in = train_x[2] * train_x[3]
    H1 = 100
    H2 = 50
    D_out = 1
    for idx, one_train_x, one_orig_test_x in zip(range(n_sets), train_x, orig_x):
        one_train_x = normalize(one_train_x)
        one_orig_test_x = normalize(one_orig_test_x)
        net = DenseNet(D_in, H1, H2, D_out)
        param = train_model.get_train_params(net)





def main():
    frames = np.arange(27, 68)
    n_frames = len(frames)
    n_seg = 104

    train_path = 'temp_outputs/0212-b/set2.npz'
    test_path = 'temp_outputs/0212-c/seg.npz'
    train_data_sets = data_io.read_from_file(train_path, 'set')
    train_x = train_data_sets[0]
    train_y = train_data_sets[1]
    original_test_x = train_data_sets[4]
    original_test_y = train_data_sets[5]

    n_sets = len(train_x)

    [v_test_data, h_test_data] = data_io.read_from_file(test_path, 'seg')
    n_v = v_test_data.shape[2]
    n_h = h_test_data.shape[2]

    train_x = train_x.reshape([n_sets, -1,n_seg, n_frames])
    original_test_x = original_test_x.reshape([n_sets, -1,n_seg, n_frames])

    v_test_data = v_test_data.transpose([2, 0, 1])
    h_test_data = h_test_data.transpose([2, 0, 1])
    new_test_x = np.concatenate([v_test_data, h_test_data], axis=0)
    new_test_y = np.concatenate([np.ones(n_v), np.zeros(n_h)])

    nn_train_test(train_x, train_y, original_test_x, original_test_y, new_test_x, new_test_y, frames)


main()
