import data_io
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from dense_net import DenseNet
import train_model
from data_set import DataSet
import torch.utils.data as data_utils
import torch
from torch import nn, optim
import segment
import create_data_set


def normalize(data, m=None, s=None):
    if m is None:
        m = np.mean(data, axis=0)
    if s is None:
        s = np.std(data, axis=0)
    return np.divide(np.subtract(data, m), s)


def svm_train_test_segs(train_x, train_y, orig_x, orig_y, new_x, new_y, mask):
    n_seg = len(np.unique(mask)) - 1
    n_sets = len(train_x)
    orig_acc = np.zeros([n_sets, n_seg])
    new_acc = np.zeros([n_sets, n_seg])
    # train_y = np.concatenate([np.ones(63*41), np.zeros(63*41)])
    new_x = normalize(new_x)
    for idx, one_train_x, one_orig_test_x in zip(range(n_sets), train_x, orig_x):
        one_train_x = normalize(one_train_x)
        one_orig_test_x = normalize(one_orig_test_x)
        for s in range(n_seg):
            train_frame = one_train_x[:,s,:]
            test_frame_orig = one_orig_test_x[:,s,:]
            test_frame_new = new_x[:,s,:]

            clf = svm.SVC(gamma='auto')
            clf.fit(np.asarray(train_frame), train_y)

            orig_pred = clf.predict(test_frame_orig)
            acc = np.mean(orig_pred == orig_y)
            orig_acc[idx,s] = acc

            new_pred = clf.predict(test_frame_new)
            acc = np.mean(new_pred == new_y)
            new_acc[idx,s] = acc
        print(idx)

    np.save('orig', orig_acc)
    np.save('new', new_acc)
    orig_acc = np.mean(orig_acc, axis=0)
    new_acc = np.mean(new_acc, axis=0)
    image_o = segment.recreate_image(mask, orig_acc)
    image_n = segment.recreate_image(mask, new_acc)
    max_val = np.max(np.max(orig_acc), np.max(new_acc))

    fig, axes = plt.subplots(nrows=1, ncols=2)
    im_o = axes.flat[0].imshow(image_o.reshape([100, 100]), vmin=0.5, vmax=max_val)
    axes[0].set_title('Accuracy on session b')
    im_n = axes.flat[1].imshow(image_n.reshape([100, 100]), vmin=0.5, vmax=max_val)
    axes[1].set_title('Accuracy on session c')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im_n, cax=cbar_ax)
    plt.show()


def svm_train_test_frame(train_x, train_y, orig_x, orig_y, new_x, new_y, frames):
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

            clf = svm.SVC(C=5, gamma='auto')
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
    plt.title('SVM trained on long SOA separate frames - NO AUGMENTATION')
    plt.show()


def svm_train_test_flat(train_x, train_y, orig_x, orig_y, new_x, new_y, frames):
    train_x = train_x.reshape([train_x.shape[0], train_x.shape[1], -1])
    orig_x = orig_x.reshape([orig_x.shape[0], orig_x.shape[1], -1])
    new_x = new_x.reshape([new_x.shape[0], -1])

    n_frames = len(frames)
    n_sets = len(train_x)

    orig_acc = np.zeros([n_sets,])
    new_acc = np.zeros([n_sets,])
    new_x = normalize(new_x)
    for idx, one_train_x, one_orig_test_x in zip(range(n_sets), train_x, orig_x):
        one_train_x = normalize(one_train_x)
        one_orig_test_x = normalize(one_orig_test_x)

        clf = svm.SVC(C=2, gamma='scale')
        clf.fit(np.asarray(one_train_x), train_y)

        orig_pred = clf.predict(one_orig_test_x)
        acc = np.mean(orig_pred == orig_y)
        orig_acc[idx] = acc

        new_pred = clf.predict(new_x)
        acc = np.mean(new_pred == new_y)
        new_acc[idx] = acc
        print(idx)
    plt.figure()
    plt.scatter(list(range(n_sets)), orig_acc, label='b test accuracy (mean={0: .4f})'.format(np.mean(orig_acc)))
    plt.scatter(list(range(n_sets)), new_acc, label='c test accuracy (mean={0: .4f})'.format(np.mean(new_acc)))
    plt.ylabel('accuracy')
    plt.xlabel('different cross validation sets')
    plt.title('SVM accuracy trained on b tested on c, frames flattened')
    plt.legend()
    plt.show()


def create_train_val_no_aug(all_v, all_h, n_sets=40):
    n_v = all_v.shape[2]
    n_h = all_h.shape[2]
    all_v = all_v.transpose([2, 0, 1])
    all_h = all_h.transpose([2, 0, 1])
    n_train = min(n_v, n_h) - 2
    [train_v_idx, test_v_idx] = create_data_set.get_train_test_indices(n_v, n_train, n_test=2, number_of_sets=n_sets,
                                                                  random=True)
    [train_h_idx, test_h_idx] = create_data_set.get_train_test_indices(n_h, n_train, n_test=2, number_of_sets=n_sets,
                                                                  random=True)
    train_x = []
    test_x = []
    train_y = np.concatenate([np.ones(n_train), np.zeros(n_train)])
    test_y = np.asarray([1,1,0,0])

    for s in range(n_sets):
        train_v = all_v[train_v_idx[s], :, :]
        train_h = all_h[train_h_idx[s], :, :]
        train_x.append(np.concatenate([train_v, train_h], axis=0))

        test_v = all_v[test_v_idx[s], :, :]
        test_h = all_h[test_h_idx[s], :, :]
        test_x.append(np.concatenate([test_v, test_h], axis=0))

    return np.asarray(train_x), train_y, np.asarray(test_x), test_y


def nn_train_test(train_x, train_y, orig_x, orig_y, new_x, new_y, frames):
    train_x = train_x.reshape([train_x.shape[0], train_x.shape[1], -1])
    orig_x = orig_x.reshape([orig_x.shape[0], orig_x.shape[1], -1])
    new_x = new_x.reshape([new_x.shape[0], -1])
    # n_frames = len(frames)
    n_sets = len(train_x)
    # orig_acc = np.zeros([n_sets, n_frames])
    # new_acc = np.zeros([n_sets, n_frames])

    D_in = train_x.shape[2]
    H1 = 100
    H2 = 10
    D_out = 1
    new_dataset = DataSet(new_x, new_y).normalize()

    n_epochs = 60

    loss_fn = nn.BCELoss()
    optimizer_type = optim.Adam
    scheduler_type = optim.lr_scheduler.MultiStepLR
    lr = 0.001

    train_losses = torch.zeros([n_sets, n_epochs])
    orig_losses = torch.zeros([n_sets, n_epochs])
    orig_accuracies = torch.zeros([n_sets, n_epochs])
    new_losses = torch.zeros([n_sets, n_epochs])
    new_accuracies = torch.zeros([n_sets, n_epochs])

    for idx, one_train_x, one_orig_x in zip(range(n_sets), train_x, orig_x):
        print(idx)
        train_dataset = DataSet(one_train_x, train_y).normalize()
        orig_dataset = DataSet(one_orig_x, orig_y).normalize()

        net = DenseNet(D_in, H1, H2, D_out)
        net = net.double()
        optimizer = optimizer_type(net.parameters(), lr=lr, weight_decay=0.1)
        scheduler = scheduler_type(optimizer, [20], gamma=0.1)

        train_loader = data_utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
        orig_y_int = orig_y.int()
        new_y_int = new_y.int()

        for e in range(n_epochs):
            scheduler.step()
            epoch_train_loss = []
            net.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = net(x)
                y_pred = y_pred.view(y_pred.numel())
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())
            train_losses[idx,e] = sum(epoch_train_loss) / len(epoch_train_loss)

            net.eval()
            with torch.no_grad():
                outputs = net(orig_dataset.all_x)
                outputs = outputs.view(outputs.numel())
                orig_losses[idx,e] = loss_fn(outputs, orig_dataset.all_y).item()

                predictions = torch.round(outputs).int()
                orig_accuracies[idx,e] = (predictions == orig_y_int).sum().item() / len(orig_y_int)

                outputs = net(new_dataset.all_x)
                outputs = outputs.view(outputs.numel())
                new_losses[idx,e] = loss_fn(outputs, new_dataset.all_y).item()

                predictions = torch.round(outputs).int()
                new_accuracies[idx,e] = (predictions == new_y_int).sum().item() / len(new_y_int)

            # if orig_losses[-1] < 0.01:
            #     return net, train_losses, orig_losses, orig_accuracies
            # if e > 18 and np.mean(orig_losses[-5:]) > np.mean(orig_losses[-10:-5]):
            #     print('finished at epoch {}'.format(e))
            #     return net, train_losses, orig_losses, orig_accuracies
            if e % 5 == 0:
                print("{}. train loss: {}   orig loss: {}  new loss: {}".format(
                    e, train_losses[idx,e], orig_losses[idx,e], new_losses[idx,e]))
    np.savez('diff_sess_loss', train_losses=train_losses, orig_losses=orig_losses, orig_accuracies=orig_accuracies,
             new_losses=new_losses, new_accuracies=new_accuracies)


def nn_train_test_frame(train_x, train_y, orig_x, orig_y, new_x, new_y, frames):
    train_x = train_x.reshape([train_x.shape[0], train_x.shape[1], -1])
    orig_x = orig_x.reshape([orig_x.shape[0], orig_x.shape[1], -1])
    new_x = new_x.reshape([new_x.shape[0], -1])
    n_frames = len(frames)
    n_sets = len(train_x)
    # orig_loss = np.zeros([n_sets, n_frames])
    # new_loss = np.zeros([n_sets, n_frames])

    D_in = train_x.shape[2]
    H1 = 10
    H2 = 5
    D_out = 1
    new_dataset = DataSet(new_x, new_y).normalize()

    n_epochs = 60

    loss_fn = nn.BCELoss()
    optimizer_type = optim.Adam
    scheduler_type = optim.lr_scheduler.MultiStepLR
    lr = 0.001

    train_losses = torch.zeros([n_sets, n_epochs])
    orig_losses = torch.zeros([n_sets, n_epochs])
    orig_accuracies = torch.zeros([n_sets, n_epochs])
    new_losses = torch.zeros([n_sets, n_epochs])
    new_accuracies = torch.zeros([n_sets, n_epochs])

    for idx, one_train_x, one_orig_x in zip(range(n_sets), train_x, orig_x):
        print(idx)
        train_dataset = DataSet(one_train_x, train_y).normalize()
        orig_dataset = DataSet(one_orig_x, orig_y).normalize()

        net = DenseNet(D_in, H1, H2, D_out)
        net = net.double()
        optimizer = optimizer_type(net.parameters(), lr=lr, weight_decay=0.1)
        scheduler = scheduler_type(optimizer, [20], gamma=0.1)

        train_loader = data_utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
        orig_y_int = orig_y.int()
        new_y_int = new_y.int()

        for e in range(n_epochs):
            scheduler.step()
            epoch_train_loss = []
            net.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = net(x)
                y_pred = y_pred.view(y_pred.numel())
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())
            train_losses[idx,e] = sum(epoch_train_loss) / len(epoch_train_loss)

            net.eval()
            with torch.no_grad():
                outputs = net(orig_dataset.all_x)
                outputs = outputs.view(outputs.numel())
                orig_losses[idx,e] = loss_fn(outputs, orig_dataset.all_y).item()

                predictions = torch.round(outputs).int()
                orig_accuracies[idx,e] = (predictions == orig_y_int).sum().item() / len(orig_y_int)

                outputs = net(new_dataset.all_x)
                outputs = outputs.view(outputs.numel())
                new_losses[idx,e] = loss_fn(outputs, new_dataset.all_y).item()

                predictions = torch.round(outputs).int()
                new_accuracies[idx,e] = (predictions == new_y_int).sum().item() / len(new_y_int)

            # if orig_losses[-1] < 0.01:
            #     return net, train_losses, orig_losses, orig_accuracies
            # if e > 18 and np.mean(orig_losses[-5:]) > np.mean(orig_losses[-10:-5]):
            #     print('finished at epoch {}'.format(e))
            #     return net, train_losses, orig_losses, orig_accuracies
            if e % 5 == 0:
                print("{}. train loss: {}   orig loss: {}  new loss: {}".format(
                    e, train_losses[idx,e], orig_losses[idx,e], new_losses[idx,e]))
    np.savez('diff_sess_loss', train_losses=train_losses, orig_losses=orig_losses, orig_accuracies=orig_accuracies,
             new_losses=new_losses, new_accuracies=new_accuracies)


def main():
    frames = np.arange(27, 68)
    n_frames = len(frames)

    seg_train_path = 'Data/Segmented/b_sq.npz'
    train_path = 'Data/Sets/b_sq_no_norm.npz'
    test_path = 'Data/Segmented/c_sq.npz'
    mask = np.load('Data/Masks/a_sq.npy')
    n_seg = len(np.unique(mask)) - 1

    train_data_sets = data_io.read_from_file(train_path, 'set')
    train_x = train_data_sets[0]
    train_y = train_data_sets[1]
    original_test_x = train_data_sets[4]
    original_test_y = train_data_sets[5]

    n_sets = len(train_x)
    train_x = train_x.reshape([n_sets, -1,n_seg, n_frames])
    original_test_x = original_test_x.reshape([n_sets, -1,n_seg, n_frames])


    [v_test_data, h_test_data] = data_io.read_from_file(test_path, 'seg')
    n_v = v_test_data.shape[2]
    n_h = h_test_data.shape[2]
    v_test_data = v_test_data.transpose([2, 0, 1])
    h_test_data = h_test_data.transpose([2, 0, 1])
    new_test_x = np.concatenate([v_test_data, h_test_data], axis=0)
    new_test_y = np.concatenate([np.ones(n_v), np.zeros(n_h)])

    [v_seg, h_seg] = data_io.read_from_file(seg_train_path, 'seg')
    train_x, train_y, test_orig_x, test_orig_y = create_train_val_no_aug(v_seg, h_seg)

    svm_train_test_frame(train_x, train_y, test_orig_x, test_orig_y, new_test_x, new_test_y, frames)


    # nn_train_test_frame(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(original_test_x), torch.from_numpy(original_test_y), torch.from_numpy(new_test_x), torch.from_numpy(new_test_y), frames)


    # svm_train_test_segs(train_x, train_y, original_test_x, original_test_y, new_test_x, new_test_y, mask)

    # nn_train_test(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(original_test_x), torch.from_numpy(original_test_y), torch.from_numpy(new_test_x), torch.from_numpy(new_test_y), frames)


main()
