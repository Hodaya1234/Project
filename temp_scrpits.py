# Convert some of the data set to mat files in order to view them in matlab
import numpy as np
import segment
import scipy.io as sio
import model
import matplotlib.pyplot as plt
import create_data_set
import scipy.io as sio
import segment

data_set = np.load('temp_outputs/set.npz')
test_x = data_set['test_x']
seg_file = np.load('temp_outputs/seg.npz')
segs = seg_file['mask']
n_seg = len(np.unique(segs)) - 1
test_x = test_x.T # pixels X n_examples
test_x = test_x.reshape((n_seg, -1, 100))
test_x = segment.recreate_image(segs, test_x)
sio.savemat('mat_data', {'test':test_x})


# set_file = np.load('temp_outputs/seg.npz')
# seg_v = set_file['seg_v']
# seg_h = set_file['seg_h']
# h = seg_h[:,:,:-1]
# train_x = np.concatenate((seg_v, h), 2)
# train_x = np.reshape(train_x, (-1, 14)).T
#
# train_y = np.concatenate((np.ones(7), np.zeros(7)))
# data_sets = create_data_set.np_to_tensor([train_x, train_y, train_x, train_y, train_x, train_y])
# train_losses, test_losses = model.train(data_sets)
# plt.figure()
# plt.plot(train_losses, label="train loss")
# # plt.plot(test_losses, label="test loss")
# plt.legend()
# plt.show()

# set_file = np.load('temp_outputs/set.npz')
# train_x = set_file['train_x']
# seg_v = train_x[0,:]
# seg_h = train_x[2000,:]
# seg_file = np.load('temp_outputs/seg.npz')
# mask = seg_file['mask']
# n_seg = len(np.unique(mask))
# if np.unique(mask)[0] == 0:
#     n_seg = n_seg - 1
# seg_v = seg_v.reshape((n_seg, -1))
# seg_h = seg_h.reshape((n_seg, -1))
# orig_v = segment.recreate_image(mask, seg_v)
# orig_h = segment.recreate_image(mask, seg_h)
# sio.savemat('temp_outputs/mat_examples', {'v': orig_v, 'h': orig_h})
