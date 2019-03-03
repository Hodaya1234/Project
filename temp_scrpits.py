# Convert some of the data set to mat files in order to view them in matlab
import numpy as np
import segment
import scipy.io as sio

set_file = np.load('temp_outputs/set.npz')
train_x = set_file['train_x']
seg_v = train_x[0,:]
seg_h = train_x[2000,:]
seg_file = np.load('temp_outputs/seg.npz')
mask = seg_file['mask']
n_seg = len(np.unique(mask))
if np.unique(mask)[0] == 0:
    n_seg = n_seg - 1
seg_v = seg_v.reshape((n_seg, -1))
seg_h = seg_h.reshape((n_seg, -1))
orig_v = segment.recreate_image(mask, seg_v)
orig_h = segment.recreate_image(mask, seg_h)
sio.savemat('temp_outputs/mat_examples', {'v': orig_v, 'h': orig_h})
