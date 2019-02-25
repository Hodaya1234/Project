from __future__ import print_function

import segment_data
import numpy as np
import data_augmentation
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import segmentation


def create_data(param, n, segs=None):
    if segs is None:
        return [data_augmentation.augment_examples(param) for i in range(n)]  # n_segments X n_frames
    else:
        return [segment_data.recreate_image(segs, data_augmentation.augment_examples(
            param)) for i in range(n)]  # n_segments X n_frames


def create_bm(mat_name, out_name):
    clean_data = sio.loadmat(mat_name)

    # upload the data
    full_clean_vert = clean_data['clean_vert']       # 10000 X 256 X n_trials
    full_clean_horiz = clean_data['clean_horiz']      # 10000 X 256 X n_trials

    clean_vert = full_clean_vert[:, range(28, 45), :]         # 10000 X n_frames X n_trials
    clean_horiz = full_clean_horiz[:, range(28, 45), :]       # 10000 X n_frames X n_trials
    clean_vert_avg_for_seg = np.mean(clean_vert, axis=2)      # average over trials, 10000 X n_frames
    clean_horiz_avg_for_seg = np.mean(clean_horiz, axis=2)    # average over trials, 10000 X n_frames

    mask = segment_data.get_mask(np.reshape(clean_vert[:, 0, 0], [100, 100]))

    # create segmentation for both trials
    segments = segment_data.vert_horiz_seg(clean_vert_avg_for_seg, clean_horiz_avg_for_seg, mask)     # 100 X 100 integers. same number indicates a single group of pixels

    segments_vert = segment_data.divide_data_to_segments(segments, clean_vert, mask)       # n_segments X n_frames X n_trials
    segments_horiz = segment_data.divide_data_to_segments(segments, clean_horiz, mask)     # n_segments X n_frames X n_trials

    recreated_vert = segment_data.recreate_image(segments, segments_vert)    # 10000 X n_frames X n_trials
    recreated_horiz = segment_data.recreate_image(segments, segments_horiz)  # 10000 X n_frames X n_trials

    parameters_v = data_augmentation.get_augmentation_parameters(segments_vert)
    parameters_h = data_augmentation.get_augmentation_parameters(segments_horiz)

    seg_h = create_data(parameters_h, 10, segments)
    seg_v = create_data(parameters_v, 10, segments)

    sio.savemat(out_name, {'v': seg_v, 'h': seg_h, 'orig_v': recreated_vert, 'orig_h': recreated_horiz, 'segments': segments})


def create_ret(mat_name, out_name):
    clean_data = sio.loadmat(mat_name)

    # upload the data
    full_clean_vert = clean_data['clean_vert']       # 10000 X 256 X n_trials
    full_clean_horiz = clean_data['clean_horiz']      # 10000 X 256 X n_trials
    segments = clean_data['segments']

    clean_vert = full_clean_vert[:, range(28, 45), :]         # 10000 X n_frames X n_trials
    clean_horiz = full_clean_horiz[:, range(28, 45), :]       # 10000 X n_frames X n_trials

    mask = segment_data.get_mask(np.reshape(clean_vert[:, 0, 0], [100, 100]))

    segments_vert = segment_data.divide_data_to_segments(segments, clean_vert, mask, keep_background=True)       # n_segments X n_frames X n_trials
    segments_horiz = segment_data.divide_data_to_segments(segments, clean_horiz, mask, keep_background=True)     # n_segments X n_frames X n_trials

    recreated_vert = segment_data.recreate_image(segments, segments_vert)    # 10000 X n_frames X n_trials
    recreated_horiz = segment_data.recreate_image(segments, segments_horiz)  # 10000 X n_frames X n_trials

    parameters_v = data_augmentation.get_augmentation_parameters(segments_vert)
    parameters_h = data_augmentation.get_augmentation_parameters(segments_horiz)

    seg_h = create_data(parameters_h, 10, segments)
    seg_v = create_data(parameters_v, 10, segments)

    sio.savemat(out_name, {'v': seg_v, 'h': seg_h, 'orig_v': recreated_vert, 'orig_h': recreated_horiz, 'segments': segments})


# list_f = ['clean-02.12', 'clean-14.10', 'clean-16.09', 'clean-23.09', 'clean-25.11']
# filename = 'C:\\Users\\H\\Desktop\\clean bm-02.12.mat'
# create_ret(filename, 'C:\\Users\\H\\Desktop\\02.12_bm_seg_ret.mat')
# create_bm(filename, 'C:\\Users\\H\\Desktop\\02.12_bm_seg_bm.mat')



# # create train set from the new examples, and two types of test data: from the new examples and from the original ones
# train_x = []
# train_y = []
# test_x_augmented = []
# test_y_augmented = []
# test_x_real = np.concatenate((np.asarray([segments_vert[:, :, i] for i in range(segments_vert.shape[2])]),
#                               np.asarray([segments_horiz[:, :, i] for i in range(segments_horiz.shape[2])])))
# test_y_real = np.concatenate((np.ones(segments_vert.shape[2]), np.zeros(segments_horiz.shape[2])))
#
# for i in range(3000):
#     print('example generation number {}'.format(i))
#     new_examp_v = data_augmentation.augment_examples(parameters_v)  # n_segments X n_frames
#     new_examp_h = data_augmentation.augment_examples(parameters_h)  # n_segments X n_frames
#     train_x.append(new_examp_v)
#     train_x.append(new_examp_h)
#     train_y.append(1)
#     train_y.append(0)
#
# for i in range(50):
#     new_examp_v = data_augmentation.augment_examples(parameters_v)  # n_segments X n_frames
#     new_examp_h = data_augmentation.augment_examples(parameters_h)  # n_segments X n_frames
#     test_x_augmented.append(new_examp_v)
#     test_x_augmented.append(new_examp_h)
#     test_y_augmented.append(1)  # vert is 1
#     test_y_augmented.append(0)  # horiz is 0
#
#
# np.save('train_x', np.asarray(train_x))
# np.save('train_y', np.asarray(train_y))
# np.save('test_x', np.asarray(test_x_augmented))
# np.save('test_y', np.asarray(test_y_augmented))
# np.save('test_x_real', test_x_real)
# np.save('test_y_real', test_y_real)

