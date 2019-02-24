import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import felzenszwalb, join_segmentations
import warnings


def segment_felz(data, background_mask, scale=37, sigma=0, min_size=9):
    segments = felzenszwalb(data, scale=scale, sigma=sigma, min_size=min_size)
    segments[background_mask == 0] = 0
    return segments


def change_data_dim(data, to_array=True):
    #   change the size of the first dimension from 100 X 100 to 10000 or from 10000 to 100 X 100
    data_shape = data.shape
    new_shape = np.copy(data_shape)
    if (to_array and data_shape[0] == 10000) or (~to_array and data_shape[0] == 100):
        return data

    if len(data_shape) == 1:
        new_shape = (100, 100)

    elif len(data_shape) == 2:
        if to_array:
            new_shape = (10000, 1)
        else:
            new_shape = (100, 100, data_shape[1])

    elif len(data_shape) == 3:
        if to_array:
            new_shape = (10000, data_shape[2])
        else:
            new_shape = (100, 100, data_shape[1], data_shape[2])

    elif len(data_shape) == 4:
        if to_array:
            new_shape = (10000, data_shape[2], data_shape[3])

    return np.reshape(data, new_shape)


def get_mask(frame):
    # return a boolean matrix - 0 where the frame is 0, and 1 otherwise
    mask = np.zeros_like(frame, dtype=int)
    mask[frame > 0] = 1
    return mask


def vert_horiz_seg(vert, horiz, mask):
    # assume vert data and horiz data are 10000 X n_frames or 100 X 100 X n_frames
    # if there are several trials, need to average over trials first
    if vert.shape[0] != horiz.shape[0]:
        warnings.warn('Frame size of both conditions don\'t match')
        return
    vert_data = np.copy(vert)
    horiz_data = np.copy(horiz)
    vert_data = change_data_dim(vert_data, to_array=False)  # make sure both are represented as matrixes and not arrays
    horiz_data = change_data_dim(horiz_data, to_array=False)

    vert_data = change_range(vert_data)
    horiz_data = change_range(horiz_data)

    combination_data = np.concatenate((horiz_data, vert_data), axis=2)
    all_segments = segment_felz(combination_data, mask)
    diff_numbers = np.unique(all_segments)
    for idx, num in enumerate(diff_numbers):    # in case the segment numbers are sparse, "squeeze" them. might happen because the background mask "took over" several segment numbers
        all_segments[all_segments == num] = idx
    return all_segments


def change_range(data, res_min=0, res_max=1, no_zero=True):
    if no_zero:
        data[data == 0] = np.nan
    min_d = np.nanmin(data)
    max_d = np.nanmax(data)
    new_data = np.divide(np.subtract(data, min_d), max_d - min_d)
    new_data = new_data*(res_max-res_min) + res_min
    if no_zero:
        new_data[np.isnan(new_data)] = 0
    return new_data


def bin_seg(bin_size=10):
    # divide the 100X100 frame to square bins
    if 100 % bin_size != 0:
        bin_size = 10
    n_bins = int(100 / bin_size)  # number of bins in one dimension
    mat_array = np.empty([0, 0], dtype=int)
    for i in range(n_bins):
        current_row = np.empty(0, dtype=int)
        for j in range(n_bins):
            current_row = np.concatenate([current_row, np.repeat(n_bins * i + j, bin_size)])
        if mat_array.size == 0:
            mat_array = np.copy(np.tile(current_row, (bin_size, 1)))
        else:
            mat_array = np.concatenate([mat_array, np.tile(current_row, (bin_size, 1))], axis=0)
    return mat_array


def tune_parameters(img, mask):
    segments_images = []
    scales = [10, 40]
    sigmas = [0.1]
    min_sizes = [5, 10]
    titles = []
    for scale in scales:
        for sigma in sigmas:
            for min_size in min_sizes:
                current_segment = felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
                segments_images.append(current_segment)
                titles.append('scale={}, min_size={}, num_segments={}'.format(scale, min_size,
                                                                              len(np.unique(current_segment))))

    total_size = scales.__len__() * sigmas.__len__() * min_sizes.__len__()
    cols = int(np.floor(np.sqrt(total_size)))
    rows = int(np.ceil(np.sqrt(total_size)))

    fig, ax = plt.subplots(rows, cols)
    for i, seg in enumerate(segments_images):
        print(i)
        segments = join_segmentations(mask, seg)
        r = np.mod(i, rows)
        c = int(np.floor(i / rows))
        ax[r, c].imshow(segments)
        ax[r, c].axis('off')
        ax[r, c].set_title(titles[i], {'fontsize': 9})
    plt.show()


def divide_data_to_segments(segments_matrix, frames_data, background_mask=None, keep_background=False):
    # segments_matrix - size 100 X 100 int
    # frames_data - size 10,000 X n_frames X n_trials or size 100 X 100 X n_frames X n_trials
    # background_mask - 100X100 matrix. 0 indicates background and 1 indicates data
    seg_numbers = np.unique(segments_matrix)
    if not keep_background:
        seg_numbers = seg_numbers[1:]
    else:
        frames_data[frames_data == 0] = np.nan
    array_segments = np.reshape(segments_matrix, (10000, 1))
    frames_shape = np.copy(frames_data.shape)
    frames_data = change_data_dim(frames_data, to_array=True)   # if the first two dimension are 100X100 turn them to 10,000
    frames_shape[0] = len(seg_numbers)  # The new first dimension will be n_segments instead of 10,000
    segmented_data = np.zeros(frames_shape)
    if len(frames_shape) == 3:      # in case there are numerous trials and frames
        for idx, num in enumerate(seg_numbers):
            pixel_indexes = np.nonzero(array_segments == num)[0]
            segmented_data[idx, :, :] = np.nanmean(frames_data[pixel_indexes, :, :], axis=0)
    elif len(frames_shape) == 2:    # in case there are numerous trials or numerous frames
        for idx, num in enumerate(seg_numbers):
            pixel_indexes = np.nonzero(array_segments == num)[0]
            segmented_data[idx, :] = np.nanmean(frames_data[pixel_indexes, :], axis=0)
    else:                           # in case there is a single frame total
        for idx, num in enumerate(seg_numbers):
            pixel_indexes = np.nonzero(array_segments == num)[0]
            segmented_data[idx] = np.nanmean(frames_data[pixel_indexes], axis=0)
    if keep_background:
        frames_data[np.isnan(frames_data)] = 0
        segmented_data[np.isnan(segmented_data)] = 0
    return segmented_data


def recreate_image(segments_mask, segments_values):
    #   segments_mask is 10,000 X 1 or 100 X 100, and segments_values is n_segments X n_frames X n_trials

    segments_mask = np.reshape(segments_mask, (10000, 1))
    images_size = np.copy(segments_values.shape)
    images_size[0] = 10000
    images = np.zeros(images_size)
    seg_numbers = np.unique(segments_mask)
    if seg_numbers[0] == 0:
        seg_numbers = seg_numbers[1:]
    if len(images_size) == 3:
        for index, number in enumerate(seg_numbers):
            pixel_indexes = np.nonzero(segments_mask == number)[0]
            images[pixel_indexes, :, :] = segments_values[index, :, :]
    elif len(images_size) == 2:
        for index, number in enumerate(seg_numbers):
            pixel_indexes = np.nonzero(segments_mask == number)[0]
            images[pixel_indexes, :] = segments_values[index, :]
    else:
        for index, number in enumerate(seg_numbers):
            pixel_indexes = np.nonzero(segments_mask == number)[0]
            images[pixel_indexes] = segments_values[index]
    return images

