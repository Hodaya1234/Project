import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import felzenszwalb, join_segmentations
import warnings


def vert_horiz_seg(vert, horiz, square=False):
    """
    The 'main' function of this module - take the two conditions and divide the pixels to smaller groups.
    :param vert: The vertical condition data of size [10,000 X n_frames] OR [100 X 100 X n_frames]
    :param horiz: The horizontal condition data data of size [10,000 X n_frames] OR [100 X 100 X n_frames]
    :return: A [100 X 100] integer mask where each number marks a group of neighboring related pixels.
    """
    # if there are several trials, need to average over trials first and remove this dimension
    if vert.shape[0] != horiz.shape[0]:
        warnings.warn('Frame size of both conditions don\'t match')
        return
    vert_data = np.copy(vert)
    horiz_data = np.copy(horiz)
    vert_data = change_data_dim(vert_data, to_array=False)  # make sure both are represented as matrices and not arrays
    horiz_data = change_data_dim(horiz_data, to_array=False)

    vert_data = np.mean(vert_data[:, :, :, :], 3)
    horiz_data = np.mean(horiz_data[:, :, :, :], 3)
    if square:
        vert_data = np.square(vert_data)
        horiz_data = np.square(horiz_data)

    vert_data = change_range(vert_data)
    horiz_data = change_range(horiz_data)
    combination_data = np.concatenate((horiz_data, vert_data), axis=2)
    bg_mask = get_bg_mask(combination_data[:, :, 1])
    all_segments = segment_felz(combination_data, bg_mask)

    diff_numbers = np.unique(all_segments)
    for idx, num in enumerate(diff_numbers):
        # if the segment numbers are sparse, "squeeze" them.
        # might happen because the background mask "took over" several segment numbers
        all_segments[all_segments == num] = idx
    # TODO: remove this
    print(len(np.unique(all_segments)))
    return all_segments


def divide_data_to_segments(segments_matrix, raw_data, keep_background=False):
    """
    Take the original data and the segments mask matrix and create a segmented data
    :param segments_matrix: [100 X 100] integer mask
    :param raw_data: The original data, where the size of the first dimension is 10,000 or 100 X 100.
    :param frames_for_data A list of numbers of the relevant frames
    :return: The data rearranged in the segments, where the value in each segment is the mean of the pixels in it.
    """
    # if the first two dimension are 100X100 turn them to 10,000:
    data_for_seg = np.copy(raw_data)
    data_for_seg = change_data_dim(data_for_seg, to_array=True)

    seg_numbers = np.unique(segments_matrix)
    if seg_numbers[0] == 0:
        seg_numbers = seg_numbers[1:]
    else:
        data_for_seg[data_for_seg == 0] = np.nan
    array_segments = np.reshape(segments_matrix, (10000, 1))
    frames_shape = np.copy(data_for_seg.shape)

    frames_shape[0] = len(seg_numbers)  # The new first dimension will be n_segments instead of 10,000
    segmented_data = np.zeros(frames_shape)
    if len(frames_shape) == 3:
        # in case there are numerous trials and frames
        for idx, num in enumerate(seg_numbers):
            pixel_indexes = np.nonzero(array_segments == num)[0]
            segmented_data[idx, :, :] = np.nanmean(data_for_seg[pixel_indexes, :, :], axis=0)
    elif len(frames_shape) == 2:
        # in case there are numerous trials or numerous frames
        for idx, num in enumerate(seg_numbers):
            pixel_indexes = np.nonzero(array_segments == num)[0]
            segmented_data[idx, :] = np.nanmean(data_for_seg[pixel_indexes, :], axis=0)
    else:
        # in case there is a single frame total
        for idx, num in enumerate(seg_numbers):
            pixel_indexes = np.nonzero(array_segments == num)[0]
            segmented_data[idx] = np.nanmean(data_for_seg[pixel_indexes], axis=0)
    if keep_background:
        segmented_data[np.isnan(segmented_data)] = 0
    # TODO: transpose the data and adjust all the relevant functions
    return segmented_data


def recreate_image(segments_mask, segments_values):
    """
    Use the mask and the value of each segment and recreate an image.
    :param segments_mask: The [100 X 100] segments mask.
    :param segments_values: The data of size [n_segments X n_frames X n_trials].
    :return: The recreated data, of shape [10,000 X n_frames X n_trials].
    """
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


def segment_felz(data, background_mask, scale=37, sigma=0, min_size=9):
    """
    Use the scikit-image implementation of felzenszwalb algorithm
    :param data: The original data, of dimensions [100 X 100 X n_frames*2] (The teo conditions are concatenated on the
    third dimension)
    :param background_mask: A boolean matrix that is 0 where the pixels should be treated as background
    :param scale: The approximated size of segments
    :param sigma: Used to smooth the data. Should not be used here so the default is 0.
    :param min_size: Minimum size og all segments.
    :return: [100 X 100] integer matrix of the segments.
    """
    segments = felzenszwalb(data, scale=scale, sigma=sigma, min_size=min_size)
    segments[background_mask == 0] = 0
    return segments


def change_data_dim(data, to_array=True):
    """
    change the frame size to either a 1d array of [10,000] pixels OR a 2d matrix of [100 X 100].
    :param data: The original data of size [10,000 X n_frames X n_trials]
    Or of size [100 X 100 X n_frames X n_trials].
    Where n_trials could be 0, or both n_trials and n_frames could be 0.
    :param to_array: True if the data needs to be transformed to a 1d array, and false if it needs to be transformed
    to a matrix.
    :return: The transformed data, or the original if its already in the right form.
    """
    data_shape = data.shape
    new_shape = np.copy(data_shape)
    if (to_array and data_shape[0] == 10000) or (~to_array and data_shape[0] == 100):
        return data         # It's already in the right form
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


def get_bg_mask(frame):
    """
    Return a boolean matrix - 0 where the frame is 0, and 1 otherwise
    :param frame: a [10,000] or [100 X 100] single frame, where there is 0 in the background and positive real numbers
    elsewhere.
    :return: A boolean matrix of same size that is 0 where the original data is 0, and 1 otherwise.
    """
    mask = np.zeros_like(frame, dtype=int)
    mask[frame > 0] = 1
    return mask


def change_range(data, res_min=0, res_max=1, no_zero=True):
    """
    In order to use the scikit-image functions, need to change the range of the data to fit standard images.
    This does the change in a linear fashion.
    :param data: original data, values are around 1 or 0 where there is background.
    :param res_min: The new minimum of the range.
    :param res_max: The new maximum.
    :param no_zero: if this is true, ignore the background in the calculation.
    :return:
    """
    if no_zero:
        data[data == 0] = np.nan
    min_d = np.nanmin(data)
    max_d = np.nanmax(data)
    new_data = np.divide(np.subtract(data, min_d), max_d - min_d)
    new_data = new_data*(res_max-res_min) + res_min
    if no_zero:
        new_data[np.isnan(new_data)] = 0
        data[np.isnan(data)] = 0
    return new_data


def bin_seg(bin_size=10):
    """
    Divide the data into square bins of equal sizes.
    :param bin_size: The size of a side of each bin.
    :return: A [100 X 100] integers mask of squared bins.
    """
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


def tune_parameters(img, bg_mask):
    """
    Try different hyper parameters of segmentation in order to find the best ones.
    :param img: The data to be segmented.
    :param bg_mask: The background mask
    :return: Show a figure presenting the result.
    """
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
        segments = join_segmentations(bg_mask, seg)
        r = np.mod(i, rows)
        c = int(np.floor(i / rows))
        ax[r, c].imshow(segments)
        ax[r, c].axis('off')
        ax[r, c].set_title(titles[i], {'fontsize': 9})
    plt.show()

