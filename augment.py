import numpy as np
from scipy import linalg


def get_parameters(segmented_data):
    """
    Extract parameters from the segmented data, in order to save the re calculation of them when the data is augmented.
    :param segmented_data: The data to be augmented.
    :return: The following parameters:
        n_pixels: number of data points. It is in fact n_segments * n_frames.
        sqrt_cov: the square root of the covariance matrix of these pixels, over trials.
        mean_trials: the mean of the data points over trials.
        new_shape: the shape of the new data, usually [100 X 100 X n_frames].
        n_frames is given as the second dimension (if the first is 10,000) or the third (if the first two are 100X100)
    """
    # arrayed_segments = segmented_data.reshape(num_trials, -1)  # squeeze the pixels and frames to a 1d vector.
    arrayed_segments = segmented_data
    num_pixels = arrayed_segments.shape[1]
    mean_trials = np.nanmean(arrayed_segments, axis=0)

    cov_mat = np.cov(arrayed_segments.T)      # n_segments X n_segments
    sqrt_cov = linalg.sqrtm(cov_mat).real   # n_segments X n_segments
    parameters = [num_pixels, sqrt_cov, mean_trials]
    return parameters


def get_new_data(parameters, n=10):
    """
    Use the parameters that were found on the get_parameters function and create n new data examples.
    :param parameters: a list containing:
        n_pixels: number of data points. It is in fact n_segments * n_frames.
        sqrt_cov: the square root of the covariance matrix of these pixels, over trials.
        mean_trials: the mean of the data points over trials.
        new_shape: the shape of the new data, usually [100 X 100 X n_frames].
    :param n: the number of data examples to create.
    :return: n new data examples.
    """
    num_pixels, sqrt_cov, mean_trials = parameters
    data = np.empty([n, num_pixels])
    for i in range(n):
        random_mat = np.random.normal(0, 1, num_pixels)
        new_example = (np.dot(random_mat, sqrt_cov) + mean_trials)
        data[i, :] = new_example
    return data
