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
    num_trials = segmented_data.shape[2]
    arrayed_segments = segmented_data.reshape(-1, num_trials)  # squeeze the pixels and frames to a 1d vector.
    num_pixels = arrayed_segments.shape[0]
    mean_trials = np.nanmean(arrayed_segments, axis=1)

    cov_mat = np.cov(arrayed_segments)
    sqrt_cov = linalg.sqrtm(cov_mat).real
    new_shape = segmented_data.shape[:-1]
    parameters = [num_pixels, sqrt_cov, mean_trials, new_shape]
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
    num_pixels, sqrt_cov, mean_trials, new_shape = parameters
    data = []
    for _ in range(n):
        random_mat = np.reshape(np.random.multivariate_normal(np.zeros(num_pixels), np.identity(num_pixels)), (1, -1))
        new_example = (np.dot(random_mat, sqrt_cov) + mean_trials).reshape(new_shape)
        data.append(new_example)
    return np.asarray(data)
