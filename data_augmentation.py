import numpy as np
from scipy import linalg


def get_augmentation_parameters(segmented_data):
    num_trials = segmented_data.shape[2]
    arrayed_segments = segmented_data.reshape(-1, num_trials)
    num_pixels = arrayed_segments.shape[0]
    mean_trials = np.nanmean(arrayed_segments, axis=1)

    cov_mat = np.cov(arrayed_segments)
    sqrt_cov = linalg.sqrtm(cov_mat).real
    new_shape = segmented_data.shape[:-1]
    parameters = [num_pixels, sqrt_cov, mean_trials, new_shape]
    return parameters


def augment_examples(parameters):
    num_pixels, sqrt_cov, mean_trials, new_shape = parameters
    random_mat = np.reshape(np.random.multivariate_normal(np.zeros(num_pixels), np.identity(num_pixels)), (1, -1))
    new_example = (np.dot(random_mat, sqrt_cov) + mean_trials).reshape(new_shape)
    return new_example
