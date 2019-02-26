import numpy as np
import augment
import torch


def get_data(param_v, param_h, n_train=3000, n_valid=50, n_test=50, flat_x=True, to_tensor=True):
    """
    Create the data set from the relevant parameters
    :param param_v: parameters for the creation of vertical condition
    :param param_h: parameters for the creation of horizontal condition
    :param n_train: number of training examples
    :param n_valid: number of validation examples
    :param n_test: number of test examples
    :param flat_x: True if the examples should be a 1d vector, false for 2d.
    :param to_tensor: True if the result should be a pytorch tensor, false for numpy ndarray.
    :return: A list of the three data-sets, including the y targets.
    """
    # create the train set
    train_v = augment.get_new_data(param_v, n_train)
    train_h = augment.get_new_data(param_h, n_train)
    train_x = np.concatenate([train_v, train_h])
    train_y = np.concatenate([np.ones(n_train), np.zeros(n_train)])

    # create the validation set
    valid_v = augment.get_new_data(param_v, n_valid)
    valid_h = augment.get_new_data(param_h, n_valid)
    valid_x = np.concatenate([valid_v, valid_h])
    valid_y = np.concatenate([np.ones(n_valid), np.zeros(n_valid)])

    # create the test set
    test_v = augment.get_new_data(param_v, n_test)
    test_h = augment.get_new_data(param_h, n_test)
    test_x = np.concatenate([test_v, test_h])
    test_y = np.concatenate([np.ones(n_test), np.zeros(n_test)])

    if flat_x:
        train_x = np.reshape(train_x, (n_train*2, -1))
        valid_x = np.reshape(valid_x, (n_valid*2, -1))
        test_x = np.reshape(test_x, (n_test*2, -1))

    if to_tensor:
        new_sets = []
        for d in [train_x, train_y, valid_x, valid_y, test_x, test_y]:
            new_sets.append(torch.from_numpy(d))
        return new_sets
    else:
        return train_x, train_y, valid_x, valid_y, test_x, test_y

