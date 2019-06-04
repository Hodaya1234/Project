import numpy as np
import augment
import torch
from data_set import DataSet
import matplotlib.pyplot as plt


def turn_to_torch_dataset(x,y):
    return DataSet(torch.from_numpy(x), torch.from_numpy(y))


def turn_to_torch_dataset_old(data_sets, cv=True):
    """
    This function is for the 'cv' data, that is made of dictionaries
    :param data_sets: numpy arrays of the the train, validation and test
    :param cv: does the data come in list of arrays or as simple arrays.
    :return: Three DataSet objects
    """
    if not cv:
        train_x, train_y, validation_x, validation_y, test_x, test_y = data_sets
        # TODO get the d_in out of here
        D_in = train_x.shape[1]
        train = DataSet(torch.from_numpy(train_x), torch.from_numpy(train_y))
        valid = DataSet(torch.from_numpy(validation_x), torch.from_numpy(validation_y))
        test = DataSet(torch.from_numpy(test_x), torch.from_numpy(test_y))
        return train, valid, test, D_in
    else:
        train_sets_x, train_y, validation_sets_x, valid_y, test_sets_x, test_y = data_sets
        D_in = train_sets_x[0].shape[1]
        train = []
        valid = []
        test = []
        train_y, valid_y, test_y = torch.tensor(train_y), torch.tensor(valid_y), torch.tensor(test_y)
        for tr in train_sets_x:
            train.append(DataSet(torch.from_numpy(tr), train_y))
        for v in validation_sets_x:
            valid.append(DataSet(torch.from_numpy(v), valid_y))
        for te in test_sets_x:
            test.append(DataSet(torch.from_numpy(te), test_y))
        return train, valid, test, D_in


def get_train_test_indices(n_total, n_train, n_test, number_of_sets=1, random=False):
    if number_of_sets == 1:
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:n_train+n_test]
        return train_indices, test_indices
    else:
        all_train_indices = []
        all_test_indices = []
        if random:
            for i in range(number_of_sets):
                train_indices, test_indices = get_train_test_indices(n_total, n_train, n_test, number_of_sets=1, random=random)
                all_train_indices.append(train_indices)
                all_test_indices.append(test_indices)
        else:
            indices = np.random.permutation(n_total)    # The original examples that are left out are selected one by one, but the original indices are mixed so that in each run of the program different sets will be selected
            for i in range(number_of_sets):
                test_indices = indices[[j % n_total for j in range(i, i+n_test)]]
                train_indices = indices[[j % n_total for j in range(i+n_test, i+n_test+n_train)]]
                all_train_indices.append(train_indices)
                all_test_indices.append(test_indices)
        return all_train_indices, all_test_indices


def create_one_data_sets(train, n_train, n_valid):
    params = augment.get_parameters(train)
    train_set = augment.get_new_data(params, n_train)
    train_set = np.concatenate((train_set, train), 0)
    valid_set = augment.get_new_data(params, n_valid)
    return train_set, valid_set


def get_data(seg_v, seg_h, n_train=3000, n_valid=50, n_test=2, cv=True, flat_x=True, to_tensor=True, random=False, normalize=False):
    """
    Create the data set from the relevant parameters
    :param seg_v: The segmented original vertical condition
    :param seg_h: The segmented original horizontal condition
    :param n_train: number of training examples
    :param n_valid: number of validation examples
    :param n_test: number of test examples
    :param cv: should the data be created using cross validation
    :param flat_x: True if the examples should be a 1d vector, false for 2d.
    :param to_tensor: True if the result should be a pytorch tensor, false for numpy ndarray.
    :param random: true if the left out original trials should be picked at random, false if they should be picked by order.
    :return: A list of the three data-sets, including the y targets.
    """
    seg_v, seg_h = np.transpose(seg_v, [2, 0, 1]), np.transpose(seg_h, [2, 0, 1])
    num_v = seg_v.shape[0]
    num_h = seg_h.shape[0]
    seg_v = seg_v.reshape([num_v, -1])
    seg_h = seg_h.reshape([num_h, -1])
    if not normalize:
        seg_v = seg_v - 1
        seg_h = seg_h - 1

    if cv:
        train_sets_x = []
        validation_sets_x = []
        test_sets_x = []

        number_of_sets = num_v + num_h    # if the data sets are not the same size, use the minimal size
        n_original = min(num_v, num_h) - n_test  # number of original training examples for each set from which to create the augmented data
        train_indices_v, test_indices_v = get_train_test_indices(num_v, n_original, n_test, number_of_sets=number_of_sets, random=random)
        train_indices_h, test_indices_h = get_train_test_indices(num_h, n_original, n_test, number_of_sets=number_of_sets, random=random)

        for i in range(number_of_sets):
            test_v_i = seg_v[test_indices_v[i], :]
            test_h_i = seg_h[test_indices_h[i], :]
            test_x = np.concatenate((test_v_i, test_h_i), 0)

            train_v_i = seg_v[train_indices_v[i], :]
            train_h_i = seg_h[train_indices_h[i], :]

            if normalize:
                train_i = np.concatenate([train_v_i, train_h_i], axis=0)
                mean_i = np.mean(train_i, axis=0)
                std_i = np.std(train_i, axis=0)
                mean_i = mean_i[np.newaxis, :]
                std_i = std_i[np.newaxis, :]
                train_v_i = np.divide(np.subtract(train_v_i, mean_i), std_i)
                train_h_i = np.divide(np.subtract(train_h_i, mean_i), std_i)
                test_x = np.divide(np.subtract(test_x, mean_i), std_i)

            train_set_v, valid_set_v = create_one_data_sets(train_v_i, n_train, n_valid)
            train_set_h, valid_set_h = create_one_data_sets(train_h_i, n_train, n_valid)
            train_x = np.concatenate((train_set_v, train_set_h), 0)
            valid_x = np.concatenate((valid_set_v, valid_set_h), 0)

            print('finished train number ' + str(i))
            if flat_x:
                train_x = np.reshape(train_x, (train_x.shape[0], -1))
                valid_x = np.reshape(valid_x, (valid_x.shape[0], -1))
                test_x = np.reshape(test_x, (test_x.shape[0], -1))
            if to_tensor:
                train_x, valid_x, test_x = np_to_tensor([train_x, valid_x, test_x])
            if not normalize:
                train_x = train_x + 1
                valid_x = valid_x + 1
                test_x = test_x + 1
            train_sets_x.append(train_x)
            validation_sets_x.append(valid_x)
            test_sets_x.append(test_x)
        train_y = np.concatenate((np.ones(n_train+n_original), np.zeros(n_train+n_original))) # n_train is the number of augmented examples, while n_train_from_original is the number of original examples in each data set
        valid_y = np.concatenate((np.ones(n_valid), np.zeros(n_valid)))
        test_y = np.concatenate((np.ones(n_test), np.zeros(n_test)))
        if to_tensor:
            train_y, valid_y, test_y = np_to_tensor([train_y, valid_y, test_y])
        return train_sets_x, train_y, validation_sets_x, valid_y, test_sets_x, test_y
    else:
        # get the parameters for each condition:
        param_v = augment.get_parameters(seg_v)
        param_h = augment.get_parameters(seg_h)
        # create the train set
        print('creating train')
        train_v = augment.get_new_data(param_v, n_train)
        print('done train vertical')
        train_h = augment.get_new_data(param_h, n_train)
        print('done train horizontal')
        train_x = np.concatenate([train_v, train_h])
        train_y = np.concatenate([np.ones(n_train), np.zeros(n_train)])

        # create the validation set
        print('creating validation')
        valid_v = augment.get_new_data(param_v, n_valid)
        valid_h = augment.get_new_data(param_h, n_valid)
        valid_x = np.concatenate([valid_v, valid_h])
        valid_y = np.concatenate([np.ones(n_valid), np.zeros(n_valid)])

        # create the test set
        print('creating test')
        test_v = augment.get_new_data(param_v, n_test)
        test_h = augment.get_new_data(param_h, n_test)
        test_x = np.concatenate([test_v, test_h])
        test_y = np.concatenate([np.ones(n_test), np.zeros(n_test)])

        if flat_x:
            train_x = np.reshape(train_x, (n_train*2, -1))
            valid_x = np.reshape(valid_x, (n_valid*2, -1))
            test_x = np.reshape(test_x, (n_test*2, -1))

        if to_tensor:
            return np_to_tensor([train_x, train_y, valid_x, valid_y, test_x, test_y])
        else:
            return train_x, train_y, valid_x, valid_y, test_x, test_y


def np_to_tensor(data_sets):
    new_sets = []
    for d in data_sets:
        new_sets.append(torch.from_numpy(d))
    return new_sets
