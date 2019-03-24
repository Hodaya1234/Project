import numpy as np
import augment
import torch
from data_set import DataSet


def turn_to_torch_dataset(data_sets, cv=True):
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


def get_train_test_indices(n_total, n_train, n_test):
    total_indices = np.random.choice(n_total, n_train + n_test, replace=False)
    test_indices_from_total = np.random.choice(n_train + n_test, n_test, replace=False)
    test_indices = total_indices[test_indices_from_total]
    train_indices = np.delete(total_indices, test_indices_from_total)
    return train_indices, test_indices


def get_data(seg_v, seg_h, n_train=3000, n_valid=50, n_test=2, cv=True, flat_x=True, to_tensor=True):
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
    :return: A list of the three data-sets, including the y targets.
    """
    if cv:
        train_sets_x = []
        validation_sets_x = []
        test_sets_x = []
        n_v = seg_v.shape[2]
        n_h = seg_h.shape[2]
        min_original_set = min(n_v, n_h)
        train_set_size = min_original_set - n_test
        for i in range(min(n_v, n_h)):
            train_indices_v, test_indices_v = get_train_test_indices(n_v, train_set_size, n_test)
            train_indices_h, test_indices_h = get_train_test_indices(n_h, train_set_size, n_test)
            if n_test == 1:
                test_v = np.expand_dims(seg_v[:, :, test_indices_v], axis=0)
                test_h = np.expand_dims(seg_h[:, :, test_indices_h], axis=0)
            else:
                test_v = np.transpose(seg_v[:, :, test_indices_v], (2, 0, 1))
                test_h = np.transpose(seg_h[:, :, test_indices_h], (2, 0, 1))

            test_x = np.concatenate((test_v, test_h), 0)
            print('finished test number' + str(i))

            param_v = augment.get_parameters(seg_v[:, :, train_indices_v])
            curr_train_v = augment.get_new_data(param_v, n_train)
            curr_valid_v = augment.get_new_data(param_v, n_valid)

            param_h = augment.get_parameters(seg_h[:, :, train_indices_h])
            curr_train_h = augment.get_new_data(param_h, n_train)
            curr_valid_h = augment.get_new_data(param_h, n_valid)

            train_x = np.concatenate((curr_train_v, curr_train_h), 0)
            valid_x = np.concatenate((curr_valid_v, curr_valid_h), 0)
            print('finished train number' + str(i))
            if flat_x:
                train_x = np.reshape(train_x, (n_train * 2, -1))
                valid_x = np.reshape(valid_x, (n_valid * 2, -1))
                test_x = np.reshape(test_x, (n_test * 2, -1))
            if to_tensor:
                train_x, valid_x, test_x = np_to_tensor([train_x, valid_x, test_x])
            train_sets_x.append(train_x)
            validation_sets_x.append(valid_x)
            test_sets_x.append(test_x)
        train_y = np.concatenate((np.ones(n_train), np.zeros(n_train)))
        valid_y = np.concatenate((np.ones(n_valid), np.zeros(n_valid)))
        test_y = [1, 0]
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
