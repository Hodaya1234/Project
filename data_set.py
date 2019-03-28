import torch.utils.data as data_utils
import torch


class DataSet(data_utils.Dataset):
    def __init__(self, x, y):
        super(DataSet, self).__init__()
        self.all_x = x
        self.all_y = y
        self.n_data = len(y)

    def __getitem__(self, index):
        x = self.all_x[index]
        y = self.all_y[index]
        return x, y

    def __len__(self):
        return len(self.all_y)

    def change_device(self, device):
        return DataSet(self.all_x.to(device), self.all_y.to(device))

    def normalize(self):
        mean_x = self.all_x.mean().item()
        std_x = self.all_x.std().item()
        return DataSet(torch.div(torch.sub(self.all_x, mean_x), std_x), self.all_y)


def data_to_cuda(data_sets, device, cv=True):
    new_data = []
    if cv:
        for set_type in data_sets:
            set_type_list = []
            for one_dataset in set_type:
                set_type_list.append(one_dataset.change_device(device))
            new_data.append(set_type_list)
    else:
        for s in data_sets:
            new_data.append(s.change_device(device))
    return new_data


def normalize_datasets(data_sets, cv=True):
    new_data = []
    if cv:
        for set_type in data_sets:
            set_type_list = []
            for one_dataset in set_type:
                set_type_list.append(one_dataset.normalize())
            new_data.append(set_type_list)
    else:
        for s in data_sets:
            new_data.append(s.normalize())
    return new_data
