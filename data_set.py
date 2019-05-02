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

    def normalize(self, mean_x=None, std_x=None, simple=False):
        if simple:
            return DataSet(torch.sub(self.all_x, 1), self.all_y)
        if mean_x is None:
            mean_x = torch.mean(self.all_x, dim=0).repeat(self.n_data, 1)
        else:
            mean_x = mean_x.repeat(self.n_data, 1)
        if std_x is None:
            std_x = torch.std(self.all_x, dim=0).repeat(self.n_data, 1)
        else:
            std_x = std_x.repeat(self.n_data, 1)
        return DataSet(torch.div(torch.sub(self.all_x, mean_x), std_x), self.all_y)

    def calc_mean_std(self):
        return torch.mean(self.all_x, dim=0), torch.std(self.all_x, dim=0)


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
