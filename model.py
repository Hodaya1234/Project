import torch
import numpy as np
import torch.utils.data as data_utils
from torch import optim


class SimpleNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(SimpleNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        return x


class DataSet(data_utils.Dataset):
    def __init__(self, x, y):
        super(DataSet, self).__init__()
        self.all_x = x
        self.all_y = y

    def __getitem__(self, index):
        x = self.all_x[index]
        y = self.all_y[index]
        return x, y

    def __len__(self):
        return len(self.all_y)


# upload data as numpy ndarray and convert to tensor
train_x = torch.from_numpy(np.load('train_x.npy'))
train_y = torch.from_numpy(np.load('train_y.npy'))
test_x = torch.from_numpy(np.load('test_x.npy'))
test_y = torch.from_numpy(np.load('test_y.npy'))
test_x_real = torch.from_numpy(np.load('test_x_real.npy'))
test_y_real = torch.from_numpy(np.load('test_y_real.npy'))

# save the new data as tensors to dave the conversion time
torch.save(train_x, 'train_x')
torch.save(train_y, 'train_y')
torch.save(test_x, 'test_x')
torch.save(test_y, 'test_y')
torch.save(test_x_real, 'test_x_real')
torch.save(test_y_real, 'test_y_real')

# upload the tensor data
train_x = torch.load('train_x')             # size: n_train_examples X n_segments X n_frames
train_y = torch.load('train_y').float()     # size: n_train_examples
test_x = torch.load('test_x')               # size: n_test_examples X n_segments X n_frames
test_y = torch.load('test_y')  .int()       # size: n_test_examples
test_x_real = torch.load('test_x_real')     # size: n_real_data_examples X n_segments X n_frames
test_y_real = torch.load('test_y_real').int()  # size: n_real_data_examples

# get the dimensions
n_train_examples, n_segments, n_frames = train_x.shape
n_test_examples = test_x.shape[0]
n_real = test_x_real.shape[0]

# flatten all the x data:
train_x = train_x.view(n_train_examples, -1)
test_x = test_x.view(n_test_examples, -1)
test_x_real = test_x_real.view(n_real, -1)

# create the dataset class and data loader:
train_dataset = DataSet(train_x, train_y)
train_loader = data_utils.DataLoader(train_dataset, batch_size=8, shuffle=True)

# create a basic FC network:
D_in = train_x.shape[1]
H1 = 50
H2 = 20
H3 = 5
D_out = 1

model = SimpleNet(D_in, H1, H2, H3, D_out).double()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss(reduction='sum')

n_epochs = 100
for e in range(n_epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(x).float()
        y_pred = y_pred.view(y_pred.numel())
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

    if e % 5 == 0:
        # accuracy on the real data:
        correct = 0
        outputs = model(test_x_real)
        predictions = torch.round(outputs).int().view(outputs.numel())
        correct += (predictions == test_y_real).sum().item()
        print('accuracy on the real data: {0:.2f}'.format(correct / n_real))

        # accuracy on the augmented data:
        correct = 0
        outputs = model(test_x)
        predictions = torch.round(outputs).int().view(outputs.numel())
        correct += (predictions == test_y).sum().item()
        print('accuracy on the augmented data: {0:.2f}'.format(correct / n_test_examples))
