import torch
from torch import nn
import torchvision
import csv
import numpy as np
from torch import optim
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, D_in, H1, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, D_in)
        self.linear2 = nn.Linear(D_in, H1)
        self.linear3 = nn.Linear(H1, D_out)

    def forward(self, x):
        input_vec = x
        out = self.linear1(x)
        # out += input_vec
        out = self.linear2(out)
        out = torch.sigmoid(self.linear3(out))
        return out


def read_data():
    # read data
    data = np.load('temp_outputs/mnist_test.npz')
    return data['X_train'], data['Y_train'], data['X_test'], data['Y_test']


X_train, Y_train, X_test, Y_test = read_data()
n_examples = len(Y_train)
# Y_train = np.zeros(n_examples, )
Y_train = np.ndarray([[0 if y != i else 1 for i in range(10)] for y in Y_train])
Y_test = np.ndarray([[0 if y != i else 1 for i in range(10)] for y in Y_test])

Y_train = Y_train.astype(np.float64)
Y_test = Y_test.astype(np.float64)
X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)
X_test = torch.from_numpy(X_test)
Y_test = torch.from_numpy(Y_test)
D_in = X_train.shape[1]
H1 = 100
D_out = len(np.unique(Y_test))

net = Net(D_in, H1, D_out).double()
optimizer = optim.Adam(net.parameters(), lr=1e-2)
criterion = nn.MSELoss()

train_losses = []
validation_losses = []
min_valid_loss = np.inf
best_model_state = net.state_dict()
# train
n_examples = len(Y_train)
n_valid = int(n_examples / 50)
n_epoch = 2
for e in range(n_epoch):
    print(e + 1)
    train_loss_epoch = []
    perm = torch.randperm(n_examples)
    indices_valid = perm[:50]
    indices_train = perm[50:100]
    net.train()
    for x, y in zip(X_train[indices_train], Y_train[indices_train]):
        optimizer.zero_grad()  # zero the gradient buffers
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss_epoch.append(loss.item())
    print('done train epoch')
    train_losses.append(np.mean(train_loss_epoch))
    net.eval()
    eval_output = net(X_train[indices_valid])
    eval_output = eval_output.view(eval_output.numel())
    validation_loss_epoch = torch.tensor.mean(criterion(eval_output, Y_train[indices_valid]).item())
    if validation_loss_epoch < min_valid_loss:
        min_valid_loss = validation_loss_epoch
        best_model_state = net.state_dict()
    validation_losses.append(validation_loss_epoch)

torch.save(best_model_state, 'temp_outputs/fc_cam_model')
# net = Net(D_in, H1, D_out)
# net.load_state_dict(torch.load('temp_outputs/fc_cam_model'))
net.eval()
test_result = net(X_test)
eval_output = test_result.view(test_result.numel())
test_loss = torch.tensor.mean(criterion(test_result, Y_train).item())

print(test_loss)
plt.figure()
plt.plot(train_losses, label='train losses')
plt.plot(validation_losses, label='validation losses')
plt.show()