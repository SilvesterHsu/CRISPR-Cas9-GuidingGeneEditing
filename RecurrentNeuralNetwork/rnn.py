import torch
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

train_data0 = pd.read_csv('data/danrer11_chopchop_train.csv')
test_data = pd.read_csv('data/danrer11_chopchop_test.csv')

print("loading successful!")

# train validation split
train_data = train_data0[:200000]
val_data = train_data0[200000:]

def transform_sequence(seq):
    m = np.zeros((len(seq), 4))
    for i, char in enumerate(seq):
        if char == 'A':
            m[i][0] = 1
        elif char == 'T':
            m[i][1] = 1
        elif char == 'C':
            m[i][2] = 1
        elif char == 'G':
            m[i][3] = 1
    m = m.reshape(m.shape[0]*m.shape[1])
    return m

def transform_sequence_rnn(seq):
    m = np.zeros((len(seq), 4))
    for i, char in enumerate(seq):
        if char == 'A':
            m[i][0] = 1
        elif char == 'T':
            m[i][1] = 1
        elif char == 'C':
            m[i][2] = 1
        elif char == 'G':
            m[i][3] = 1
    return m

class GeneDataset(object):
    def __init__(self, data, use_rnn=False):
        self.target_sequence = data['GUIDE'].values
        self.efficiency = data['EFFICIENCY'].values
        self.use_rnn = use_rnn
        self.seqs = [torch.as_tensor(transform_sequence_rnn(i), dtype=torch.float32) for i in  self.target_sequence ]
        self.effs = [torch.as_tensor(i / 100, dtype=torch.float32) for i in self.efficiency]

    def __getitem__(self, idx):
        if self.use_rnn:
            #seq = torch.as_tensor(transform_sequence_rnn(self.target_sequence[idx]), dtype=torch.float32)
            #seq = torch.as_tensor(self.seqs[idx], dtype=torch.float32)
            seq = self.seqs[idx]
        else:
            seq = torch.as_tensor(transform_sequence(self.target_sequence[idx]), dtype=torch.float32)
        #eff = torch.as_tensor(self.efficiency[idx] / 100, dtype=torch.float32)
        eff = self.effs[idx]
        return seq, eff

    def __len__(self):
        return len(self.target_sequence)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(92, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

class RNN_Net(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(RNN_Net, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 1)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x1, _ = self.lstm(x)
        # Extract the mean of output channal as the final output
        #x1 = nn.Dropout(p=self.dropout_prob)(x1)
        x2 = torch.mean(x1, 1)
        # Normalize the output using sigmoid to (0, 1)
        x3 = torch.sigmoid(x2)

        return x3


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
#net.train()
#net = net.to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9)

train_set = GeneDataset(train_data, use_rnn=True)
val_set = GeneDataset(val_data, use_rnn=True)
print("Creating dataset successful!")
trainloader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_set, batch_size=256,
                                          shuffle=True, num_workers=2)

training_loss_history = []
validation_loss_history = []
for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, ele in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        seq, eff = ele[0].to(device), ele[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(seq)
        loss = criterion(outputs[:, 0], eff)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(trainloader)
    training_loss_history.append(train_loss)

    running_loss = 0.0
    for i, ele in enumerate(valloader):
        seq, eff = ele[0].to(device), ele[1].to(device)
        outputs = net(seq)
        loss = criterion(outputs[:, 0], eff)
        running_loss += loss.item()

    val_loss = running_loss / len(valloader)
    validation_loss_history.append(val_loss)


    print("Epoch {} , training loss: {:3f}, validation loss: {:3f}".format(epoch, train_loss, val_loss))

print('Finished Training')
# train_time[8] = time.time() - now
PATH = "nn.pth"
torch.save(net.state_dict(), PATH)

mse = 0
mae = 0
mse_error = []
eff_list = []

test_set = GeneDataset(test_data, use_rnn=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=True, num_workers=2)
criterion2 = torch.nn.L1Loss().to(device)
#net.eval()

rmae = 0

for i, ele in enumerate(testloader):
        # get the inputs; data is a list of [inputs, labels]
    seq, eff = ele[0].to(device), ele[1].to(device)

    # forward + backward + optimize
    outputs = net(seq)
    print(outputs[:, 0], eff)
    e = criterion(outputs[:, 0], eff)
    e2 = criterion2(outputs[:, 0], eff)
    rmae += np.sum(np.abs(outputs[:, 0].cpu().detach().numpy() / eff.cpu().detach().numpy()))
    mse += e * len(eff)
    mae += e2 * len(eff)

mse = mse / len(test_set)
mae = mae / len(test_set)
rmae = rmae / len(test_set)
print("MSE:", mse.cpu().detach().numpy())
print("MAE:", mae.cpu().detach().numpy())
print("RMAE:", rmae)

