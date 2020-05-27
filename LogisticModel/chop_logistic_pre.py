import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable     
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import sys
import getopt

encoding = {'A':np.array([1,0,0,0]),
            'C':np.array([0,1,0,0]),
            'G':np.array([0,0,1,0]),
            'T':np.array([0,0,0,1])}

def one_hot(guide,encoding):
    data = np.zeros((4,len(guide)))
    assert data.shape == (4,23)
    for i in range(data.shape[-1]):
        data[:,i] = encoding[guide[i]]
    return data

def batch_one_hot(data,encoding):
    guides = np.zeros((len(data),4,23))
    i=0
    for guide in data['GUIDE']:
        guides[i] = one_hot(guide,encoding)
        i+=1
    return guides

class GGEDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample.float()

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(92, 60),
            nn.ReLU(),
            nn.Linear(60, 40),
            nn.ReLU(),
            nn.Linear(40, 30)
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU(),
            nn.Linear(40, 60),
            nn.ReLU(),
            nn.Linear(60, 92),
            nn.Sigmoid()
        )
        
        self.conv1 = nn.Conv2d(1, 6, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.ConvTranspose2d(6, 1, 2)
        self.pool2 =nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        x = x.view(-1,1*4*23)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1,1,4,23)

        return x

class LogisticModel(nn.Module):
    def __init__(self,in_dim, n_class):
        super(LogisticModel, self).__init__()
        self.l1 = torch.nn.Linear(in_dim, 20)
        self.l2 = torch.nn.Linear(20, 10)
        self.l3 = torch.nn.Linear(10, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.l1(x))
        x = self.sigmoid(self.l2(x))
        y_pred = self.sigmoid(self.l3(x))
        return y_pred


def ImportAT(modelpath = 'model/autoencoder.pth.tar'):
    # import autoencoder
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Autoencoder().to(device)
    checkpoint = torch.load(modelpath)
    net.load_state_dict(checkpoint['state_dict'])
    
    return net

def ImportLog(modelpath = 'model/choplogistic.pth.tar'):
    # import logistic model
    model = LogisticModel(30,1) 
    # Construct loss function and optimizer
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # import model
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['state_dict'])
    return model

#==============================================================================
# Command line processing
class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], "h",["guide=","output=", "guide_file="])
        opts = dict(opts)
        self.exit = True
        self.output = 'predictions.csv'
        if '-h' in opts:
            self.printHelp()
            return

        if len(args) > 0:
            print("*** ERROR: no arg files - only options! ***", file=sys.stderr)
            self.printHelp()
            return
        
        if '--guide_file' in opts:
            self.readfile = True
            self.filepath = opts['--guide_file']
            self.exit = False
            if '--output' in opts:
                self.output = opts['--output']

        elif '--guide' in opts:
            self.readfile = False
            self.guide = opts['--guide']
            self.exit = False
        else:
            self.printHelp()
            return
        
    def printHelp(self):
        print("xxx.py h [--guide_file filepath --output output_file_name --guide guide_sequence]\n")
        print("where:\n")
        print("\t h                  print help\n")
        print("\t[--guide_file filepath]    Optionally specifies the csv guide filepath 'filepath' to perform. \n")
        print("\t[--output outputpath]    Optionally specify the output file path when reading csv file.\n")
        print("\t[--guide guide_sequence]   Optionally specifies a guide sequence to get a prediction.\n")
        print("you must choose only one command from [--guide_file filepath] and [--guide guide_sequence].\n")

      
if __name__ == '__main__':
    config = CommandLine()
    if config.exit == False:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if config.readfile == True:
            filepath = config.filepath
            chop_data = pd.read_csv(filepath, index_col=0)
            guides = batch_one_hot(chop_data, encoding)
            GGE_dataset = GGEDataset(data = guides, transform = transforms.Compose([transforms.ToTensor()]))
            Dataloader = torch.utils.data.DataLoader(GGE_dataset, batch_size=len(GGE_dataset),shuffle=False, num_workers=0)
            net = ImportAT(modelpath = 'model/autoencoder.pth.tar')
            with torch.no_grad():
                for data in Dataloader:
                    da_fea = net.encoder(data.view(-1,1*4*23).to(device)).to('cpu')
            model = ImportLog(modelpath = 'model/choplogistic.pth.tar')
            chop_pre = model(da_fea)
            chop_data['prediction']=chop_pre.detach().numpy().reshape(1,-1)[0].tolist()
            chop_pre_cla = lambda x: 0 if x<=0.5 else 1
            chop_data["prediction"]=chop_data.prediction.apply(chop_pre_cla)
            chop_data.to_csv(config.output, sep=',')
        elif config.readfile == False:
            guides = config.guide
            one_hot_guide = one_hot(guides,encoding)
            transform = transforms.Compose([transforms.ToTensor()])
            tensor_guide = transform(one_hot_guide).float()
            net = ImportAT(modelpath = 'model/autoencoder.pth.tar')
            auto_fea = net.encoder(tensor_guide.view(-1,1*4*23).to(device)).to('cpu')
            model = ImportLog(modelpath = 'model/choplogistic.pth.tar')
            pre = model(auto_fea)
            pre = pre.detach().numpy().reshape(1,-1)[0][0].tolist()
            pre_label = 1 if pre > 0.5 else 0
            print('the predicted label for sequence %s is %d'% (guides, pre_label))

            
