import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import subprocess


def handle():
    
    def runcmd(cmd, verbose = False, *args, **kwargs):

        process = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            shell = True
        )
        std_out, std_err = process.communicate()
        
    torch.set_printoptions(profile="full")
    torch.manual_seed(1)
    device = torch.device("cpu")
    modeldir = './models'
    datadir = './data_practice'

    os.makedirs(datadir, exist_ok=True) 
    os.makedirs(modeldir, exist_ok=True)

    runcmd('wget https://cs.slu.edu/~hou/downloads/PropertyInferenceAttack/data_practice.tar.gz --no-check-certificate', verbose = False)
    runcmd('wget https://cs.slu.edu/~hou/downloads/PropertyInferenceAttack/models.tar.gz --no-check-certificate', verbose = False)
    runcmd('tar -zxf data_practice.tar.gz', verbose = False)
    runcmd('tar -zxf models.tar.gz', verbose=(False))
    
    
  
    
    # os.system("wget https://cs.slu.edu/~hou/downloads/PropertyInferenceAttack/data_practice.tar.gz --no-check-certificate >/dev/null")
    # os.system("tar -zxf data_practice.tar.gz >/dev/null")
    # os.system("wget https://cs.slu.edu/~hou/downloads/PropertyInferenceAttack/models.tar.gz --no-check-certificate >/dev/null")
    # os.system("tar -zxf models.tar.gz >/dev/null")


    def load_10fold_data(datadir, foldnum):
        path_data = datadir + '/mnist_10fold/X_train_fold'+str(foldnum)+'.pt'
        X_train = torch.load(datadir +'/mnist_10fold/X_train_fold'+str(foldnum)+'.pt')
        X_test = torch.load( datadir +'/mnist_10fold/X_test_fold'+str(foldnum)+'.pt')
        y_train = torch.load( datadir +'/mnist_10fold/y_train_fold'+str(foldnum)+'.pt')
        y_test = torch.load(datadir +'/mnist_10fold/y_test_fold'+str(foldnum)+'.pt')
        return X_train,X_test,y_train,y_test



# Create PyTorch Datasets from SkLearn output
    class TrainDataset(torch.utils.data.Dataset):
    
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data
        
        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]
        
        def __len__ (self):
            return len(self.X_data)

    class Flatten(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return x.view(batch_size, -1)

    """**(6) define test datasets**"""

    X_train,X_test,y_train,y_test = load_10fold_data(datadir, 0)
    val_dataset = TrainDataset(X_test, y_test)
    val_loader = torch.utils.data.DataLoader(val_dataset , batch_size=128, shuffle=False, drop_last=True)



    subfeature4 = X_test[:,:,14:28,0:28]
        
    return subfeature4
