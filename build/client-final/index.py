import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import subprocess
import warnings
warnings.filterwarnings('ignore')

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
runcmd('curl http://gateway.openfaas:8080/function/server-1 -o server_1.txt', verbose=False)
runcmd('curl http://gateway.openfaas:8080/function/server-2 -o server_2.txt', verbose=False)
runcmd('curl http://gateway.openfaas:8080/function/server-3 -o server_3.txt', verbose=False)
runcmd('curl http://gateway.openfaas:8080/function/server-4 -o server_4.txt', verbose=False)

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class secondpart_bundleNet2(nn.Module):
    def __init__(self, num_classes=10, prior_percent=0.8, denoising=None, bundlenet=True):
        super(secondpart_bundleNet2, self).__init__()

        image_channels = 1

        
        self.denoising = denoising
        
        #self.inputlayer = nn.Conv2d(image_channels, 32, kernel_size=3, stride=1)

        self.inputlayer2 = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, stride=2, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inputlayer = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.convlayer1 = self.__make_convlayer(16,32)
        self.convlayer2 = self.__make_convlayer(32,32)
        self.convlayer3 = self.__make_convlayer(32,64)
        self.convlayer4 = self.__make_convlayer(64,128)
        
        ## resnet34: [3,4,6,3]
        ## resnet18: [2,2,2,2]
        self.inplanes = 64
        self.resnetlayer1 = self._make_resnetlayer(BasicBlock, 64, 1)
        self.resnetlayer2 = self._make_resnetlayer(BasicBlock, 128, 1, stride=2)
        self.resnetlayer3 = self._make_resnetlayer(BasicBlock, 256, 1, stride=2)
        self.resnetlayer4 = self._make_resnetlayer(BasicBlock, 512, 1, stride=2)
        
        self.before_fc = [self.inputlayer, self.convlayer1]

        self.flatten = Flatten()
        self.fc1 = self.__make_fclayer(1920, 128)
        self.fch = self.__make_fclayer(128, 128)

        self.fc2 = self.__make_fclayer_nodrop(128, num_classes)

        self.model = self.before_fc + [self.flatten,self.fc1,self.fc2]


        client_part = int(math.ceil(prior_percent * len(self.model)))
        if client_part < 1:
          client_part = 1
        if client_part >= len(self.model)-1:
          client_part = len(self.model)-1 # set at least one layer to server
        
        self.client_modules = self.model[0:client_part]
        self.server_modules = self.model[client_part:]
        
        
        self.customlayer = self.__make_fclayer(512, num_classes)
        self.server_modules[0] = self.customlayer
        
        
        
        self.normlayer1D = self.__make_norm1Dlayer(512)
        
        
        #print('self.input_dim: ',input_dim)
        #print('Setting last_modules: ',self.server_modules)


    def forward(self, x, scale=None):
        
        for layer in self.server_modules:
          #print('x.shape: ',x.size(), ' layer: ',layer)
          x = layer(x)
        output = F.log_softmax(x, dim=1) #next step, let bundle net do EVERYTHING, then aggregate w/ a small linear layer
        return output
    
    def __make_convlayer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ) 
    
    def __make_convlayer_nopool(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)
        )
    
    def __make_norm1Dlayer(self, in_channels):
        return nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
    def __make_norm2Dlayer(self, in_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels)
        )
    


    def _make_resnetlayer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)    
     
    
    def __make_fclayer(self, in_channels, out_channels, dropout=0):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels), 
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
    
        
    def __make_fclayer_nodrop(self, in_channels, out_channels, dropout=0):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )
        
    

##########################
### Resnet MODEL
##########################
# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-mnist.ipynb

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

def load_10fold_data(datadir, foldnum):
    path_data = datadir + '/mnist_10fold/X_train_fold'+str(foldnum)+'.pt'
    X_train = torch.load(datadir +'/mnist_10fold/X_train_fold'+str(foldnum)+'.pt')
    X_test = torch.load( datadir +'/mnist_10fold/X_test_fold'+str(foldnum)+'.pt')
    y_train = torch.load( datadir +'/mnist_10fold/y_train_fold'+str(foldnum)+'.pt')
    y_test = torch.load(datadir +'/mnist_10fold/y_test_fold'+str(foldnum)+'.pt')
    return X_train,X_test,y_train,y_test

class TrainDataset(torch.utils.data.Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

X_train,X_test,y_train,y_test = load_10fold_data(datadir, 0)
val_dataset = TrainDataset(X_test, y_test)
val_loader = torch.utils.data.DataLoader(val_dataset , batch_size=128, shuffle=False, drop_last=True)

f = open('./server_1.txt','r')
a = [', grad_fn=<SqueezeBackward1>']
lst = []
for line in f:
    for word in a:
        if word in line:
            line = line.replace(word,'')
    lst.append(line)
f.close()
f = open('./server_1.txt','w')
for line in lst:
    f.write(line)
f.close()

f = open('./server_2.txt','r')
a = [', grad_fn=<SqueezeBackward1>']
lst = []
for line in f:
    for word in a:
        if word in line:
            line = line.replace(word,'')
    lst.append(line)
f.close()
f = open('./server_2.txt','w')
for line in lst:
    f.write(line)
f.close()

f = open('./server_3.txt','r')
a = [', grad_fn=<SqueezeBackward1>']
lst = []
for line in f:
    for word in a:
        if word in line:
            line = line.replace(word,'')
    lst.append(line)
f.close()
f = open('./server_3.txt','w')
for line in lst:
    f.write(line)
f.close()

f = open('./server_4.txt','r')
a = [', grad_fn=<SqueezeBackward1>']
lst = []
for line in f:
    for word in a:
        if word in line:
            line = line.replace(word,'')
    lst.append(line)
f.close()
f = open('./server_4.txt','w')
for line in lst:
    f.write(line)
f.close()

f1 = open('server_1.txt')
subfeature1 = f1.read()
f1.close

f2 = open('server_2.txt')
subfeature2 = f2.read()
f2.close

f3 = open('server_3.txt')
subfeature3 = f3.read()
f3.close

f4 = open('server_4.txt')
subfeature4 = f4.read()
f4.close

from torch import Tensor as tensor

server1_output = eval(subfeature1)
server2_output = eval(subfeature2)
server3_output = eval(subfeature3)
server4_output = eval(subfeature4)

server_output_collection = torch.cat((server1_output,server2_output,server3_output,server4_output),1)
  
    
    ## client
client_function = torch.load('models/client_model_final.pt')
client_function.eval()

    # make final prediction
final_prediction = client_function(server_output_collection)




### Step 4: evaluate on client node 
target = torch.index_select(y_test, 1, torch.tensor([0]))
target = torch.reshape(target, (-1,))

test_loss = F.nll_loss(final_prediction, target, reduction='sum').item()  # sum up batch loss
pred = final_prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
correct = pred.eq(target.view_as(pred)).sum().item()/len(target)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}\n'.format(
    #     test_loss, correct))

runcmd('rm -rf models.tar.gz data_practice.tar.gz models data_practice server_1.txt server_2.txt server_3.txt server_4.txt', verbose=False)        
print('\nTest set: Average loss: {:.4f}, Accuracy: {}\n'.format(test_loss, correct))
