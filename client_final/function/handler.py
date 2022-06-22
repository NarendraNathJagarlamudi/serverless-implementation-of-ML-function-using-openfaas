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
    
    ### Step 3: configure the function in client, this function must be run after the server calculation finishes

    #  collect results from each server 
    # server1 = runcmd('curl http://127.0.0.1:8080/function/server-1 -o server_1.txt', verbose=False)
    # server2 = runcmd('curl http://127.0.0.1:8080/function/server-2 -o server_2.txt', verbose=False)
    # server3 = runcmd('curl http://127.0.0.1:8080/function/server-3 -o server_3.txt', verbose=False)
    # server4 = runcmd('curl http://127.0.0.1:8080/function/server-4 -o server_4.txt', verbose=False)

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
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=False)

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

        
    return print('\nTest set: Average loss: {:.4f}, Accuracy: {}\n'.format(
        test_loss, correct))


print(handle())