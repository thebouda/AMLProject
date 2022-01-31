# IMPLEMENTED USING https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320

import torch
import torch.nn as nn

N_CLASSES = 10

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()

        # input 32x32x1
        #C1 : convolutional layer with 6 kernels of size 5x5 and stride of 1
        # 28x28x6
        #S2 : pooling layer with 6 kernels of size 2x2 and stride of 2
        # 14x14x6
        #C3 : convolutional layer with 16 kernels of size 5x5 and stride of 1
        # 10x10x16
        #S4 : pooling layer with 16 kernels of size 2x2 and stride of 2
        # 5x5x16
        #C5 : convolutional layer with 120 kernels of size 5x5 => Basically a fully connected layer
        # 1x1x120
        #F6 : layer 84 (fully-connected) (tanh ?)
        # 1x1x84 
        #F7 : dense layer with 10 outputs
        # output 10x1
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120,out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84,out_features=N_CLASSES)
        )

    def forward(self,x):
        x = self.feature_extractor(x)
        x = torch.flatten(x,1)
        logits = self.classifier(x)
        probs = torch.softmax(logits,dim=1)
        return logits,probs