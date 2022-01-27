
import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset   

# import class
from lenet5 import LeNet
from utils import client_update,server_aggregate, test

##### Hyperparameters for federated learning #########
# DATA TO CHANGE/JUSTIFY
num_clients = 20 # number of total clients
num_selected = 6 # number of  clients selected for the training 
num_rounds = 10
epochs = 5
batch_size = 32

#############################################################
##### Creating desired data distribution among clients  #####
#############################################################

# Image augmentation 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Loading CIFAR10 using torchvision.datasets
traindata = datasets.CIFAR10('./data', train=True, download=False,
                       transform= transform_train)

# Dividing the training data into num_clients, with each client having equal number of images
traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])

# Creating a pytorch loader for a Deep Learning model
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

# Normalizing the test images
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Loading the test iamges and thus converting them into a test_loader
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        ), batch_size=batch_size, shuffle=True)


################ start ###############
modelChosen = LeNet()



### initializing models 

centralizedModel = modelChosen

federatedModels= [modelChosen for _ in range(num_selected)] # why num_selected and not num_clients

for models in federatedModels:
    models.load_state_dict(centralizedModel.state_dict()) # initial synchronization with centralised model


# optimizers
# we will use SGD /  TAKE INTO ACCOUNT THAT MAYBE ADAM is better
LR = 1e-5 # with lr 1e-5 seems like learns a bit
opt = [optim.SGD(models.parameters(), lr=LR) for models in federatedModels]

###### training #######
###### List containing info about learning #########
losses_train = []
losses_test = []
acc_train = []
acc_test = []
# Runnining FL

for r in range(num_rounds):
    # select random clients
    client_idx = np.random.permutation(num_clients)[:num_selected]
    # client update
    loss = 0
    for i in tqdm(range(num_selected)):
        loss += client_update(federatedModels[i], opt[i], train_loader[client_idx[i]], epoch=epochs)
    
    losses_train.append(loss)
    # server aggregate
    server_aggregate(centralizedModel, federatedModels)
    
    test_loss, acc = test(centralizedModel, test_loader)
    losses_test.append(test_loss)
    acc_test.append(acc)
    print('%d-th round' % r)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))

