from pkg_resources import get_distribution
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from lenet5 import LeNet
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import random 

from utilsfedv2 import *
from lenet5 import LeNet



# Check if cuda is available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Training Parameters
LEARNING_RATE = 2e-3  # LR
BATCH_SIZE = 32
ROUNDS = 10  # R
LOCAL_EPOCHS = 5  # E
NUM_CLIENTS = 10  # K: number of total clients
C = 0.3  # percentage of clients selected at each round
# m = C * K : number of  clients selected at each round
NUM_SELECTED = max(int(C * NUM_CLIENTS), 1)

# Save plots in the folder ./plots or show them
SAVE_PLOTS = True
# If the clients have different numbers of images or not
DIFFERENT_SIZES = False 

# Use batch normalization or not
BATCH_NORM = False
# group normalization
GROUP_NORM = True

# group normalization parameters
groupNormParams= {
'groupNL1' : 2,
'groupNL2' :2
}


if GROUP_NORM ==True & BATCH_NORM ==True:
    print(" Cannot have group an batch normalization True at the same time")
    exit()

# Image augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalizing the test images
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Loading CIFAR10 using torchvision.datasets
traindata = datasets.CIFAR10('./data', train=True, download=False,
                             transform=transform_train)
# print(traindata)
total_data = traindata.data.shape[0] # number of data
if DIFFERENT_SIZES:
    # Dividing the training data into num_clients, with each clients having different number of images
    delta = 500 # controls how much the images numbers can vary from client to client
    min_val = max(int(total_data/ NUM_CLIENTS) - delta, 1) # min value of number of images for each client
    max_val = min(int(total_data/ NUM_CLIENTS) + delta, total_data - 1) # max value of number of images for each client

    indices = list(range(NUM_CLIENTS)) # list of indices for the splits of the data
    lengths = [random.randint(min_val,max_val) for i in indices] # List of lengths of splits to be produced

    diff = sum(lengths) - total_data # we are off by this abount 

    # Iterate through, incrementing/decrementing a random index 
    while diff != 0:  
        addthis = 1 if diff > 0 else -1 # +/- 1 depending on if we were above or below target.
        diff -= addthis

        idx = random.choice(indices) # Pick a random index to modify, check if it's OK to modify
        while not (min_val < (lengths[idx] - addthis) < max_val): 
            idx = random.choice(indices) # Not OK to modify.  Pick another.

        lengths[idx] -= addthis #Update that index.
    
    print("Number of Images for each client:")
    print(lengths)
    
    traindata_split = torch.utils.data.random_split(traindata, lengths)

else:
    # Dividing the training data into num_clients, with each client having equal number of images
    traindata_split = torch.utils.data.random_split(traindata, [int(total_data/ NUM_CLIENTS) for _ in range(NUM_CLIENTS)])



# Creating a pytorch loader for a Deep Learning model
train_loader = [torch.utils.data.DataLoader(
    x, batch_size=BATCH_SIZE, shuffle=True) for x in traindata_split]


# Loading the test iamges and thus converting them into a test_loader
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=False,
                                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                                                           ), batch_size=BATCH_SIZE, shuffle=True)



model = LeNet(BATCH_NORM,GROUP_NORM,groupNormParams).to(DEVICE)
centralizedModel = model

# list of models, model per device SELECTED ( same model for each device in our case)
federatedModels = [model for _ in range(NUM_SELECTED)]

for models in federatedModels:
    # we initialize every model with the central
    models.load_state_dict(centralizedModel.state_dict())


optimizers = [torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
              for model in federatedModels]
criterion = nn.CrossEntropyLoss()


centralizedModel, federatedModels, optimizers, (train_losses, valid_losses), (train_accuracies, valid_accuracies) = training_loop(
    centralizedModel, federatedModels, criterion, optimizers, train_loader, test_loader, ROUNDS, LOCAL_EPOCHS, NUM_CLIENTS, NUM_SELECTED, DEVICE)
