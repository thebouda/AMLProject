from pkg_resources import get_distribution
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from lenet5 import LeNet
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from utilsfedv2 import *
from lenet5 import LeNet



# Check if cuda is available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Training Parameters
LEARNING_RATE = 2e-3  # LR
BATCH_SIZE = 32
ROUNDS = 10  # R
LOCAL_EPOCHS = 10  # E
NUM_CLIENTS = 20  # K: number of total clients
C = 0.3  # percentage of clients selected at each round
# m = C * K : number of  clients selected at each round
NUM_SELECTED = max(int(C * NUM_CLIENTS), 1)

# Save plots in the folder ./plots or show them
SAVE_PLOTS = True

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

# divide the dataset in partitions
xTrain = np.array(traindata.data)
yTrain =np.array(traindata.targets)

N = yTrain.shape[0]


# Dividing the training data into num_clients, with each client having equal number of images
traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0]
                                                                / NUM_CLIENTS) for _ in range(NUM_CLIENTS)])



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
