import torch
import torch.nn as nn
import fedAvg
from torchvision import datasets, transforms
from lenet5_ import LeNet
from datetime import datetime

# check for cuda
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
N_CLIENTS = 3

def train(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for X, y_target in train_loader:
        
        # set gradient to zero
        optimizer.zero_grad()

        # if there is a GPU

        # X = X.to(device)
        # y_true = y_true.to(device)

        # prediction

        # call model forward()
        y_predict, _ = model(X)
        # get loss
        loss = criterion(y_predict, y_target)
        running_loss += loss.item() * X.size(0)
        
        # adjusting weights
        loss.backward()
        optimizer.step()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

def test(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0

    for X, y_target in valid_loader:
        # if there is a GPU

        # X = X.to(device)
        # y_true = y_true.to(device)

        # prediction and loss

        # call model forward()
        y_predict, _ = model(X)
        # get loss
        loss = criterion(y_predict, y_target)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss

def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def training_loop(model, criterion, optimizer, train_loader, test_loader,
                epochs, device, print_every=1):

    train_losses = []
    valid_losses = []

    # train model
    for epoch in range(epochs):
        model, optimizer, train_loss = train(train_loader, model,
                                criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation

        # disable gradient calculation to save memory
        with torch.no_grad():
            model, valid_loss = test(test_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device)
            test_acc = get_accuracy(model, test_loader, device)

            print(f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * test_acc:.2f}')

    return model, optimizer, (train_losses, valid_losses)

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
#traindata = datasets.CIFAR10('./data', train=True, download=False,
#                       transform= transform_train)
traindata = datasets.CIFAR10('./data', train=True, download=False, 
                transform=transform_train)

# # Dividing the training data into num_clients, with each client having equal number of images
# traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] 
#                     / num_clients) for _ in range(num_clients)])
# Creating a pytorch loader for a Deep Learning model
train_loader = torch.utils.data.DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True) 

# Loading the test iamges and thus converting them into a test_loader
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data', train=False, 
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            ), batch_size=BATCH_SIZE, shuffle=True)

model = LeNet()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
#criterion = nn.nll_loss()

# model, optimizer, (train_losses, valid_losses) = training_loop(model, criterion, optimizer,
#                         train_loader, test_loader, EPOCHS, DEVICE)

model_dict, optimizer_dict, criterion_dict = fedAvg.create_clients_model(N_CLIENTS, LEARNING_RATE)

# train
for i in range(N_CLIENTS):
    model_dict[str(i)], optimizer_dict[str(i)], (train_losses, valid_losses) = training_loop(model_dict[str(i)], criterion_dict[str(i)], 
            optimizer_dict[str(i)], train_loader, test_loader, EPOCHS, DEVICE)

model = fedAvg.update_main_model(model, model_dict, N_CLIENTS)
with torch.no_grad():
    model, valid_loss = test(test_loader, model, criterion, DEVICE)
    test_acc = get_accuracy(model, test_loader, DEVICE)
    print(f'Valid loss: {valid_loss:.4f}\t'f'Valid accuracy: {100 * test_acc:.2f}')