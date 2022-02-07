import numpy as np
import torch

def train(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for X, y_target in train_loader:

        # set gradient to zero
        optimizer.zero_grad()

        # If there is a GPU, pass the data to GPU
        X = X.to(device)
        y_target = y_target.to(device)

        # Prediction

        # Call model forward()
        y_predict, _ = model(X)

        # Get loss
        loss = criterion(y_predict, y_target)
        running_loss += loss.item() * X.size(0)

        # Adjusting weights
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def test(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0

    for X, y_target in valid_loader:

        # If there is a GPU, pass the data to the GPU
        X = X.to(device)
        y_target = y_target.to(device)

        # Prediction and loss

        # Call model forward()
        y_predict, _ = model(X)

        # Get loss
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


def client_update(model, optimizer, train_loader, device, criterion, epochs):
    """
    This function updates/trains client model on client data
    """
    for e in range(epochs):
        model, optimizer, train_loss = train(train_loader, model,
                                             criterion, optimizer, device)
    return train_loss


def server_aggregate(global_model, client_models, lengths):
    """
    This function has aggregation method 'mean'
    """
    # This will take simple mean of the weights of models

    totLength= float(sum(lengths))
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        globDict = 0
        for i in range (len(client_models)):
            globDict += client_models[i].state_dict()[k].float() * float(lengths[i]) / totLength
        global_dict[k] =globDict
            

    global_model.load_state_dict(global_dict)

    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def training_loop(centralizedModel, federatedModels, criterion, optimizers, train_loader, test_loader,
                  rounds, epochs, num_clients, num_selected, device, print_every=1):

    global_train_losses = []  # Average train losses between clients
    global_valid_losses = []  # Average validation losses between clients

    global_train_accuracies = []  # Average train accuracies between clients
    global_valid_accuracies = []  # Average validation accuracies between clients

    # Train model
    for round in range(rounds):

        # Select random clients
        # Select in the total number of clients, a random array of clients of size num_selected at each round
        client_idx = np.random.permutation(num_clients)[:num_selected]

        local_train_losses = []  # Local train losses of the clients in this round
        local_valid_losses = []  # Local validation losses of the clients in this round

        local_train_accuracies = []  # Local train accuracies of the clients in this round
        # Local validation accuracies of the clients in this round
        local_valid_accuracies = []
        local_len = []

        for i in range(num_selected):
            # Train federated model locally in client i for num_epochs epochs
            local_train_loss = client_update(
                federatedModels[i], optimizers[i], train_loader[client_idx[i]], device, criterion, epochs)
            local_train_acc = get_accuracy(
                federatedModels[i], train_loader[client_idx[i]], device)

            local_train_losses.append(local_train_loss)
            local_train_accuracies.append(local_train_acc)

            local_valid_loss = test(
                test_loader, federatedModels[i], criterion, device)[1]
            local_valid_acc = get_accuracy(
                federatedModels[i], test_loader, device)

            local_valid_losses.append(local_valid_loss)
            local_valid_accuracies.append(local_valid_acc)
            lenDataLoad = len(train_loader[client_idx[i]]) # number of images
            local_len.append(lenDataLoad) # gets the number of images per data loader


        server_aggregate(centralizedModel, federatedModels, local_len)

        # Calculate avg training loss over all selected users at each round
        local_train_loss_avg = sum(
            local_train_losses) / len(local_train_losses)
        global_train_losses.append(local_train_loss_avg)

        # Calculate avg training accuracy over all selected users at each round
        local_train_acc_avg = sum(
            local_train_accuracies) / len(local_train_accuracies)
        global_train_accuracies.append(local_train_acc_avg)

        # Calculate avg valid loss over all selected users at each round
        local_valid_loss_avg = sum(
            local_valid_losses) / len(local_valid_losses)
        global_valid_losses.append(local_valid_loss_avg)

        # Calculate avg valid accuracy over all selected users at each round
        local_valid_acc_avg = sum(
            local_valid_accuracies) / len(local_valid_accuracies)
        global_valid_accuracies.append(local_valid_acc_avg)

        print(f'Round: {round}\t'
              f'Train loss: {local_train_loss_avg:.4f}\t'
              f'Valid loss: {local_valid_loss_avg:.4f}\t'
              f'Train accuracy: {100 * local_train_acc_avg:.2f}\t'
              f'Valid accuracy: {100 * local_valid_acc_avg:.2f}')
    return centralizedModel, federatedModels, optimizers, (global_train_losses, global_valid_losses), (global_train_accuracies, global_valid_accuracies)


# def partitionData(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    
#     min_size = 0
#     min_require_size = 10
#     K = 10
#    
#     N = y_train.shape[0]
#     np.random.seed(2020)
#     net_dataidx_map = {}

#     while min_size < min_require_size:
#         idx_batch = [[] for _ in range(n_parties)]
#         for k in range(K):
#             idx_k = np.where(y_train == k)[0]
#             np.random.shuffle(idx_k)
#             proportions = np.random.dirichlet(np.repeat(beta, n_parties))
#             # logger.info("proportions1: ", proportions)
#             # logger.info("sum pro1:", np.sum(proportions))
#             ## Balance
#             proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
#             # logger.info("proportions2: ", proportions)
#             proportions = proportions / proportions.sum()
#             # logger.info("proportions3: ", proportions)
#             proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
#             # logger.info("proportions4: ", proportions)
#             idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
#             min_size = min([len(idx_j) for idx_j in idx_batch])
#             # if K == 2 and n_parties <= 10:
#             #     if np.min(proportions) < 200:
#             #         min_size = 0
#             #         break


#     for j in range(n_parties):
#         np.random.shuffle(idx_batch[j])
#         net_dataidx_map[j] = idx_batch[j]

#     traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
#     return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

# def record_net_data_stats(y_train, net_dataidx_map, logdir):

#     net_cls_counts = {}

#     for net_i, dataidx in net_dataidx_map.items():
#         unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
#         tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
#         net_cls_counts[net_i] = tmp

#     logger.info('Data statistics: %s' % str(net_cls_counts))

#     return net_cls_counts