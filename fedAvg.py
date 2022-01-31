# IMPLEMENTED USING https://towardsdatascience.com/federated-learning-a-simple-implementation-of-fedavg-federated-averaging-with-pytorch-90187c9c9577

import torch
import torch.nn as nn
from lenet5_ import LeNet

def create_clients_model(n_clients, learning_rate):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(n_clients):
        model_name = "" + str(i)
        model = LeNet()
        model_dict.update({model_name : model})

        optimizer_name = "" + str(i)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer_dict.update({optimizer_name : optimizer})

        criterion_name = "" + str(i)
        criterion = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name : criterion})

    return model_dict, optimizer_dict, criterion_dict

def get_averaged_weights(model_dict, n_clients):
    mean_weights = []
    mean_bias = []

    layer0_mean_weight = torch.zeros(model_dict["0"].feature_extractor[0].weight.shape)
    mean_weights.append(layer0_mean_weight)
    layer0_mean_bias = torch.zeros(model_dict["0"].feature_extractor[0].bias.shape)
    mean_bias.append(layer0_mean_bias)

    layer3_mean_weight = torch.zeros(model_dict["0"].feature_extractor[3].weight.shape)
    mean_weights.append(layer3_mean_weight)
    layer3_mean_bias = torch.zeros(model_dict["0"].feature_extractor[3].bias.shape)
    mean_bias.append(layer3_mean_bias)

    layer6_mean_weight = torch.zeros(model_dict["0"].feature_extractor[6].weight.shape)
    mean_weights.append(layer6_mean_weight)
    layer6_mean_bias = torch.zeros(model_dict["0"].feature_extractor[6].bias.shape)
    mean_bias.append(layer6_mean_bias)

    layer8_mean_weight = torch.zeros(model_dict["0"].classifier[0].weight.shape)
    mean_weights.append(layer8_mean_weight)
    layer8_mean_bias = torch.zeros(model_dict["0"].classifier[0].bias.shape)
    mean_bias.append(layer8_mean_bias)
    
    layer10_mean_weight = torch.zeros(model_dict["0"].classifier[2].weight.shape)
    mean_weights.append(layer10_mean_weight)
    layer10_mean_bias = torch.zeros(model_dict["0"].classifier[2].bias.shape)
    mean_bias.append(layer10_mean_bias)

    with torch.no_grad():
        for model_name in model_dict:
            layer0_mean_weight += model_dict[model_name].feature_extractor[0].weight.data.clone()
            layer0_mean_bias += model_dict[model_name].feature_extractor[0].bias.data.clone()
            layer3_mean_weight += model_dict[model_name].feature_extractor[3].weight.data.clone()
            layer3_mean_bias += model_dict[model_name].feature_extractor[3].bias.data.clone()
            layer6_mean_weight += model_dict[model_name].feature_extractor[6].weight.data.clone()
            layer6_mean_bias += model_dict[model_name].feature_extractor[6].bias.data.clone()

            layer8_mean_weight += model_dict[model_name].classifier[0].weight.data.clone()
            layer8_mean_bias += model_dict[model_name].classifier[0].bias.data.clone()
            layer10_mean_weight += model_dict[model_name].classifier[2].weight.data.clone()
            layer10_mean_bias += model_dict[model_name].classifier[2].bias.data.clone()

        layer0_mean_weight = layer0_mean_weight / n_clients
        layer0_mean_bias = layer0_mean_bias / n_clients
        layer3_mean_weight = layer3_mean_weight / n_clients
        layer3_mean_bias = layer3_mean_bias / n_clients
        layer6_mean_weight = layer6_mean_weight / n_clients
        layer6_mean_bias = layer6_mean_bias / n_clients

        layer8_mean_weight = layer8_mean_weight / n_clients
        layer8_mean_bias = layer8_mean_bias / n_clients
        layer10_mean_weight = layer10_mean_weight / n_clients
        layer10_mean_bias = layer10_mean_bias / n_clients
    
    return mean_weights, mean_bias

def update_main_model(main_model, model_dict, n_clients):
    mean_weigts, mean_bias = get_averaged_weights(model_dict, n_clients)

    with torch.no_grad():
        main_model.feature_extractor[0].weight.data = mean_weigts[0]
        main_model.feature_extractor[0].bias.data = mean_bias[0]
        main_model.feature_extractor[3].weight.data = mean_weigts[1]
        main_model.feature_extractor[3].bias.data = mean_bias[1]
        main_model.feature_extractor[6].weight.data = mean_weigts[2]
        main_model.feature_extractor[6].bias.data = mean_bias[2]

        main_model.classifier[0].weight.data = mean_weigts[3]
        main_model.classifier[0].bias.data = mean_bias[3]
        main_model.classifier[2].weight.data = mean_weigts[4]
        main_model.classifier[2].bias.data = mean_bias[4]

    return main_model