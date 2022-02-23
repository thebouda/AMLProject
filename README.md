# AMLProject - Federated Learning: where machine learning and data privacy can coexist

Introduced by Google in 2016, Federated Learning enables models to be iteratively trained in a decentralized way, on local devices. At each round, a subset of sources (also called clients) are selected to perform local training on their own data. Then, the centralized model aggregates the ensemble of updates to make the new global model and send back it to another subset of clients.

As a part of the Advanced Machine Learning course at Politecnico di Torino, this paper will cover the work on the replication of the experiment proposed on the CIFAR10 dataset which consists of analyzing the effects that client's data distributions have on federated models. Especially distributions such as Non-Identical Class Distribution (each client has a different class distribution) and Imbalanced Client Sizes (each client has a different amount of data). 

# centralisedModel.ipynb
Contains the code implementation of the centralised model

# federated.ipynb
Code implementation of the federated model

# federatedServerMomentum.ipynb
Code implementation of the federated model with server momentum (FedAvgM)

# AMLReport.pdf
Final pdf report

# lenet5
CNN network class

# cifar folder
Folder for the dirichlet distribution

# main.tex 
latex version of the final report

# reference.bib
References used for the report

