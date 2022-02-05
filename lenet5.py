# IMPLEMENTED USING https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320

import torch
import torch.nn as nn

N_CLASSES = 10


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*5*5, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=192),
            nn.ReLU(),
            nn.Linear(in_features=192, out_features=N_CLASSES)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        # remove this bcs CrossEntropy does it
        probs = torch.softmax(logits, dim=1)
        return logits, probs
