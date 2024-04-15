import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import json
import numpy as np

# Assuming X_train.shape[1] is defined elsewhere and is the input size
class DNN(nn.Module):
    def __init__(self, layers, dropout_rate):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(layers[-2], layers[-1])

    def forward(self, x):
        for layer in self.layers:
            x = nn.functional.relu(layer(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

