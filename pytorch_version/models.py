from typing import Type
import numpy as np
import torch.nn as nn


class Network(nn.Module):
    """
Class representing the Artificial Neural Network
    """
    def __init__(self, n_inputs: int, n_layers: int, n_neurons: int, n_outputs: int):
        super(Network, self).__init__()
        self.network = nn.ModuleList()
        
        self.network.append(nn.Linear(n_inputs, n_neurons))
        self.network.append(nn.ReLU())
        
        for _ in range(n_layers):
            self.network.append(nn.Linear(n_neurons, n_neurons))
            self.network.append(nn.ReLU())
         
        self.network.append(nn.Linear(n_neurons, n_outputs))
        self.network.append(nn.Softmax(dim=1))
        
        self.params = ""
        self.train_losses = []
        self.train_accu = []
        self.val_losses = []
        self.val_accu = []
    
    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x
