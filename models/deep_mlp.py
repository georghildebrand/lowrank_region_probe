import torch
import torch.nn as nn


class DeepMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(256, 128, 64)):
        super().__init__()
        self.hidden_dims = tuple(hidden_dims)
        self.fcs = nn.ModuleList()
        d = input_dim
        for h in self.hidden_dims:
            self.fcs.append(nn.Linear(d, h))
            d = h
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        for fc in self.fcs:
            x = torch.relu(fc(x))
        return self.out(x)

    def layer_states(self, x):
        """Per hidden layer: (input activations, preactivations). No grad."""
        with torch.no_grad():
            states = []
            h = x
            for fc in self.fcs:
                z = fc(h)
                states.append((h, z))
                h = torch.relu(z)
            return states

    def gate_pattern(self, x, layer=0):
        return self.layer_states(x)[layer][1] > 0

    def layer_weight(self, layer):
        fc = self.fcs[layer]
        return fc.weight.data, fc.bias.data
