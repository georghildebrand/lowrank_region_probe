import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def gate_pattern(self, x):
        with torch.no_grad():
            pre_activation = self.fc1(x)
            return pre_activation > 0

    def get_W1(self):
        return self.fc1.weight.data
