import torch
import copy
from models.mlp import MLP

def get_random_network_baseline(input_dim, hidden_dim):
    """Returns a fresh model with random weights (untrained)."""
    return MLP(input_dim=input_dim, hidden_dim=hidden_dim)

def get_label_shuffle_data(X, y):
    """Returns labels shuffled randomly."""
    y_shuffled = y[torch.randperm(y.shape[0])]
    return X, y_shuffled
