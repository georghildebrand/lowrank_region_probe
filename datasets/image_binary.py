"""
Binary MNIST / Fashion-MNIST splits for the real-data stability experiments.

Splits:
- mnist_even_odd: digit parity
- fashion_easy:   footwear {sandal, sneaker, ankle boot} vs everything else
- fashion_hard:   {t-shirt, dress} vs {pullover, coat, shirt}, other classes
                  dropped — maximally overlapping upper-body garments
"""
import os
import numpy as np
import torch
from torchvision import datasets

DATA_ROOT = os.path.expanduser("~/.torch-datasets")

SPLITS = {
    "mnist_even_odd": ("mnist", None),
    "fashion_easy": ("fashion", None),
    "fashion_hard": ("fashion", None),
}


def _labels(split_name, raw_targets):
    t = np.asarray(raw_targets)
    if split_name == "mnist_even_odd":
        return (t % 2 == 0).astype(np.float32), np.ones(len(t), dtype=bool)
    if split_name == "fashion_easy":
        return np.isin(t, [5, 7, 9]).astype(np.float32), np.ones(len(t), dtype=bool)
    if split_name == "fashion_hard":
        keep = np.isin(t, [0, 3, 2, 4, 6])
        return np.isin(t, [0, 3]).astype(np.float32), keep
    raise ValueError(f"unknown split: {split_name}")


def load_binary_dataset(split_name, n_train=20000, n_eval=10000, seed=0):
    source, _ = SPLITS[split_name]
    cls = datasets.MNIST if source == "mnist" else datasets.FashionMNIST
    train_ds = cls(DATA_ROOT, train=True, download=True)
    test_ds = cls(DATA_ROOT, train=False, download=True)

    def prep(ds, n, rng):
        X = ds.data.numpy().reshape(len(ds), -1).astype(np.float32) / 255.0
        y, keep = _labels(split_name, ds.targets)
        X, y = X[keep], y[keep]
        idx = rng.permutation(len(X))[:n]
        return X[idx], y[idx]

    rng = np.random.default_rng(seed)
    X_train, y_train = prep(train_ds, n_train, rng)
    X_eval, y_eval = prep(test_ds, n_eval, rng)

    mean, std = X_train.mean(), X_train.std() + 1e-8
    X_train = (X_train - mean) / std
    X_eval = (X_eval - mean) / std

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    return (to_t(X_train), to_t(y_train).unsqueeze(1),
            to_t(X_eval), to_t(y_eval).unsqueeze(1))
