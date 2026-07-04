import numpy as np
import torch
from datasets.synthetic_polytopes import generate_soft_checkerboard


def hard_labels(X):
    cell = torch.zeros(X.shape[0], dtype=torch.long)
    cell[(X[:, 0] >= 0) & (X[:, 1] >= 0)] = 0
    cell[(X[:, 0] < 0) & (X[:, 1] >= 0)] = 1
    cell[(X[:, 0] < 0) & (X[:, 1] < 0)] = 2
    cell[(X[:, 0] >= 0) & (X[:, 1] < 0)] = 3
    return (cell % 2).float().unsqueeze(1)


def test_shapes_and_dtypes():
    X, y, cell_id = generate_soft_checkerboard(n_samples=500, softness=0.1, seed=0)
    assert X.shape == (500, 2) and X.dtype == torch.float32
    assert y.shape == (500, 1) and y.dtype == torch.float32
    assert cell_id.shape == (500,) and cell_id.dtype == torch.long
    assert set(y.unique().tolist()) <= {0.0, 1.0}


def test_softness_zero_is_hard_checkerboard():
    X, y, _ = generate_soft_checkerboard(n_samples=2000, softness=0.0, seed=1)
    assert torch.equal(y, hard_labels(X))


def test_far_points_keep_hard_label():
    X, y, _ = generate_soft_checkerboard(n_samples=5000, softness=0.05, seed=2)
    margin = torch.minimum(X[:, 0].abs(), X[:, 1].abs())
    far = margin > 0.5  # 10 length scales from the boundary
    assert far.sum() > 100
    assert torch.equal(y[far], hard_labels(X)[far])


def test_flip_rate_monotone_in_softness():
    rates = []
    for softness in [0.05, 0.2, 0.5]:
        X, y, _ = generate_soft_checkerboard(n_samples=20000, softness=softness, seed=3)
        rates.append((y != hard_labels(X)).float().mean().item())
    assert rates[0] < rates[1] < rates[2]
    assert rates[0] > 0.0


def test_seed_reproducibility():
    a = generate_soft_checkerboard(n_samples=300, softness=0.1, seed=7)
    b = generate_soft_checkerboard(n_samples=300, softness=0.1, seed=7)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
