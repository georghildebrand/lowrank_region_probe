import torch
import pytest
from datasets.image_binary import load_binary_dataset, SPLITS


@pytest.mark.parametrize("split", list(SPLITS.keys()))
def test_shapes_and_types(split):
    Xtr, ytr, Xev, yev = load_binary_dataset(split, n_train=500, n_eval=200, seed=0)
    assert Xtr.shape == (500, 784) and ytr.shape == (500, 1)
    assert Xev.shape == (200, 784) and yev.shape == (200, 1)
    assert Xtr.dtype == torch.float32
    assert set(ytr.unique().tolist()) <= {0.0, 1.0}


def test_both_classes_present():
    _, ytr, _, yev = load_binary_dataset("fashion_hard", n_train=500, n_eval=200, seed=0)
    assert ytr.mean() > 0.1 and ytr.mean() < 0.9
    assert yev.mean() > 0.1 and yev.mean() < 0.9


def test_standardization():
    Xtr, _, _, _ = load_binary_dataset("mnist_even_odd", n_train=2000, n_eval=100, seed=0)
    assert abs(Xtr.mean().item()) < 0.05
    assert abs(Xtr.std().item() - 1.0) < 0.1


def test_seed_determinism():
    a = load_binary_dataset("mnist_even_odd", n_train=300, n_eval=100, seed=1)
    b = load_binary_dataset("mnist_even_odd", n_train=300, n_eval=100, seed=1)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_mnist_lt5_x_alignment_with_even_odd():
    """mnist_lt5 and mnist_even_odd share the same X at equal seed (only labels differ)."""
    Xtr_a, _, Xev_a, _ = load_binary_dataset("mnist_even_odd", n_train=500, n_eval=200, seed=3)
    Xtr_b, _, Xev_b, _ = load_binary_dataset("mnist_lt5",      n_train=500, n_eval=200, seed=3)
    assert torch.equal(Xtr_a, Xtr_b), "Train X differs between mnist_even_odd and mnist_lt5"
    assert torch.equal(Xev_a, Xev_b), "Eval X differs between mnist_even_odd and mnist_lt5"
