import numpy as np
import torch
from analysis.conditioning import partial_spearman_mode_given_distance, min_hyperplane_distance


def test_partial_rho_zero_when_modes_identical():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=200)
    d = rng.uniform(size=200)
    rho = partial_spearman_mode_given_distance(s, s.copy(), d)
    assert abs(rho) < 0.05


def test_partial_rho_positive_when_mode_a_higher():
    rng = np.random.default_rng(0)
    d = rng.uniform(size=500)
    s_b = d + rng.normal(0, 0.05, 500)      # stability tracks distance
    s_a = s_b + 0.3                          # mode a uniformly higher
    rho = partial_spearman_mode_given_distance(s_a, s_b, d)
    assert rho > 0.3


def test_partial_rho_drops_nans():
    rng = np.random.default_rng(1)
    s_a = rng.uniform(size=100)
    s_b = rng.uniform(size=100)
    d = rng.uniform(size=100)
    s_a[:10] = np.nan
    rho = partial_spearman_mode_given_distance(s_a, s_b, d)
    assert np.isfinite(rho)


def test_min_hyperplane_distance_known_case():
    # single unit: w=(1,0), b=0 -> distance = |x_0|
    W = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([0.0])
    X = torch.tensor([[0.5, 3.0], [-2.0, 1.0]])
    d = min_hyperplane_distance(W, b, X)
    assert torch.allclose(d, torch.tensor([0.5, 2.0]))
