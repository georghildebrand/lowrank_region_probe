import numpy as np
import torch
from models.deep_mlp import DeepMLP
from probes.layer_stability import evaluate_layer_stability


def test_output_shapes_and_ranges():
    torch.manual_seed(0)
    m = DeepMLP(input_dim=6, hidden_dims=(12, 8))
    X = torch.randn(300, 6)
    res = evaluate_layer_stability(m, X, layer=1, rank=1, scale=0.1,
                                   ensemble_size=8, min_mass=5)
    for key in ["hamming_low", "hamming_full", "ri_low", "ri_full", "dist"]:
        assert res[key].shape == (300,)
    assert np.nanmin(res["hamming_low"]) >= 0 and np.nanmax(res["hamming_low"]) <= 1
    assert np.nanmin(res["ri_low"]) >= 0 and np.nanmax(res["ri_low"]) <= 1
    assert (res["dist"] >= 0).all()


def test_zero_scale_is_perfectly_stable():
    torch.manual_seed(0)
    m = DeepMLP(input_dim=6, hidden_dims=(12, 8))
    X = torch.randn(200, 6)
    res = evaluate_layer_stability(m, X, layer=0, rank=1, scale=0.0,
                                   ensemble_size=4, min_mass=5)
    assert np.allclose(res["hamming_low"], 1.0)
    valid = ~np.isnan(res["ri_low"])
    assert np.allclose(res["ri_low"][valid], 1.0)


def test_layers_are_independent():
    # perturbing layer 1 must not depend on layer-0 weights being perturbed:
    # distances are computed in layer-1's input space
    torch.manual_seed(0)
    m = DeepMLP(input_dim=6, hidden_dims=(12, 8))
    X = torch.randn(100, 6)
    res = evaluate_layer_stability(m, X, layer=1, rank=1, scale=0.05,
                                   ensemble_size=4, min_mass=5)
    h, z = m.layer_states(X)[1]
    W, b = m.layer_weight(1)
    expected_dist = (torch.abs(z) / (torch.norm(W, dim=1) + 1e-12)).min(dim=1)[0]
    assert np.allclose(res["dist"], expected_dist.numpy(), atol=1e-5)
