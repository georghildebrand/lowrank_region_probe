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


def test_directional_probe_keys_and_shapes():
    from probes.layer_stability import evaluate_layer_stability_directional
    torch.manual_seed(0)
    model = DeepMLP(input_dim=16, hidden_dims=(8, 8, 4))
    X = torch.randn(50, 16)
    W0, _ = model.layer_weight(1)
    D = torch.randn_like(W0)
    out = evaluate_layer_stability_directional(model, X, layer=1,
                                               direction=D, scale=0.1)
    assert set(out) == {"dist", "hamming_dir"}
    assert out["hamming_dir"].shape == (50,)
    assert ((out["hamming_dir"] >= 0) & (out["hamming_dir"] <= 1)).all()


def test_directional_probe_zero_scale_is_stable():
    from probes.layer_stability import evaluate_layer_stability_directional
    torch.manual_seed(0)
    model = DeepMLP(input_dim=16, hidden_dims=(8, 8, 4))
    X = torch.randn(50, 16)
    W0, _ = model.layer_weight(0)
    out = evaluate_layer_stability_directional(model, X, layer=0,
                                               direction=torch.randn_like(W0),
                                               scale=1e-8)
    assert out["hamming_dir"].min() > 0.99
