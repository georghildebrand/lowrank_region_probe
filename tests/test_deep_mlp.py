import torch
from models.deep_mlp import DeepMLP


def test_forward_shape():
    m = DeepMLP(input_dim=10, hidden_dims=(16, 8))
    out = m(torch.randn(5, 10))
    assert out.shape == (5, 1)


def test_layer_states_shapes_and_relu_consistency():
    m = DeepMLP(input_dim=10, hidden_dims=(16, 8, 4))
    x = torch.randn(7, 10)
    states = m.layer_states(x)
    assert len(states) == 3
    h0, z0 = states[0]
    assert h0.shape == (7, 10) and z0.shape == (7, 16)
    h1, z1 = states[1]
    # input to layer 1 is relu of layer 0 preactivation
    assert torch.allclose(h1, torch.relu(z0))
    assert z1.shape == (7, 8)


def test_gate_pattern_matches_preactivation_sign():
    m = DeepMLP(input_dim=6, hidden_dims=(12, 5))
    x = torch.randn(9, 6)
    for layer in range(2):
        gates = m.gate_pattern(x, layer=layer)
        _, z = m.layer_states(x)[layer]
        assert torch.equal(gates, z > 0)


def test_layer_weight_access():
    m = DeepMLP(input_dim=6, hidden_dims=(12, 5))
    W, b = m.layer_weight(1)
    assert W.shape == (5, 12) and b.shape == (5,)
