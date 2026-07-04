import torch
from cells.ensemble_batched import generate_perturbation_batch, batched_gate_patterns


def _setup():
    torch.manual_seed(0)
    W0 = torch.randn(16, 8)
    b0 = torch.randn(16)
    centroid = torch.randn(8)
    return W0, b0, centroid


def test_norm_matching():
    W0, b0, c = _setup()
    W, b = generate_perturbation_batch(W0, b0, ensemble_size=32, rank=1,
                                       scale=0.1, mode="lowrank", centroid=c)
    pert_norms = (W - W0[None]).flatten(1).norm(dim=1)
    expected = 0.1 * W0.norm()
    assert torch.allclose(pert_norms, expected.expand(32), rtol=1e-5)


def test_centroid_preservation():
    W0, b0, c = _setup()
    W, b = generate_perturbation_batch(W0, b0, ensemble_size=8, rank=2,
                                       scale=0.1, mode="lowrank", centroid=c)
    base_offset = W0 @ c + b0
    for e in range(8):
        offset = W[e] @ c + b[e]
        assert torch.allclose(offset, base_offset, atol=1e-5)


def test_lowrank_rank():
    W0, b0, c = _setup()
    W, _ = generate_perturbation_batch(W0, b0, ensemble_size=4, rank=1,
                                       scale=0.1, mode="lowrank", centroid=c)
    for e in range(4):
        assert torch.linalg.matrix_rank(W[e] - W0) == 1


def test_batched_gates_match_loop():
    W0, b0, c = _setup()
    W, b = generate_perturbation_batch(W0, b0, ensemble_size=8, rank=1,
                                       scale=0.1, mode="fullrank", centroid=c)
    X = torch.randn(100, 8)
    batched = batched_gate_patterns(X, W, b, chunk=3)  # chunk not dividing E
    for e in range(8):
        loop = (X @ W[e].T + b[e]) > 0
        assert torch.equal(batched[e], loop)


def test_directional_norm_matching():
    W0, b0, c = _setup()
    D = torch.randn(16, 8)
    W, b = generate_perturbation_batch(W0, b0, ensemble_size=4, rank=1,
                                       scale=0.1, mode="directional",
                                       centroid=c, direction=D)
    pert_norms = (W - W0[None]).flatten(1).norm(dim=1)
    assert torch.allclose(pert_norms, (0.1 * W0.norm()).expand(4), rtol=1e-5)


def test_directional_signs_alternate():
    W0, b0, c = _setup()
    D = torch.randn(16, 8)
    W, _ = generate_perturbation_batch(W0, b0, ensemble_size=2, rank=1,
                                       scale=0.1, mode="directional",
                                       centroid=c, direction=D)
    P0, P1 = W[0] - W0, W[1] - W0
    assert torch.allclose(P0, -P1, atol=1e-6)


def test_directional_alignment():
    W0, b0, c = _setup()
    D = torch.randn(16, 8)
    W, _ = generate_perturbation_batch(W0, b0, ensemble_size=2, rank=1,
                                       scale=0.1, mode="directional",
                                       centroid=c, direction=D)
    P0 = (W[0] - W0).flatten()
    cos = torch.dot(P0, D.flatten()) / (P0.norm() * D.flatten().norm())
    assert abs(abs(cos.item()) - 1.0) < 1e-5


def test_directional_centroid_preservation():
    W0, b0, c = _setup()
    D = torch.randn(16, 8)
    W, b = generate_perturbation_batch(W0, b0, ensemble_size=2, rank=1,
                                       scale=0.1, mode="directional",
                                       centroid=c, direction=D)
    base_offset = W0 @ c + b0
    for e in range(2):
        assert torch.allclose(W[e] @ c + b[e], base_offset, atol=1e-5)


def test_directional_requires_direction():
    W0, b0, c = _setup()
    import pytest
    with pytest.raises(ValueError):
        generate_perturbation_batch(W0, b0, 2, 1, 0.1, "directional", c)


def test_batched_logits_matches_loop():
    import copy
    from models.mlp import MLP
    from cells.ensemble_batched import generate_perturbation_batch, batched_logits

    torch.manual_seed(0)
    model = MLP(input_dim=4, hidden_dim=8)
    X = torch.randn(30, 4)
    W0, b0 = model.fc1.weight.data, model.fc1.bias.data
    centroid = X.mean(dim=0)

    W, b = generate_perturbation_batch(W0, b0, ensemble_size=5, rank=1,
                                       scale=0.1, mode="lowrank", centroid=centroid)
    logits = batched_logits(X, W, b, model.fc2.weight.data, model.fc2.bias.data,
                            chunk=2)
    assert logits.shape == (5, 30)

    for e in range(5):
        m = copy.deepcopy(model)
        with torch.no_grad():
            m.fc1.weight.copy_(W[e])
            m.fc1.bias.copy_(b[e])
            expected = m(X).squeeze(1)
        assert torch.allclose(logits[e], expected, atol=1e-5)
