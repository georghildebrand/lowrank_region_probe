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
