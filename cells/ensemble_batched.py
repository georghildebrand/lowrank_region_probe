"""
Vectorized perturbation ensembles: one stacked tensor instead of E deepcopied
models. ~10x faster eval at MNIST scale and MPS-friendly.
Family is always centroid_preserving (the invariant used in all experiments):
b_new = b0 + (W0 - W_new) @ centroid.
"""
import torch


def generate_perturbation_batch(W0, b0, ensemble_size, rank, scale, mode, centroid):
    out_dim, in_dim = W0.shape
    if mode == "lowrank":
        A = torch.randn(ensemble_size, rank, in_dim, device=W0.device)
        B = torch.randn(ensemble_size, out_dim, rank, device=W0.device)
        P = torch.bmm(B, A)
    elif mode == "fullrank":
        P = torch.randn(ensemble_size, out_dim, in_dim, device=W0.device)
    else:
        raise ValueError(f"mode must be 'lowrank' or 'fullrank', got {mode}")

    norms = P.flatten(1).norm(dim=1).clamp_min(1e-12)
    P = P / norms[:, None, None] * (scale * W0.norm())

    W = W0[None] + P
    b = b0[None] + torch.einsum("eoi,i->eo", W0[None] - W, centroid)
    return W, b


def batched_gate_patterns(X, W, b, chunk=16):
    outs = []
    for start in range(0, W.shape[0], chunk):
        We, be = W[start:start + chunk], b[start:start + chunk]
        z = torch.einsum("ni,eoi->eno", X, We) + be[:, None, :]
        outs.append(z > 0)
    return torch.cat(outs, dim=0)
