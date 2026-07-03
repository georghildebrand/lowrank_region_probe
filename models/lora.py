import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_with_delta(model, x, layer, delta):
    """DeepMLP forward with W_layer replaced by W_layer + delta. Grads flow
    through delta only if delta requires grad; model params stay frozen by
    caller convention."""
    h = x
    for i, fc in enumerate(model.fcs):
        W = fc.weight + delta if i == layer else fc.weight
        h = torch.relu(F.linear(h, W, fc.bias))
    return F.linear(h, model.out.weight, model.out.bias)


def lora_finetune(model, layer, X, y, rank=1, steps=300, lr=1e-2, batch=256, seed=0):
    """Rank-r LoRA fine-tune of ONE layer. Returns the learned delta [out,in]
    (detached) and final train accuracy. Standard LoRA init: A ~ N(0, 0.01),
    B = 0, so delta starts at exactly zero."""
    W0, _ = model.layer_weight(layer)
    out_dim, in_dim = W0.shape
    gen = torch.Generator().manual_seed(seed)
    A = nn.Parameter(torch.randn(rank, in_dim, generator=gen) * 0.01)
    B = nn.Parameter(torch.zeros(out_dim, rank))
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([A, B], lr=lr)
    crit = nn.BCEWithLogitsLoss()
    n = X.shape[0]
    step = 0
    while step < steps:
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            logits = forward_with_delta(model, X[idx], layer, B @ A)
            crit(logits, y[idx]).backward()
            opt.step()
            step += 1
            if step >= steps:
                break
    delta = (B @ A).detach()
    with torch.no_grad():
        acc = ((forward_with_delta(model, X, layer, delta) > 0).float() == y).float().mean().item()
    return delta, acc


def dominant_direction(delta):
    """Top singular direction of delta as a unit-Frobenius [out,in] matrix."""
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    return U[:, :1] @ Vh[:1, :]


def forward_with_all_deltas(model, x, deltas):
    """DeepMLP forward with W_layer replaced by W_layer + deltas[layer] for all layers.
    deltas: dict mapping layer index -> [out,in] delta tensor (or missing = no delta)."""
    h = x
    for i, fc in enumerate(model.fcs):
        d = deltas.get(i)
        W = fc.weight + d if d is not None else fc.weight
        h = torch.relu(F.linear(h, W, fc.bias))
    return F.linear(h, model.out.weight, model.out.bias)


def lora_finetune_multilayer(model, layers, X, y, rank=1, steps=1000,
                              lr=1e-2, batch=256, seed=0):
    """Rank-r LoRA fine-tune of multiple layers simultaneously.
    Returns (deltas, ft_acc) where deltas is dict {layer_idx: [out,in] tensor}."""
    gen = torch.Generator().manual_seed(seed)
    params = []
    AB = {}
    for layer in layers:
        W0, _ = model.layer_weight(layer)
        out_dim, in_dim = W0.shape
        A = nn.Parameter(torch.randn(rank, in_dim, generator=gen) * 0.01)
        B = nn.Parameter(torch.zeros(out_dim, rank))
        AB[layer] = (A, B)
        params.extend([A, B])
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(params, lr=lr)
    crit = nn.BCEWithLogitsLoss()
    n = X.shape[0]
    step = 0
    while step < steps:
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            deltas = {l: B @ A for l, (A, B) in AB.items()}
            logits = forward_with_all_deltas(model, X[idx], deltas)
            crit(logits, y[idx]).backward()
            opt.step()
            step += 1
            if step >= steps:
                break
    deltas = {l: (B @ A).detach() for l, (A, B) in AB.items()}
    with torch.no_grad():
        acc = ((forward_with_all_deltas(model, X, deltas) > 0).float() == y).float().mean().item()
    return deltas, acc
