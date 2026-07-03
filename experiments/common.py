"""Shared helpers for LoRA-family experiments."""
import torch
import torch.nn as nn

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def train_model(model, X, y, max_epochs=60, target_acc=0.99, lr=1e-3, batch=256):
    model = model.to(DEVICE)
    X, y = X.to(DEVICE), y.to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    acc = 0.0
    for epoch in range(max_epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            crit(model(X[idx]), y[idx]).backward()
            opt.step()
        with torch.no_grad():
            acc = ((model(X) > 0).float() == y).float().mean().item()
        if acc >= target_acc:
            break
    return model.cpu(), acc
