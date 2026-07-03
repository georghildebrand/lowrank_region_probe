"""
LoRA-prediction experiment: does per-point rank-1 probe fragility predict
where a LoRA fine-tune changes model behavior?

Base task: mnist_even_odd; fine-tune task: mnist_lt5 (same X, new labels).
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from models.deep_mlp import DeepMLP
from models.lora import forward_with_delta, lora_finetune
from datasets.image_binary import load_binary_dataset
from probes.layer_stability import evaluate_layer_stability
from analysis.conditioning import partial_spearman

SEEDS = [0, 1, 2]
LAYERS = [0, 1, 2]
PROBE_RANK = 1
PROBE_SCALE = 0.1
LORA_RANK = 1
LORA_STEPS = 300
LORA_LR = 1e-2
HIDDEN_DIMS = (256, 128, 64)
N_TRAIN = 20000
N_EVAL = 10000
ENSEMBLE = 64

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


def main():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    all_rows = []

    for seed in SEEDS:
        # Load both splits with same seed — X must align
        X_tr_a, y_tr_a, X_ev_a, y_ev_a = load_binary_dataset(
            "mnist_even_odd", n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)
        X_tr_b, y_tr_b, X_ev_b, y_ev_b = load_binary_dataset(
            "mnist_lt5", n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)

        assert torch.equal(X_tr_a, X_tr_b), "Train X alignment violated"
        assert torch.equal(X_ev_a, X_ev_b), "Eval X alignment violated"

        # Train base model on even/odd (once per seed)
        torch.manual_seed(seed)
        base = DeepMLP(784, HIDDEN_DIMS)
        base, base_acc = train_model(base, X_tr_a, y_tr_a, max_epochs=60, target_acc=0.99)
        print(f"[seed={seed}] base model acc={base_acc:.3f}")

        # Shared eval inputs (CPU)
        X_eval = X_ev_a
        X_tr = X_tr_a
        y_lt5_tr = y_tr_b  # same X rows, lt5 labels

        for layer in LAYERS:
            # Probe: evaluate layer stability under rank-1 perturbation
            probe = evaluate_layer_stability(base, X_eval, layer=layer,
                                             rank=PROBE_RANK, scale=PROBE_SCALE,
                                             ensemble_size=ENSEMBLE)
            frag_low = 1.0 - probe["hamming_low"]   # fragility = 1 - stability
            frag_full = 1.0 - probe["hamming_full"]
            dist = probe["dist"]

            # Logits before fine-tune
            with torch.no_grad():
                logit_before = base(X_eval).squeeze(1).numpy()

            # LoRA fine-tune on lt5 labels
            delta, ft_acc = lora_finetune(base, layer, X_tr, y_lt5_tr,
                                          rank=LORA_RANK, steps=LORA_STEPS,
                                          lr=LORA_LR, seed=seed)

            # Logits after fine-tune
            with torch.no_grad():
                logit_after = forward_with_delta(base, X_eval, layer, delta).squeeze(1).numpy()

            change = np.abs(logit_after - logit_before)

            # Correlations
            rho_low = spearmanr(frag_low, change)[0]
            rho_full = spearmanr(frag_full, change)[0]
            rho_dist = spearmanr(-dist, change)[0]
            rho_low_given_dist = partial_spearman(frag_low, change, dist)

            row = {
                "seed": seed, "layer": layer,
                "rho_low": float(rho_low),
                "rho_full": float(rho_full),
                "rho_dist": float(rho_dist),
                "rho_low_given_dist": float(rho_low_given_dist),
                "ft_acc": float(ft_acc),
                "base_acc": float(base_acc),
            }
            all_rows.append(row)
            print(f"  layer={layer}  rho_low={rho_low:+.3f}  rho_full={rho_full:+.3f}"
                  f"  rho_dist={rho_dist:+.3f}  partial={rho_low_given_dist:+.3f}"
                  f"  ft_acc={ft_acc:.3f}")

    # Save results
    config = {
        "seeds": SEEDS, "layers": LAYERS, "probe_rank": PROBE_RANK,
        "probe_scale": PROBE_SCALE, "lora_rank": LORA_RANK, "lora_steps": LORA_STEPS,
        "lora_lr": LORA_LR, "hidden_dims": list(HIDDEN_DIMS),
        "n_train": N_TRAIN, "n_eval": N_EVAL, "ensemble": ENSEMBLE,
    }
    with open("results/logs/lora_prediction_results.json", "w") as f:
        json.dump({"config": config, "results": all_rows}, f, indent=2)

    # Figure: grouped bar per layer, 4 correlations
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["rho_low", "rho_full", "rho_dist", "rho_low_given_dist"]
    labels = ["rho(frag_r1)", "rho(frag_full)", "rho(-dist)", "partial rho(frag_r1|dist)"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    x = np.arange(len(LAYERS))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, 4)
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        means, stds = [], []
        for layer in LAYERS:
            vals = [r[metric] for r in all_rows if r["layer"] == layer]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.bar(x + offsets[i], means, width, yerr=stds, label=label,
               color=color, capsize=4, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"layer {l}" for l in LAYERS])
    ax.set_ylabel("Spearman rho")
    ax.set_title("LoRA prediction: fragility vs behavior change (mean ± std over seeds)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("results/figures/lora_prediction.png", dpi=150)
    plt.close(fig)

    # Summary print
    print("\n==== LORA PREDICTION SUMMARY (mean over seeds) ====")
    print(f"{'layer':>5}  {'rho(frag_r1)':>13}  {'rho(frag_full)':>14}  "
          f"{'rho(-dist)':>10}  {'partial rho(frag_r1|dist)':>25}  {'ft_acc':>6}")
    for layer in LAYERS:
        rows = [r for r in all_rows if r["layer"] == layer]
        print(f"{layer:>5}  "
              f"{np.mean([r['rho_low'] for r in rows]):>+13.3f}  "
              f"{np.mean([r['rho_full'] for r in rows]):>+14.3f}  "
              f"{np.mean([r['rho_dist'] for r in rows]):>+10.3f}  "
              f"{np.mean([r['rho_low_given_dist'] for r in rows]):>+25.3f}  "
              f"{np.mean([r['ft_acc'] for r in rows]):>6.3f}")
    print("\nSaved results/logs/lora_prediction_results.json + figure.")


if __name__ == "__main__":
    main()
