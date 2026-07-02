"""
Real-data replication: geometry bonus on MNIST / Fashion-MNIST binary tasks
with a 3-hidden-layer MLP, probed layer-wise.

Per (split, seed): train real-label + shuffled-label models. For each layer
and rank, compute hamming + region-identity stability under lowrank vs
fullrank single-layer perturbation, then partial rho conditioned on distance
in that layer's input space. Bonus = trained rho - shuffled rho.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from models.deep_mlp import DeepMLP
from datasets.image_binary import load_binary_dataset, SPLITS
from probes.layer_stability import evaluate_layer_stability
from analysis.conditioning import partial_spearman_mode_given_distance

SPLIT_NAMES = ["mnist_even_odd", "fashion_easy", "fashion_hard"]
SEEDS = [0, 1, 2]
RANKS = [1, 8]
SCALE = 0.1
HIDDEN_DIMS = (256, 128, 64)
LAYERS = [0, 1, 2]
N_TRAIN = 20000
N_EVAL = 10000
ENSEMBLE_SIZE = 64
MIN_MASS = 10

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def train_model(model, X, y, max_epochs, target_acc, lr=1e-3, batch=256):
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


def eval_condition(model, X_eval):
    """All layer x rank stability results for one model."""
    res = {}
    for layer in LAYERS:
        for rank in RANKS:
            r = evaluate_layer_stability(model, X_eval, layer=layer, rank=rank,
                                         scale=SCALE, ensemble_size=ENSEMBLE_SIZE,
                                         min_mass=MIN_MASS)
            res[(layer, rank)] = {
                "rho_hamming": partial_spearman_mode_given_distance(
                    r["hamming_low"], r["hamming_full"], r["dist"]),
                "rho_ri": partial_spearman_mode_given_distance(
                    r["ri_low"], r["ri_full"], r["dist"]),
                "mean_hamming_low": float(np.mean(r["hamming_low"])),
                "mean_ri_low": float(np.nanmean(r["ri_low"])),
                "n_ri_valid": int(np.sum(~np.isnan(r["ri_low"]))),
            }
    return res


def main():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    all_rows = []

    for split in SPLIT_NAMES:
        for seed in SEEDS:
            X_tr, y_tr, X_ev, _ = load_binary_dataset(split, n_train=N_TRAIN,
                                                      n_eval=N_EVAL, seed=seed)
            torch.manual_seed(seed)
            m_t = DeepMLP(784, HIDDEN_DIMS)
            m_t, acc_t = train_model(m_t, X_tr, y_tr, max_epochs=60, target_acc=0.99)

            torch.manual_seed(seed)
            m_s = DeepMLP(784, HIDDEN_DIMS)
            y_sh = y_tr[torch.randperm(y_tr.shape[0])]
            m_s, acc_s = train_model(m_s, X_tr, y_sh, max_epochs=200, target_acc=0.90)

            print(f"[{split} seed={seed}] trained acc={acc_t:.3f} shuffled acc={acc_s:.3f}")

            res_t = eval_condition(m_t, X_ev)
            res_s = eval_condition(m_s, X_ev)

            for (layer, rank), rt in res_t.items():
                rs = res_s[(layer, rank)]
                row = {
                    "split": split, "seed": seed, "layer": layer, "rank": rank,
                    "acc_trained": acc_t, "acc_shuffled": acc_s,
                    "trained": rt, "shuffled": rs,
                    "bonus_hamming": rt["rho_hamming"] - rs["rho_hamming"],
                    "bonus_ri": rt["rho_ri"] - rs["rho_ri"],
                }
                all_rows.append(row)
                print(f"    layer={layer} rank={rank} "
                      f"bonus_hamming={row['bonus_hamming']:+.3f} "
                      f"bonus_ri={row['bonus_ri']:+.3f}")

    with open("results/logs/real_data_results.json", "w") as f:
        json.dump({"splits": SPLIT_NAMES, "seeds": SEEDS, "ranks": RANKS,
                   "scale": SCALE, "hidden_dims": HIDDEN_DIMS,
                   "results": all_rows}, f, indent=2)

    # Layer profile figure: bonus vs layer, per split (rank=1, both metrics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, metric in zip(axes, ["bonus_hamming", "bonus_ri"]):
        for split in SPLIT_NAMES:
            means, stds = [], []
            for layer in LAYERS:
                vals = [r[metric] for r in all_rows
                        if r["split"] == split and r["layer"] == layer and r["rank"] == 1]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax.errorbar(LAYERS, means, yerr=stds, marker="o", capsize=3, label=split)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("layer")
        ax.set_xticks(LAYERS)
        ax.set_title(metric)
    axes[0].set_ylabel("geometry bonus (trained − shuffled partial ρ)")
    axes[0].legend()
    fig.suptitle("Layer-wise geometry bonus, rank=1, scale=0.1")
    fig.tight_layout()
    fig.savefig("results/figures/real_data_layer_profile.png", dpi=150)
    plt.close(fig)

    print("\n==== REAL DATA SUMMARY (rank=1, mean over seeds) ====")
    print(f"{'split':>16} {'layer':>6} {'bonus_ham':>10} {'bonus_ri':>9}")
    for split in SPLIT_NAMES:
        for layer in LAYERS:
            rows = [r for r in all_rows
                    if r["split"] == split and r["layer"] == layer and r["rank"] == 1]
            bh = np.mean([r["bonus_hamming"] for r in rows])
            br = np.mean([r["bonus_ri"] for r in rows])
            print(f"{split:>16} {layer:>6} {bh:>+10.3f} {br:>+9.3f}")
    print("\nRank-specificity check (rank=8 should be ~0):")
    for split in SPLIT_NAMES:
        rows = [r for r in all_rows if r["split"] == split and r["rank"] == 8]
        print(f"  {split}: mean bonus_hamming={np.mean([r['bonus_hamming'] for r in rows]):+.3f}")
    print("\nSaved results/logs/real_data_results.json + layer profile figure.")


if __name__ == "__main__":
    main()
