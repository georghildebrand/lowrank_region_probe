"""
Capacity-ratio sweep: vary hidden_dim at fixed input_dim=5.

Prediction from prior results: geometry bonus (trained - shuffled partial ρ)
grows as hidden_dim approaches input_dim. In the overcomplete regime
(hidden_dim >> input_dim), random training already creates dense polytopes,
saturating the low-rank probe. In the capacity-constrained regime, only
real structure earns robust regions.

Sweeps hidden_dim in [8, 16, 32, 64, 128, 256] at input_dim=5, rank=1, scale=0.1.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

from models.mlp import MLP
from datasets.gaussian_mixture import generate_gaussian_mixture
from cells.ensemble import generate_perturbation_ensemble
from probes.stability_metrics import compute_point_exact_stability
from controls.baselines import get_label_shuffle_data

SEEDS = [0, 1, 2]
HIDDEN_DIMS = [8, 16, 32, 64, 128, 256]
SCALE = 0.1
RANK = 1
ENSEMBLE_SIZE = 64
INPUT_DIM = 5
N_SAMPLES = 10000
TRAIN_STEPS = 5000
N_CLUSTERS = 4


def train_model(model, X, y, steps, lr=0.005):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model


def model_accuracy(model, X, y):
    with torch.no_grad():
        preds = (model(X) > 0).float()
        return (preds == y).float().mean().item()


def min_hyperplane_distance(model, X):
    with torch.no_grad():
        W = model.fc1.weight
        b = model.fc1.bias
        z = X @ W.T + b
        norms = torch.norm(W, dim=1)
        return torch.min(torch.abs(z) / (norms + 1e-12), dim=1)[0]


def partial_spearman_mode_given_distance(s_low, s_full, dist):
    y = np.concatenate([s_low, s_full])
    m = np.concatenate([np.ones_like(s_low), np.zeros_like(s_full)])
    d = np.concatenate([dist, dist])
    rho_ym = spearmanr(y, m)[0]
    rho_yd = spearmanr(y, d)[0]
    denom = np.sqrt(max(1.0 - rho_yd ** 2, 1e-9))
    return rho_ym / denom


def eval_model(model, X):
    dist = min_hyperplane_distance(model, X).cpu().numpy()
    data_centroid = X.mean(dim=0)
    stab = {}
    for mode in ["lowrank", "fullrank"]:
        ens_cfg = {
            "mode": mode, "rank": RANK, "scale": SCALE,
            "family": "centroid_preserving", "data_centroid": data_centroid,
        }
        ens = generate_perturbation_ensemble(model, ensemble_size=ENSEMBLE_SIZE, perturbation_config=ens_cfg)
        stab[mode] = compute_point_exact_stability(model, ens, X).cpu().numpy()
    return {
        "partial_rho": partial_spearman_mode_given_distance(stab["lowrank"], stab["fullrank"], dist),
        "mean_stab_low": float(stab["lowrank"].mean()),
        "mean_stab_full": float(stab["fullrank"].mean()),
    }


def main():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    all_results = []
    n_total = len(SEEDS) * len(HIDDEN_DIMS)
    i = 0

    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        X, y, _ = generate_gaussian_mixture(
            n_samples=N_SAMPLES, dim=INPUT_DIM, n_clusters=N_CLUSTERS,
            separation=3.0, seed=seed,
        )

        for hidden_dim in HIDDEN_DIMS:
            i += 1
            ratio = hidden_dim / INPUT_DIM

            # Trained
            m_trained = MLP(input_dim=INPUT_DIM, hidden_dim=hidden_dim)
            torch.manual_seed(seed * 1000 + hidden_dim)
            train_model(m_trained, X, y, TRAIN_STEPS)
            acc_t = model_accuracy(m_trained, X, y)

            # Shuffled (same init)
            m_shuffled = MLP(input_dim=INPUT_DIM, hidden_dim=hidden_dim)
            torch.manual_seed(seed * 1000 + hidden_dim)
            _, y_shuffled = get_label_shuffle_data(X, y)
            train_model(m_shuffled, X, y_shuffled, TRAIN_STEPS)

            r_t = eval_model(m_trained, X)
            r_s = eval_model(m_shuffled, X)
            bonus = r_t["partial_rho"] - r_s["partial_rho"]

            print(f"[{i}/{n_total}] seed={seed} hidden={hidden_dim:3d} (ratio={ratio:.1f}) "
                  f"acc={acc_t:.3f}  trained ρ={r_t['partial_rho']:+.3f}  "
                  f"shuffled ρ={r_s['partial_rho']:+.3f}  bonus={bonus:+.3f}")

            all_results.append({
                "seed": seed, "hidden_dim": hidden_dim, "ratio": ratio,
                "acc_trained": acc_t,
                "trained": r_t, "shuffled": r_s, "geometry_bonus": bonus,
            })

    with open("results/logs/capacity_ratio_results.json", "w") as f:
        json.dump({
            "seeds": SEEDS, "hidden_dims": HIDDEN_DIMS,
            "scale": SCALE, "rank": RANK, "input_dim": INPUT_DIM,
            "results": all_results,
        }, f, indent=2)

    # Plot: geometry bonus vs hidden_dim/input_dim ratio
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ratios = [h / INPUT_DIM for h in HIDDEN_DIMS]
    mean_bonus = []
    std_bonus = []
    mean_t_rho = []
    mean_s_rho = []

    for h in HIDDEN_DIMS:
        rows = [r for r in all_results if r["hidden_dim"] == h]
        b = np.array([r["geometry_bonus"] for r in rows])
        mean_bonus.append(b.mean())
        std_bonus.append(b.std())
        mean_t_rho.append(np.mean([r["trained"]["partial_rho"] for r in rows]))
        mean_s_rho.append(np.mean([r["shuffled"]["partial_rho"] for r in rows]))

    ax = axes[0]
    ax.errorbar(ratios, mean_bonus, yerr=std_bonus, marker="o", capsize=4, color="steelblue")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("hidden_dim / input_dim (capacity ratio)")
    ax.set_ylabel("geometry bonus (trained − shuffled partial ρ)")
    ax.set_title("Geometry bonus vs capacity ratio")
    ax.set_xscale("log")

    ax = axes[1]
    ax.plot(ratios, mean_t_rho, "o-", label="trained", color="steelblue")
    ax.plot(ratios, mean_s_rho, "o--", label="shuffled", color="coral")
    ax.set_xlabel("hidden_dim / input_dim")
    ax.set_ylabel("partial ρ(stability, mode | distance)")
    ax.set_title("Trained vs shuffled ρ across capacity ratios")
    ax.set_xscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig("results/figures/capacity_ratio.png", dpi=150)
    plt.close(fig)

    print("\n==== CAPACITY RATIO SUMMARY ====")
    print(f"{'hidden':>7}  {'ratio':>6}  {'trained ρ':>10}  {'shuffled ρ':>11}  {'bonus':>8}")
    for h, r, t, s, b in zip(HIDDEN_DIMS, ratios, mean_t_rho, mean_s_rho, mean_bonus):
        print(f"{h:>7}  {r:>6.1f}  {t:>+10.3f}  {s:>+11.3f}  {b:>+8.3f}")
    print("\nSaved results/logs/capacity_ratio_results.json + results/figures/capacity_ratio.png")


if __name__ == "__main__":
    main()
