"""
Label-shuffle control for the distance-conditioned lowrank stability experiment.

If rank-1 lowrank advantage (partial ρ > 0) is purely a geometry artifact,
a model trained on shuffled labels should show the same effect — it learns
similar hyperplane structure but no real partition. If trained >> shuffled,
the hypothesis gains a second leg: the effect requires LEARNED geometry.

Design:
  - For each seed: train model on real labels, train on shuffled labels
  - Both: rank-1 lowrank vs fullrank, distance-conditioned comparison (scale sweep)
  - Key metric: partial ρ(stability, mode | distance) for each condition
  - Secondary: Δ_trained − Δ_shuffled as the "geometry bonus"
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from models.mlp import MLP
from datasets.synthetic_polytopes import generate_synthetic_polytopes
from cells.ensemble import generate_perturbation_ensemble
from probes.stability_metrics import compute_point_exact_stability
from controls.baselines import get_label_shuffle_data

SEEDS = [0, 1, 2, 3, 4]
SCALES = [0.02, 0.05, 0.1]
RANK = 1  # most informative regime from sweep
ENSEMBLE_SIZE = 64
HIDDEN_DIM = 64
INPUT_DIM = 2
N_SAMPLES = 10000
TRAIN_STEPS = 3000


def train_model(model, X, y, steps):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model


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


def decile_delta(s_low, s_full, dist, n=10):
    delta = s_low - s_full
    edges = np.quantile(dist, np.linspace(0, 1, n + 1))
    means = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        mask = (dist >= lo) & (dist <= hi if i == n - 1 else dist < hi)
        means.append(float(delta[mask].mean()) if mask.sum() > 0 else None)
    return means


def eval_model(model, X, scale, seed_offset):
    dist = min_hyperplane_distance(model, X).cpu().numpy()
    data_centroid = X.mean(dim=0)
    stab = {}
    for mode in ["lowrank", "fullrank"]:
        ens_cfg = {
            "mode": mode, "rank": RANK, "scale": scale,
            "family": "centroid_preserving", "data_centroid": data_centroid,
        }
        ens = generate_perturbation_ensemble(model, ensemble_size=ENSEMBLE_SIZE, perturbation_config=ens_cfg)
        stab[mode] = compute_point_exact_stability(model, ens, X).cpu().numpy()
    return {
        "dist": dist,
        "stab_low": stab["lowrank"],
        "stab_full": stab["fullrank"],
        "partial_rho": partial_spearman_mode_given_distance(stab["lowrank"], stab["fullrank"], dist),
        "delta_mean": float((stab["lowrank"] - stab["fullrank"]).mean()),
        "decile_deltas": decile_delta(stab["lowrank"], stab["fullrank"], dist),
    }


def main():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    all_results = []
    n = len(SEEDS) * len(SCALES)
    i = 0
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        X, y, _ = generate_synthetic_polytopes(n_samples=N_SAMPLES, dim=INPUT_DIM)

        # Trained model
        m_trained = MLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        torch.manual_seed(seed)
        train_model(m_trained, X, y, TRAIN_STEPS)

        # Label-shuffled model (same init, different labels)
        m_shuffled = MLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        torch.manual_seed(seed)  # identical init
        _, y_shuffled = get_label_shuffle_data(X, y)
        train_model(m_shuffled, X, y_shuffled, TRAIN_STEPS)

        for scale in SCALES:
            i += 1
            print(f"[{i}/{n}] seed={seed} scale={scale}")
            r_trained = eval_model(m_trained, X, scale, seed)
            r_shuffled = eval_model(m_shuffled, X, scale, seed)
            geo_bonus = r_trained["partial_rho"] - r_shuffled["partial_rho"]
            print(f"    trained partial ρ={r_trained['partial_rho']:+.3f}  "
                  f"shuffled partial ρ={r_shuffled['partial_rho']:+.3f}  "
                  f"geometry bonus={geo_bonus:+.3f}")
            all_results.append({
                "seed": seed, "scale": scale,
                "trained": {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in r_trained.items() if k != "dist"},
                "shuffled": {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in r_shuffled.items() if k != "dist"},
                "geometry_bonus": geo_bonus,
            })

    with open("results/logs/label_shuffle_results.json", "w") as f:
        json.dump({"seeds": SEEDS, "scales": SCALES, "rank": RANK, "results": all_results}, f, indent=2)

    # Plot 1: partial ρ trained vs shuffled per scale
    fig, axes = plt.subplots(1, len(SCALES), figsize=(5 * len(SCALES), 4), sharey=True)
    for ax, scale in zip(axes, SCALES):
        rows = [r for r in all_results if r["scale"] == scale]
        t_rhos = [r["trained"]["partial_rho"] for r in rows]
        s_rhos = [r["shuffled"]["partial_rho"] for r in rows]
        x = np.arange(len(SEEDS))
        ax.bar(x - 0.2, t_rhos, 0.4, label="trained", color="steelblue", alpha=0.8)
        ax.bar(x + 0.2, s_rhos, 0.4, label="shuffled", color="coral", alpha=0.8)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"scale={scale}")
        ax.set_xlabel("seed")
        ax.set_xticks(x, SEEDS)
    axes[0].set_ylabel("partial ρ(stability, mode | distance)")
    axes[0].legend()
    fig.suptitle("Trained vs label-shuffled: rank-1 lowrank advantage after distance conditioning")
    fig.tight_layout()
    fig.savefig("results/figures/label_shuffle_partial_rho.png", dpi=150)
    plt.close(fig)

    # Plot 2: geometry bonus (trained − shuffled) per scale, with per-seed bars
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(SCALES))
    for si, seed in enumerate(SEEDS):
        bonuses = [r["geometry_bonus"] for r in all_results if r["seed"] == seed]
        ax.bar(x + si * 0.15, bonuses, 0.14, alpha=0.75, label=f"seed={seed}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x + 0.3, [f"scale={s}" for s in SCALES])
    ax.set_ylabel("geometry bonus (trained − shuffled partial ρ)")
    ax.set_title("Geometry bonus: does training matter beyond geometry?")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/label_shuffle_geometry_bonus.png", dpi=150)
    plt.close(fig)

    # Summary
    print("\n==== SUMMARY (mean over seeds) ====")
    print(f"{'scale':>6}  {'trained ρ':>10}  {'shuffled ρ':>11}  {'geo bonus':>10}")
    for scale in SCALES:
        rows = [r for r in all_results if r["scale"] == scale]
        print(f"{scale:>6}  "
              f"{np.mean([r['trained']['partial_rho'] for r in rows]):>+10.3f}  "
              f"{np.mean([r['shuffled']['partial_rho'] for r in rows]):>+11.3f}  "
              f"{np.mean([r['geometry_bonus'] for r in rows]):>+10.3f}")
    print("\nSaved results/logs/label_shuffle_results.json + figures.")


if __name__ == "__main__":
    main()
