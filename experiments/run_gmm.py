"""
Replication of distance-conditioned lowrank stability + label-shuffle on
5D Gaussian mixture dataset.

Tests whether the rank-1 geometry bonus found on the 2D synthetic polytope
generalises to a higher-dimensional dataset with a known latent partition.

Key change from 2D: input_dim=5, hidden_dim=128 to give the network enough
capacity to learn 4-cluster geometry.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from models.mlp import MLP
from datasets.gaussian_mixture import generate_gaussian_mixture
from cells.ensemble import generate_perturbation_ensemble
from probes.stability_metrics import compute_point_exact_stability
from controls.baselines import get_label_shuffle_data
from analysis.conditioning import min_hyperplane_distance, partial_spearman_mode_given_distance

SEEDS = [0, 1, 2, 3, 4]
SCALES = [0.02, 0.05, 0.1]
RANK = 1
ENSEMBLE_SIZE = 64
INPUT_DIM = 5
HIDDEN_DIM = 128
N_SAMPLES = 10000
TRAIN_STEPS = 5000
N_CLUSTERS = 4


def train_model(model, X, y, steps):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
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


def eval_model(model, X, scale):
    dist = min_hyperplane_distance(model.fc1.weight, model.fc1.bias, X).cpu().numpy()
    data_centroid = X.mean(dim=0)
    stab = {}
    for mode in ["lowrank", "fullrank"]:
        ens_cfg = {
            "mode": mode, "rank": RANK, "scale": scale,
            "family": "centroid_preserving", "data_centroid": data_centroid,
        }
        ens = generate_perturbation_ensemble(model, ensemble_size=ENSEMBLE_SIZE, perturbation_config=ens_cfg)
        stab[mode] = compute_point_exact_stability(model, ens, X).cpu().numpy()
    delta = stab["lowrank"] - stab["fullrank"]
    return {
        "partial_rho": partial_spearman_mode_given_distance(stab["lowrank"], stab["fullrank"], dist),
        "delta_mean": float(delta.mean()),
        "mean_stab_low": float(stab["lowrank"].mean()),
        "mean_stab_full": float(stab["fullrank"].mean()),
    }


def main():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    all_results = []
    n_total = len(SEEDS) * len(SCALES)
    i = 0

    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        X, y, cluster_ids = generate_gaussian_mixture(
            n_samples=N_SAMPLES, dim=INPUT_DIM, n_clusters=N_CLUSTERS,
            separation=3.0, seed=seed,
        )

        # Trained model
        m_trained = MLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        torch.manual_seed(seed)
        train_model(m_trained, X, y, TRAIN_STEPS)
        acc_trained = model_accuracy(m_trained, X, y)

        # Label-shuffled (identical init, shuffled labels)
        m_shuffled = MLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        torch.manual_seed(seed)
        _, y_shuffled = get_label_shuffle_data(X, y)
        train_model(m_shuffled, X, y_shuffled, TRAIN_STEPS)
        acc_shuffled = model_accuracy(m_shuffled, X, y_shuffled)

        print(f"seed={seed}: trained acc={acc_trained:.3f}, shuffled acc={acc_shuffled:.3f}")

        for scale in SCALES:
            i += 1
            print(f"  [{i}/{n_total}] scale={scale}")
            r_trained = eval_model(m_trained, X, scale)
            r_shuffled = eval_model(m_shuffled, X, scale)
            geo_bonus = r_trained["partial_rho"] - r_shuffled["partial_rho"]
            print(f"    trained ρ={r_trained['partial_rho']:+.3f}  "
                  f"shuffled ρ={r_shuffled['partial_rho']:+.3f}  "
                  f"bonus={geo_bonus:+.3f}")
            all_results.append({
                "seed": seed, "scale": scale,
                "trained_acc": acc_trained, "shuffled_acc": acc_shuffled,
                "trained": r_trained, "shuffled": r_shuffled,
                "geometry_bonus": geo_bonus,
            })

    with open("results/logs/gmm_results.json", "w") as f:
        json.dump({"seeds": SEEDS, "scales": SCALES, "rank": RANK,
                   "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM,
                   "results": all_results}, f, indent=2)

    # Wilcoxon test: are geometry bonuses consistently > 0?
    bonuses = np.array([r["geometry_bonus"] for r in all_results])
    nonzero = bonuses[bonuses != 0]
    wx_stat, wx_p = wilcoxon(nonzero) if len(nonzero) >= 10 else (None, None)

    # Summary
    print("\n==== GMM 5D SUMMARY (mean over seeds) ====")
    print(f"{'scale':>6}  {'trained ρ':>10}  {'shuffled ρ':>11}  {'bonus':>8}")
    for scale in SCALES:
        rows = [r for r in all_results if r["scale"] == scale]
        t = np.mean([r["trained"]["partial_rho"] for r in rows])
        s = np.mean([r["shuffled"]["partial_rho"] for r in rows])
        b = np.mean([r["geometry_bonus"] for r in rows])
        print(f"{scale:>6}  {t:>+10.3f}  {s:>+11.3f}  {b:>+8.3f}")

    positives = sum(1 for r in all_results if r["geometry_bonus"] > 0)
    print(f"\nBonus positive in {positives}/{len(all_results)} configs")
    if wx_p is not None:
        print(f"Wilcoxon p={wx_p:.4f} (H0: bonus=0)")

    # Compare 2D poly vs 5D GMM at scale=0.1
    try:
        with open("results/logs/label_shuffle_results.json") as f:
            d2 = json.load(f)
        rows_2d = [r for r in d2["results"] if r["scale"] == 0.1]
        t2d = np.mean([r["trained"]["partial_rho"] for r in rows_2d])
        s2d = np.mean([r["shuffled"]["partial_rho"] for r in rows_2d])
        rows_5d = [r for r in all_results if r["scale"] == 0.1]
        t5d = np.mean([r["trained"]["partial_rho"] for r in rows_5d])
        s5d = np.mean([r["shuffled"]["partial_rho"] for r in rows_5d])
        print(f"\n=== Generalization (scale=0.1, rank=1) ===")
        print(f"{'':10}  {'trained ρ':>10}  {'shuffled ρ':>11}  {'bonus':>8}")
        print(f"{'2D poly':10}  {t2d:>+10.3f}  {s2d:>+11.3f}  {t2d-s2d:>+8.3f}")
        print(f"{'5D GMM':10}  {t5d:>+10.3f}  {s5d:>+11.3f}  {t5d-s5d:>+8.3f}")
    except FileNotFoundError:
        pass

    # Plot: trained vs shuffled partial rho across both datasets at scale=0.1
    rows_5d_01 = [r for r in all_results if r["scale"] == 0.1]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(SEEDS))
    ax.bar(x - 0.2, [r["trained"]["partial_rho"] for r in rows_5d_01], 0.4,
           label="trained (5D GMM)", color="steelblue", alpha=0.85)
    ax.bar(x + 0.2, [r["shuffled"]["partial_rho"] for r in rows_5d_01], 0.4,
           label="shuffled (5D GMM)", color="coral", alpha=0.85)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x, SEEDS)
    ax.set_xlabel("seed")
    ax.set_ylabel("partial ρ(stability, mode | distance)")
    ax.set_title("5D Gaussian mixture: rank-1 geometry bonus (scale=0.1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/gmm_label_shuffle.png", dpi=150)
    plt.close(fig)

    print("\nSaved results/logs/gmm_results.json + results/figures/gmm_label_shuffle.png")


if __name__ == "__main__":
    main()
