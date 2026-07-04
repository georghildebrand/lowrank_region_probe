# experiments/run_polytopeness.py
"""
Polytopeness gradient: does the rank-1 geometry bonus decay continuously
as ground-truth boundary softness increases?

Exp 3 (label-shuffle) design repeated across a softness gradient on the 2D
checkerboard. softness=0 is the hard-polytope case where the bonus was
positive; GMM/images (soft boundaries) showed zero/negative bonus. If the
bonus decays monotonically with softness, the original hypothesis holds
exactly when the data partition is polytope-compatible.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from models.mlp import MLP
from datasets.synthetic_polytopes import generate_soft_checkerboard
from cells.ensemble import generate_perturbation_ensemble
from probes.stability_metrics import compute_point_exact_stability
from controls.baselines import get_label_shuffle_data
from analysis.conditioning import min_hyperplane_distance, partial_spearman_mode_given_distance

SMOKE = os.environ.get("SMOKE") == "1"

SOFTNESS_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.4]
SEEDS = [0, 1, 2]
SCALES = [0.02, 0.05, 0.1]
RANK = 1
ENSEMBLE_SIZE = 64
HIDDEN_DIM = 64
INPUT_DIM = 2
N_SAMPLES = 10000
TRAIN_STEPS = 3000

if SMOKE:
    SOFTNESS_LEVELS = [0.0, 0.2]
    SEEDS = [0]
    SCALES = [0.05]
    ENSEMBLE_SIZE = 8
    N_SAMPLES = 2000
    TRAIN_STEPS = 300


def train_model(model, X, y, steps):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model


def model_accuracy(model, X, y):
    with torch.no_grad():
        return ((model(X) > 0).float() == y).float().mean().item()


def eval_model(model, X, scale):
    dist = min_hyperplane_distance(model.fc1.weight, model.fc1.bias, X).cpu().numpy()
    data_centroid = X.mean(dim=0)
    stab = {}
    for mode in ["lowrank", "fullrank"]:
        ens_cfg = {
            "mode": mode, "rank": RANK, "scale": scale,
            "family": "centroid_preserving", "data_centroid": data_centroid,
        }
        ens = generate_perturbation_ensemble(model, ensemble_size=ENSEMBLE_SIZE,
                                             perturbation_config=ens_cfg)
        stab[mode] = compute_point_exact_stability(model, ens, X).cpu().numpy()
    return partial_spearman_mode_given_distance(stab["lowrank"], stab["fullrank"], dist)


def main():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    all_results = []
    n_cells = len(SEEDS) * len(SOFTNESS_LEVELS)
    i = 0
    for seed in SEEDS:
        for softness in SOFTNESS_LEVELS:
            i += 1
            X, y, _ = generate_soft_checkerboard(n_samples=N_SAMPLES,
                                                 softness=softness, seed=seed)

            torch.manual_seed(seed)
            m_trained = MLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
            train_model(m_trained, X, y, TRAIN_STEPS)

            torch.manual_seed(seed)  # identical init
            m_shuffled = MLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
            _, y_shuffled = get_label_shuffle_data(X, y)
            train_model(m_shuffled, X, y_shuffled, TRAIN_STEPS)

            acc_t = model_accuracy(m_trained, X, y)
            acc_s = model_accuracy(m_shuffled, X, y_shuffled)
            print(f"[{i}/{n_cells}] seed={seed} softness={softness} "
                  f"acc_trained={acc_t:.3f} acc_shuffled={acc_s:.3f}")

            for scale in SCALES:
                rho_t = eval_model(m_trained, X, scale)
                rho_s = eval_model(m_shuffled, X, scale)
                bonus = rho_t - rho_s
                print(f"    scale={scale}  trained ρ={rho_t:+.3f}  "
                      f"shuffled ρ={rho_s:+.3f}  bonus={bonus:+.3f}")
                all_results.append({
                    "seed": seed, "softness": softness, "scale": scale,
                    "rho_trained": float(rho_t), "rho_shuffled": float(rho_s),
                    "geometry_bonus": float(bonus),
                    "acc_trained": acc_t, "acc_shuffled": acc_s,
                })

    config = {"softness_levels": SOFTNESS_LEVELS, "seeds": SEEDS, "scales": SCALES,
              "rank": RANK, "ensemble_size": ENSEMBLE_SIZE, "hidden_dim": HIDDEN_DIM,
              "n_samples": N_SAMPLES, "train_steps": TRAIN_STEPS, "smoke": SMOKE}
    with open("results/logs/polytopeness_results.json", "w") as f:
        json.dump({"config": config, "results": all_results}, f, indent=2)

    # Figure: geometry bonus vs softness, one line per scale (mean ± std over seeds)
    fig, ax = plt.subplots(figsize=(8, 5))
    for scale in SCALES:
        means, stds = [], []
        for softness in SOFTNESS_LEVELS:
            vals = [r["geometry_bonus"] for r in all_results
                    if r["scale"] == scale and r["softness"] == softness]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.errorbar(SOFTNESS_LEVELS, means, yerr=stds, marker="o",
                    capsize=4, label=f"scale={scale}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("boundary softness")
    ax.set_ylabel("geometry bonus (trained − shuffled partial ρ)")
    ax.set_title("Polytopeness gradient: geometry bonus vs boundary softness")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/polytopeness_bonus.png", dpi=150)
    plt.close(fig)

    print("\n==== POLYTOPENESS SUMMARY (mean over seeds and scales) ====")
    print(f"{'softness':>8}  {'bonus':>8}  {'acc_trained':>11}")
    for softness in SOFTNESS_LEVELS:
        rows = [r for r in all_results if r["softness"] == softness]
        print(f"{softness:>8}  "
              f"{np.mean([r['geometry_bonus'] for r in rows]):>+8.3f}  "
              f"{np.mean([r['acc_trained'] for r in rows]):>11.3f}")
    print("\nSaved results/logs/polytopeness_results.json + figure.")


if __name__ == "__main__":
    main()
