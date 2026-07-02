"""
Re-score the trained-vs-shuffled comparison with the REGION-IDENTITY metric
on both existing datasets (2D polytope, 5D GMM).

Question: does the geometry bonus survive when 'stability' means the
partition staying together, rather than pointwise exact gate match?
Config mirrors run_label_shuffle.py / run_gmm.py at rank=1.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from scipy.stats import wilcoxon

from models.mlp import MLP
from datasets.synthetic_polytopes import generate_synthetic_polytopes
from datasets.gaussian_mixture import generate_gaussian_mixture
from cells.ensemble import generate_perturbation_ensemble
from probes.region_identity import compute_region_identity_stability
from controls.baselines import get_label_shuffle_data
from analysis.conditioning import partial_spearman_mode_given_distance, min_hyperplane_distance

SEEDS = [0, 1, 2, 3, 4]
SCALES = [0.05, 0.1]
RANK = 1
ENSEMBLE_SIZE = 64
MIN_MASS = 10

DATASETS = {
    "poly2d": dict(input_dim=2, hidden_dim=64, train_steps=3000, lr=0.01),
    "gmm5d": dict(input_dim=5, hidden_dim=128, train_steps=5000, lr=0.005),
}


def make_data(name, seed):
    np.random.seed(seed)
    if name == "poly2d":
        X, y, _ = generate_synthetic_polytopes(n_samples=10000, dim=2)
    else:
        X, y, _ = generate_gaussian_mixture(n_samples=10000, dim=5, n_clusters=4,
                                            separation=3.0, seed=seed)
    return X, y


def train(model, X, y, steps, lr):
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        crit(model(X), y).backward()
        opt.step()
    return model


def eval_region_identity(model, X, scale):
    dist = min_hyperplane_distance(model.fc1.weight, model.fc1.bias, X).cpu().numpy()
    centroid = X.mean(dim=0)
    stab = {}
    for mode in ["lowrank", "fullrank"]:
        cfg = {"mode": mode, "rank": RANK, "scale": scale,
               "family": "centroid_preserving", "data_centroid": centroid}
        ens = generate_perturbation_ensemble(model, ensemble_size=ENSEMBLE_SIZE,
                                             perturbation_config=cfg)
        scores, _ = compute_region_identity_stability(model, ens, X, min_mass=MIN_MASS)
        stab[mode] = scores
    return {
        "partial_rho": partial_spearman_mode_given_distance(stab["lowrank"], stab["fullrank"], dist),
        "mean_ri_low": float(np.nanmean(stab["lowrank"])),
        "mean_ri_full": float(np.nanmean(stab["fullrank"])),
        "n_valid": int(np.sum(~np.isnan(stab["lowrank"]))),
    }


def main():
    os.makedirs("results/logs", exist_ok=True)
    all_results = []
    for ds_name, cfg in DATASETS.items():
        for seed in SEEDS:
            X, y = make_data(ds_name, seed)
            torch.manual_seed(seed)
            m_t = MLP(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"])
            train(m_t, X, y, cfg["train_steps"], cfg["lr"])

            torch.manual_seed(seed)
            m_s = MLP(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"])
            _, y_sh = get_label_shuffle_data(X, y)
            train(m_s, X, y_sh, cfg["train_steps"], cfg["lr"])

            for scale in SCALES:
                r_t = eval_region_identity(m_t, X, scale)
                r_s = eval_region_identity(m_s, X, scale)
                bonus = r_t["partial_rho"] - r_s["partial_rho"]
                print(f"[{ds_name} seed={seed} scale={scale}] "
                      f"trained ρ={r_t['partial_rho']:+.3f} "
                      f"shuffled ρ={r_s['partial_rho']:+.3f} bonus={bonus:+.3f}")
                all_results.append({"dataset": ds_name, "seed": seed, "scale": scale,
                                    "trained": r_t, "shuffled": r_s,
                                    "geometry_bonus": bonus})

    with open("results/logs/region_identity_results.json", "w") as f:
        json.dump({"seeds": SEEDS, "scales": SCALES, "rank": RANK,
                   "min_mass": MIN_MASS, "results": all_results}, f, indent=2)

    print("\n==== REGION-IDENTITY SUMMARY (mean over seeds) ====")
    for ds_name in DATASETS:
        for scale in SCALES:
            rows = [r for r in all_results if r["dataset"] == ds_name and r["scale"] == scale]
            t = np.mean([r["trained"]["partial_rho"] for r in rows])
            s = np.mean([r["shuffled"]["partial_rho"] for r in rows])
            print(f"{ds_name:8s} scale={scale}: trained={t:+.3f} shuffled={s:+.3f} bonus={t-s:+.3f}")
    bonuses = np.array([r["geometry_bonus"] for r in all_results])
    nz = bonuses[bonuses != 0]
    if len(nz) >= 10:
        print(f"Wilcoxon p={wilcoxon(nz).pvalue:.5f} over {len(bonuses)} configs")


if __name__ == "__main__":
    main()
