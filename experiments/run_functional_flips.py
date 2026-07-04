# experiments/run_functional_flips.py
"""
Function-weighted gate flips: are trained networks' gate flips benign?

Prior experiments count gate flips (Hamming). But a flip is only harmful if
it changes the function. Here both metrics come from the SAME perturbation
ensemble per point: Hamming gate stability AND functional stability
(-mean |Δlogit|). Trained vs label-shuffled, rank-1 lowrank vs fullrank,
on 2D checkerboard (Hamming bonus was positive) and 5D GMM (Hamming bonus
was ~0/negative). Rescue prediction: functional geometry bonus positive on
GMM even where Hamming bonus is not, and trained flips carry less |Δlogit|
per flip than shuffled flips.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from models.mlp import MLP
from datasets.synthetic_polytopes import generate_synthetic_polytopes
from datasets.gaussian_mixture import generate_gaussian_mixture
from cells.ensemble_batched import (generate_perturbation_batch,
                                    batched_gate_patterns, batched_logits)
from controls.baselines import get_label_shuffle_data
from analysis.conditioning import min_hyperplane_distance, partial_spearman_mode_given_distance

SMOKE = os.environ.get("SMOKE") == "1"

SEEDS = [0, 1, 2]
SCALES = [0.05, 0.1]
RANK = 1
ENSEMBLE_SIZE = 64
N_SAMPLES = 10000

DATASETS = {
    "checkerboard2d": {"input_dim": 2, "hidden_dim": 64, "train_steps": 3000, "lr": 0.01},
    "gmm5d": {"input_dim": 5, "hidden_dim": 128, "train_steps": 5000, "lr": 0.005},
}

if SMOKE:
    SEEDS = [0]
    SCALES = [0.05]
    ENSEMBLE_SIZE = 8
    N_SAMPLES = 2000
    DATASETS = {k: {**v, "train_steps": 300} for k, v in DATASETS.items()}


def load_dataset(name, seed):
    np.random.seed(seed)
    if name == "checkerboard2d":
        X, y, _ = generate_synthetic_polytopes(n_samples=N_SAMPLES, dim=2)
    elif name == "gmm5d":
        X, y, _ = generate_gaussian_mixture(n_samples=N_SAMPLES, dim=5,
                                            n_clusters=4, seed=seed)
    else:
        raise ValueError(name)
    return X, y


def train_model(model, X, y, steps, lr):
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
        return ((model(X) > 0).float() == y).float().mean().item()


def eval_model(model, X, scale):
    """Both stability metrics from the same ensembles, per mode."""
    W0, b0 = model.fc1.weight.data, model.fc1.bias.data
    W2, b2 = model.fc2.weight.data, model.fc2.bias.data
    centroid = X.mean(dim=0)
    dist = min_hyperplane_distance(W0, b0, X).cpu().numpy()
    with torch.no_grad():
        base_logit = model(X).squeeze(1)          # [N]
        base_gates = model.gate_pattern(X)        # [N,h]

    out = {"dist": dist}
    for mode in ["lowrank", "fullrank"]:
        W, b = generate_perturbation_batch(W0, b0, ENSEMBLE_SIZE, RANK, scale,
                                           mode, centroid)
        gates = batched_gate_patterns(X, W, b)               # [E,N,h] bool
        logits = batched_logits(X, W, b, W2, b2)             # [E,N]

        flips = (gates != base_gates[None]).sum(dim=2).float()      # [E,N]
        dlogit = (logits - base_logit[None]).abs()                   # [E,N]

        hamming_stab = 1.0 - flips.mean(dim=0).cpu().numpy() / base_gates.shape[1]
        func_stab = (-dlogit.mean(dim=0)).cpu().numpy()

        flipped = flips > 0
        per_flip = (dlogit[flipped] / flips[flipped])
        per_flip_impact = float(per_flip.mean()) if flipped.any() else float("nan")

        out[mode] = {
            "hamming_stab": hamming_stab,
            "func_stab": func_stab,
            "per_flip_impact": per_flip_impact,
            "flip_rate": float(flips.mean()),
        }
    return out


def main():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    all_results = []
    for ds_name, cfg in DATASETS.items():
        for seed in SEEDS:
            X, y = load_dataset(ds_name, seed)

            torch.manual_seed(seed)
            m_trained = MLP(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"])
            train_model(m_trained, X, y, cfg["train_steps"], cfg["lr"])

            torch.manual_seed(seed)  # identical init
            m_shuffled = MLP(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"])
            _, y_shuffled = get_label_shuffle_data(X, y)
            train_model(m_shuffled, X, y_shuffled, cfg["train_steps"], cfg["lr"])

            acc_t = model_accuracy(m_trained, X, y)
            print(f"[{ds_name} seed={seed}] acc_trained={acc_t:.3f}")

            for scale in SCALES:
                r_t = eval_model(m_trained, X, scale)
                r_s = eval_model(m_shuffled, X, scale)

                row = {"dataset": ds_name, "seed": seed, "scale": scale,
                       "acc_trained": acc_t}
                for metric in ["hamming_stab", "func_stab"]:
                    rho_t = partial_spearman_mode_given_distance(
                        r_t["lowrank"][metric], r_t["fullrank"][metric], r_t["dist"])
                    rho_s = partial_spearman_mode_given_distance(
                        r_s["lowrank"][metric], r_s["fullrank"][metric], r_s["dist"])
                    key = "hamming" if metric == "hamming_stab" else "functional"
                    row[f"rho_trained_{key}"] = float(rho_t)
                    row[f"rho_shuffled_{key}"] = float(rho_s)
                    row[f"bonus_{key}"] = float(rho_t - rho_s)
                for mode in ["lowrank", "fullrank"]:
                    row[f"per_flip_trained_{mode}"] = r_t[mode]["per_flip_impact"]
                    row[f"per_flip_shuffled_{mode}"] = r_s[mode]["per_flip_impact"]
                    row[f"flip_rate_trained_{mode}"] = r_t[mode]["flip_rate"]
                    row[f"flip_rate_shuffled_{mode}"] = r_s[mode]["flip_rate"]
                all_results.append(row)
                print(f"    scale={scale}  bonus_hamming={row['bonus_hamming']:+.3f}  "
                      f"bonus_functional={row['bonus_functional']:+.3f}  "
                      f"per_flip t/s (r1)={row['per_flip_trained_lowrank']:.3f}/"
                      f"{row['per_flip_shuffled_lowrank']:.3f}")

    config = {"seeds": SEEDS, "scales": SCALES, "rank": RANK,
              "ensemble_size": ENSEMBLE_SIZE, "n_samples": N_SAMPLES,
              "datasets": DATASETS, "smoke": SMOKE}
    with open("results/logs/functional_flips_results.json", "w") as f:
        json.dump({"config": config, "results": all_results}, f, indent=2)

    # Figure: hamming vs functional geometry bonus per dataset (mean +/- std)
    fig, ax = plt.subplots(figsize=(8, 5))
    ds_names = list(DATASETS.keys())
    x = np.arange(len(ds_names))
    for i, key in enumerate(["hamming", "functional"]):
        means, stds = [], []
        for ds in ds_names:
            vals = [r[f"bonus_{key}"] for r in all_results if r["dataset"] == ds]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.bar(x + (i - 0.5) * 0.35, means, 0.35, yerr=stds, capsize=4,
               label=f"{key} bonus", alpha=0.85)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.set_ylabel("geometry bonus (trained - shuffled partial rho)")
    ax.set_title("Hamming vs functional geometry bonus")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/functional_flips_bonus.png", dpi=150)
    plt.close(fig)

    print("\n==== FUNCTIONAL FLIPS SUMMARY (mean over seeds and scales) ====")
    print(f"{'dataset':>15}  {'bonus_hamming':>13}  {'bonus_functional':>16}  "
          f"{'per_flip t (r1)':>15}  {'per_flip s (r1)':>15}")
    for ds in ds_names:
        rows = [r for r in all_results if r["dataset"] == ds]
        print(f"{ds:>15}  "
              f"{np.mean([r['bonus_hamming'] for r in rows]):>+13.3f}  "
              f"{np.mean([r['bonus_functional'] for r in rows]):>+16.3f}  "
              f"{np.nanmean([r['per_flip_trained_lowrank'] for r in rows]):>15.3f}  "
              f"{np.nanmean([r['per_flip_shuffled_lowrank'] for r in rows]):>15.3f}")
    print("\nSaved results/logs/functional_flips_results.json + figure.")


if __name__ == "__main__":
    main()
