"""
Distance-conditioned lowrank vs fullrank stability comparison, swept over
seed x rank x scale.

Question: does lowrank perturbation stability differ from fullrank (at matched
Frobenius norm, same points) AFTER conditioning on distance to the nearest
gate hyperplane? If the effect vanishes under conditioning, raw stability
differences are just the distance confound.

Per config:
  - train fresh model (seeded)
  - lowrank + fullrank ensembles, matched norm, same family
  - per-point exact stability under both modes + min hyperplane distance
Analysis:
  - paired per-point delta = s_lowrank - s_fullrank
  - delta conditioned on distance deciles
  - partial Spearman rho(stability, mode | distance)
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
from datasets.synthetic_polytopes import generate_synthetic_polytopes
from cells.ensemble import generate_perturbation_ensemble
from probes.stability_metrics import compute_point_exact_stability

SWEEP = {
    "seeds": [0, 1, 2],
    "ranks": [1, 2, 4, 8],
    "scales": [0.02, 0.05, 0.1],
}

BASE_CONFIG = {
    "input_dim": 2,
    "hidden_dim": 64,
    "dataset_size": 10000,
    "train_steps": 3000,
    "ensemble_size": 64,
    "pert_family": "centroid_preserving",
    "n_deciles": 10,
}


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
    """Geometric distance |w.x + b| / ||w||, min over all first-layer units."""
    with torch.no_grad():
        W = model.fc1.weight
        b = model.fc1.bias
        z = X @ W.T + b
        norms = torch.norm(W, dim=1)
        return torch.min(torch.abs(z) / (norms + 1e-12), dim=1)[0]


def partial_spearman_mode_given_distance(s_low, s_full, dist):
    """
    Partial Spearman rho(stability, mode | distance) on the stacked paired
    sample. Distances are identical across modes, so rho(mode, dist) = 0 and
    the partial correlation reduces to rho_ym / sqrt(1 - rho_yd^2).
    """
    y = np.concatenate([s_low, s_full])
    m = np.concatenate([np.ones_like(s_low), np.zeros_like(s_full)])
    d = np.concatenate([dist, dist])

    rho_ym = spearmanr(y, m)[0]
    rho_yd = spearmanr(y, d)[0]
    return rho_ym / np.sqrt(1.0 - rho_yd ** 2)


def decile_conditioned_delta(delta, dist, n_deciles):
    """Mean paired delta within each distance decile."""
    edges = np.quantile(dist, np.linspace(0, 1, n_deciles + 1))
    bins = []
    for i in range(n_deciles):
        lo, hi = edges[i], edges[i + 1]
        mask = (dist >= lo) & (dist <= hi if i == n_deciles - 1 else dist < hi)
        d_bin = delta[mask]
        bins.append({
            "decile": i,
            "dist_lo": float(lo),
            "dist_hi": float(hi),
            "n": int(mask.sum()),
            "delta_mean": float(d_bin.mean()) if len(d_bin) else None,
            "delta_std": float(d_bin.std()) if len(d_bin) else None,
        })
    return bins


def run_config(seed, rank, scale, X, y, config):
    torch.manual_seed(seed)
    model = MLP(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"])
    train_model(model, X, y, steps=config["train_steps"])

    dist = min_hyperplane_distance(model, X).cpu().numpy()
    data_centroid = X.mean(dim=0)

    stab = {}
    for mode in ["lowrank", "fullrank"]:
        ens_config = {
            "mode": mode,
            "rank": rank,
            "scale": scale,
            "family": config["pert_family"],
            "data_centroid": data_centroid,
        }
        ensemble = generate_perturbation_ensemble(
            model, ensemble_size=config["ensemble_size"], perturbation_config=ens_config
        )
        stab[mode] = compute_point_exact_stability(model, ensemble, X).cpu().numpy()

    delta = stab["lowrank"] - stab["fullrank"]
    # Wilcoxon needs non-zero differences; skip if perturbation regime is degenerate
    nonzero = delta[delta != 0]
    wilcoxon_p = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 10 else None

    return {
        "seed": seed,
        "rank": rank,
        "scale": scale,
        "mean_stability_lowrank": float(stab["lowrank"].mean()),
        "mean_stability_fullrank": float(stab["fullrank"].mean()),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "raw_spearman_mode_stability": float(spearmanr(
            np.concatenate([stab["lowrank"], stab["fullrank"]]),
            np.concatenate([np.ones_like(dist), np.zeros_like(dist)]),
        )[0]),
        "partial_spearman_mode_given_dist": float(
            partial_spearman_mode_given_distance(stab["lowrank"], stab["fullrank"], dist)
        ),
        "spearman_delta_vs_dist": float(spearmanr(delta, dist)[0]),
        "wilcoxon_p_delta": wilcoxon_p,
        "decile_deltas": decile_conditioned_delta(delta, dist, config["n_deciles"]),
    }


def plot_results(results, config, out_dir):
    ranks = SWEEP["ranks"]
    scales = SWEEP["scales"]

    # 1. Decile-conditioned delta curves, one panel per scale, averaged over seeds
    fig, axes = plt.subplots(1, len(scales), figsize=(5 * len(scales), 4), sharey=True)
    for ax, scale in zip(np.atleast_1d(axes), scales):
        for rank in ranks:
            rows = [r for r in results if r["rank"] == rank and r["scale"] == scale]
            curves = np.array([[b["delta_mean"] for b in r["decile_deltas"]] for r in rows])
            mean_curve = curves.mean(axis=0)
            std_curve = curves.std(axis=0)
            x = np.arange(config["n_deciles"])
            ax.plot(x, mean_curve, marker="o", label=f"rank={rank}")
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_title(f"scale={scale}")
        ax.set_xlabel("distance decile (near → far)")
    np.atleast_1d(axes)[0].set_ylabel("Δ stability (lowrank − fullrank)")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle("Distance-conditioned stability difference (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sweep_decile_delta.png"), dpi=150)
    plt.close(fig)

    # 2. Partial correlation heatmap rank x scale, averaged over seeds
    grid = np.zeros((len(ranks), len(scales)))
    for i, rank in enumerate(ranks):
        for j, scale in enumerate(scales):
            vals = [r["partial_spearman_mode_given_dist"] for r in results
                    if r["rank"] == rank and r["scale"] == scale]
            grid[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(scales)), [str(s) for s in scales])
    ax.set_yticks(range(len(ranks)), [str(r) for r in ranks])
    ax.set_xlabel("scale")
    ax.set_ylabel("rank")
    ax.set_title("Partial Spearman ρ(stability, mode | distance)")
    for i in range(len(ranks)):
        for j in range(len(scales)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sweep_partial_corr_heatmap.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    config = BASE_CONFIG
    results = []
    n_total = len(SWEEP["seeds"]) * len(SWEEP["ranks"]) * len(SWEEP["scales"])
    i = 0
    for seed in SWEEP["seeds"]:
        # Dataset fixed per seed (seeded via numpy) so both modes and all
        # rank/scale settings see identical points
        np.random.seed(seed)
        X, y, _ = generate_synthetic_polytopes(
            n_samples=config["dataset_size"], dim=config["input_dim"]
        )
        for rank in SWEEP["ranks"]:
            for scale in SWEEP["scales"]:
                i += 1
                print(f"[{i}/{n_total}] seed={seed} rank={rank} scale={scale}")
                res = run_config(seed, rank, scale, X, y, config)
                print(
                    f"    Δ={res['delta_mean']:+.4f}  "
                    f"partial ρ={res['partial_spearman_mode_given_dist']:+.3f}  "
                    f"ρ(Δ,dist)={res['spearman_delta_vs_dist']:+.3f}"
                )
                results.append(res)

    output = {"sweep": SWEEP, "config": config, "results": results}
    with open("results/logs/sweep_results.json", "w") as f:
        json.dump(output, f, indent=2)

    plot_results(results, config, "results/figures")

    # Compact summary
    print("\n==== SUMMARY (mean over seeds) ====")
    print(f"{'rank':>5} {'scale':>6} {'Δ mean':>9} {'partial ρ':>10} {'ρ(Δ,dist)':>10}")
    for rank in SWEEP["ranks"]:
        for scale in SWEEP["scales"]:
            rows = [r for r in results if r["rank"] == rank and r["scale"] == scale]
            print(
                f"{rank:>5} {scale:>6} "
                f"{np.mean([r['delta_mean'] for r in rows]):>+9.4f} "
                f"{np.mean([r['partial_spearman_mode_given_dist'] for r in rows]):>+10.3f} "
                f"{np.mean([r['spearman_delta_vs_dist'] for r in rows]):>+10.3f}"
            )
    print("\nSaved results/logs/sweep_results.json + sweep figures.")


if __name__ == "__main__":
    main()
