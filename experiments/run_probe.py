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
from cells.regions import gate_patterns, hash_patterns, extract_cells
from probes.stability_metrics import compute_point_hamming_stability, compute_point_exact_stability, identify_boundary_points
from cells.stability import compute_cell_stability_summaries
from cells.adjacency import compute_cell_adjacency, get_cell_degrees
from cells.select import score_structural_cells

def create_random_model(input_dim, hidden_dim, seed=42):
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim)
    # Local generator: reproducible baseline without resetting the global RNG
    # (a mid-run torch.manual_seed would couple all downstream ensemble draws)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=1.0, generator=gen)
    return model

def train_model(model, X, y, steps=3000):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for step in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model

def evaluate_model(model, X, cell_ids, config):
    # 1. Base Region Extraction
    patterns = gate_patterns(model, X)
    pattern_hash = hash_patterns(patterns)
    cell_mapping, cells = extract_cells(X, pattern_hash, min_mass=config["min_mass"])

    # 2. Distance to nearest hyperplane (Step 11)
    # NOTE: min over ALL hyperplanes — intentionally broader than the boundary
    # band below, which uses config["boundary_mode"] (e.g. locally_active)
    with torch.no_grad():
        W = model.fc1.weight  # [hidden, input]
        b = model.fc1.bias    # [hidden]
        activations = torch.matmul(X, W.t()) + b  # [N, hidden]
        # Geometric distance: |w.x + b| / ||w||
        norms = torch.norm(W, dim=1)
        distances = torch.abs(activations) / norms
        min_distances = torch.min(distances, dim=1)[0]

    # 3. Base-model-only quantities (mode-independent)
    bound_idx, _ = identify_boundary_points(model, X, threshold=config["boundary_threshold"], boundary_mode=config["boundary_mode"])
    edges = compute_cell_adjacency(X.cpu().numpy(), cell_mapping, k=16)
    degrees = get_cell_degrees(cells, edges)

    # 4. Ensemble & Stability
    modes = ["lowrank", "fullrank"]
    results_by_mode = {}

    data_centroid = X.mean(dim=0)

    for mode in modes:
        ens_config = {
            "mode": mode,
            "rank": config["rank"],
            "scale": config["scale"],
            "family": config["pert_family"],
            "data_centroid": data_centroid
        }
        ensemble = generate_perturbation_ensemble(model, ensemble_size=config["ensemble_size"], perturbation_config=ens_config)
        
        # Point stability
        s_exact = compute_point_exact_stability(model, ensemble, X)
        s_ham = compute_point_hamming_stability(model, ensemble, X)

        # Cell stability
        summaries = compute_cell_stability_summaries(cells, s_exact, s_ham, boundary_indices=bound_idx)

        # Structural Selection & Stats
        for s in summaries:
            s["degree"] = degrees.get(s["cell_index"], 0)
            # Purity check
            m_idx = cells[s["cell_index"]]["member_indices"]
            if cell_ids is not None:
                c_vals = cell_ids[m_idx]
                counts = torch.bincount(c_vals)
                s["purity"] = counts.max().item() / s["mass"]
            
            # Boundary stability specifically
            # (Already handles inside compute_cell_stability_summaries as boundary_exact_mean)
        
        scored = score_structural_cells(summaries)
        
        # Global metrics for this mode
        mean_s = s_exact.mean().item()
        # None (not 0.0) when no boundary points — matches cell-level convention
        boundary_s = s_exact[bound_idx].mean().item() if len(bound_idx) > 0 else None
        
        # Step 11: corr(stability, distance)
        corr_dist_stab, _ = spearmanr(min_distances.cpu().numpy(), s_exact.cpu().numpy())
        
        results_by_mode[mode] = {
            "summaries": scored,
            "mean_exact_stability": mean_s,
            "boundary_exact_stability": boundary_s,
            "dist_stab_corr": corr_dist_stab,
            "point_exact_stability": s_exact.tolist(),
            "min_distances": min_distances.tolist(),
            "n_cells": len(cells)
        }
    
    return results_by_mode

def main():
    # Global seeds: dataset, training, ensemble draws all reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    config = {
        "input_dim": 2,
        "hidden_dim": 64,
        "dataset_size": 10000,
        "train_steps": 3000,
        "rank": 2,
        "scale": 0.05,
        "ensemble_size": 64,
        "pert_family": "centroid_preserving",
        "min_mass": 10,
        "boundary_threshold": 0.01,
        "boundary_mode": "locally_active"
    }
    
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Data
    X, y, cell_ids = generate_synthetic_polytopes(n_samples=config["dataset_size"], dim=config["input_dim"])
    
    # Main Model
    print("Training main model...")
    trained_model = MLP(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"])
    train_model(trained_model, X, y, steps=config["train_steps"])
    
    print("Creating random model...")
    random_model = create_random_model(config["input_dim"], config["hidden_dim"])
    
    print("Evaluating trained model...")
    trained_results = evaluate_model(trained_model, X, cell_ids, config)

    print("Evaluating random model...")
    random_results = evaluate_model(random_model, X, cell_ids, config)
    
    # Save Output
    output = {
        "config": config,
        "trained_model": {m: {k: v for k, v in r.items() if "point" not in k and "min_distances" not in k} for m, r in trained_results.items()},
        "random_model": {m: {k: v for k, v in r.items() if "point" not in k and "min_distances" not in k} for m, r in random_results.items()}
    }
    with open("results/logs/structural_cells.json", "w") as f:
        json.dump(output, f, indent=2)
    
    def print_stats(label, results_by_mode):
        print(f"\n---- {label} ----")
        for mode in ["lowrank", "fullrank"]:
            res = results_by_mode[mode]
            summaries = res["summaries"]
            
            masses = [s["mass"] for s in summaries]
            stabs = [s["exact_mean"] for s in summaries]
            purities = [s["purity"] for s in summaries if "purity" in s]
            
            corr_m, _ = spearmanr(masses, stabs)
            corr_p = spearmanr(purities, stabs)[0] if purities else 0.0
            
            boundary_str = f"{res['boundary_exact_stability']:.4f}" if res['boundary_exact_stability'] is not None else "n/a (no boundary points)"
            print(f"[{mode}]")
            print(f"  mean_exact_stability: {res['mean_exact_stability']:.4f}")
            print(f"  boundary_exact_stability: {boundary_str}")
            print(f"  mass_stability_corr: {corr_m:.3f}")
            print(f"  purity_stability_corr: {corr_p:.3f}")
            print(f"  dist_stability_corr: {res['dist_stab_corr']:.3f}")

    print_stats("TRAINED MODEL", trained_results)
    print_stats("RANDOM MODEL", random_results)

    # Visualization (Step 8)
    plt.figure(figsize=(12, 5))
    
    # Trained Histogram
    plt.subplot(1, 2, 1)
    plt.hist(trained_results["lowrank"]["point_exact_stability"], bins=30, alpha=0.7, label="Trained (LowRank)")
    plt.hist(random_results["lowrank"]["point_exact_stability"], bins=30, alpha=0.7, label="Random (LowRank)")
    plt.title("Exact Stability Distribution (LowRank)")
    plt.xlabel("Exact Stability")
    plt.ylabel("Count")
    plt.legend()
    
    # Mass vs Stability Comparison
    plt.subplot(1, 2, 2)
    t_sum = trained_results["lowrank"]["summaries"]
    r_sum = random_results["lowrank"]["summaries"]
    plt.scatter([s["mass"] for s in t_sum], [s["exact_mean"] for s in t_sum], label="Trained", alpha=0.5)
    plt.scatter([s["mass"] for s in r_sum], [s["exact_mean"] for s in r_sum], label="Random", alpha=0.5)
    plt.xscale('log')
    plt.xlabel("Mass")
    plt.ylabel("Mean Stability")
    plt.title("Mass vs Stability Comparison")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/figures/stability_distribution_control.png')
    print("\nSaved control visualization to results/figures/stability_distribution_control.png")

if __name__ == "__main__":
    main()
