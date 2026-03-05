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
from cells.select import score_structural_cells, select_top_cells
from controls.baselines import get_random_network_baseline

def train_model(model, X, y, steps=3000):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for step in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return model

def run_pipeline(model, X, y, cell_ids, config, label="main"):
    print(f"\n--- Running Pipeline: {label} ---")
    
    # 1. Base Region Extraction
    patterns = gate_patterns(model, X)
    pattern_hash = hash_patterns(patterns)
    cell_mapping, cells = extract_cells(X, pattern_hash, min_mass=config["min_mass"])
    print(f"Extracted {len(cells)} cells (min_mass={config['min_mass']})")
    
    # 2. Ensemble & Stability
    modes = ["lowrank", "fullrank"]
    results_by_mode = {}
    
    data_centroid = X.mean(dim=0)
    
    for mode in modes:
        print(f"Mode: {mode}")
        ens_config = {
            "mode": mode,
            "rank": config["rank"],
            "scale": config["scale"],
            "family": config["pert_family"],
            "data_centroid": data_centroid
        }
        ensemble = generate_perturbation_ensemble(model, ensemble_size=config["ensemble_size"], perturbation_config=ens_config)
        
        # Point stability
        s_ham = compute_point_hamming_stability(model, ensemble, X)
        s_exact = compute_point_exact_stability(model, ensemble, X)
        
        # Boundary band
        bound_idx, _ = identify_boundary_points(model, X, threshold=config["boundary_threshold"], boundary_mode=config["boundary_mode"])
        
        # Cell stability
        summaries = compute_cell_stability_summaries(cells, s_exact, s_ham, boundary_indices=bound_idx)
        
        # Adjacency
        edges = compute_cell_adjacency(X.cpu().numpy(), cell_mapping, k=16)
        degrees = get_cell_degrees(cells, edges)
        
        # Structural Selection
        for s in summaries: s["degree"] = degrees.get(s["cell_index"], 0)
        scored = score_structural_cells(summaries)
        top_cells = select_top_cells(scored, top_k=20)
        
        # Purity check if labels available
        if cell_ids is not None:
            for s in scored:
                m_idx = cells[s["cell_index"]]["member_indices"]
                c_vals = cell_ids[m_idx]
                counts = torch.bincount(c_vals)
                s["purity"] = counts.max().item() / s["mass"]
        
        results_by_mode[mode] = {
            "summaries": scored,
            "top_cells": top_cells,
            "point_exact_stability": s_exact.tolist(),
            "boundary_indices": bound_idx.tolist()
        }
    
    return results_by_mode, cells

def main():
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
    model = MLP(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"])
    print("Training main model...")
    train_model(model, X, y, steps=config["train_steps"])
    
    main_results, cells = run_pipeline(model, X, y, cell_ids, config, label="Trained Model")
    
    # Control: Random Network
    rand_model = get_random_network_baseline(config["input_dim"], config["hidden_dim"])
    rand_results, _ = run_pipeline(rand_model, X, y, cell_ids, config, label="Random Control")
    
    # Save Output
    output = {
        "config": config,
        "main_results": {m: {k: v for k, v in r.items() if "point" not in k} for m, r in main_results.items()},
        "control_results": {m: {k: v for k, v in r.items() if "point" not in k} for m, r in rand_results.items()}
    }
    with open("results/logs/structural_cells.json", "w") as f:
        json.dump(output, f)
        
    print("\n--- Summary ---")
    for mode in ["lowrank", "fullrank"]:
        res = main_results[mode]["summaries"]
        masses = [s["mass"] for s in res]
        stabs = [s["exact_mean"] for s in res]
        purities = [s["purity"] for s in res if "purity" in s]
        
        corr_m, _ = spearmanr(masses, stabs)
        corr_p = spearmanr(purities, stabs)[0] if purities else 0
        
        print(f"Mode {mode}: Cells={len(res)}, Corr(Mass, Stab)={corr_m:.3f}, Corr(Purity, Stab)={corr_p:.3f}")

    # Plotting (Basic check)
    plt.figure(figsize=(10,6))
    for i, mode in enumerate(["lowrank", "fullrank"]):
        res = main_results[mode]["summaries"]
        plt.scatter([s["mass"] for s in res], [s["exact_mean"] for s in res], label=mode, alpha=0.6)
    plt.xscale('log')
    plt.xlabel('Mass')
    plt.ylabel('Exact Stability')
    plt.legend()
    plt.title('Mass vs Stability (Structural Check)')
    plt.savefig('results/figures/mass_vs_stability_upgraded.png')

if __name__ == "__main__":
    main()
