import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from models.mlp import MLP
from datasets.synthetic_polytopes import generate_synthetic_polytopes
from datasets.circle_dataset import generate_circle_dataset
from probes.lowrank_ensemble import generate_perturbation_ensemble
from probes.stability_metrics import compute_point_stability, compute_region_stability, compute_boundary_stability
from analysis.region_extraction import extract_regions

def train_model(model, X, y, steps=3000):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

def run_experiment():
    # Parameters
    INPUT_DIM = 2 
    HIDDEN_WIDTH = 64
    RANK = 2
    ENSEMBLE_SIZE = 64
    PERTURBATION_SCALE = 0.05
    DATASET_SIZE = 10000
    TRAINING_STEPS = 3000
    
    # Create directories
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # 1. Generate Dataset
    X, y, cell_ids = generate_synthetic_polytopes(n_samples=DATASET_SIZE, dim=INPUT_DIM)
    
    # 2. Train Base Model
    model = MLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_WIDTH)
    print("Training base model...")
    train_model(model, X, y, steps=TRAINING_STEPS)
    
    modes = ["lowrank", "fullrank"]
    all_results = {}
    
    for mode in modes:
        print(f"\nProcessing mode: {mode}...")
        
        # 4. Generate Ensemble
        print(f"Generating {mode} ensemble...")
        ensemble = generate_perturbation_ensemble(
            model, mode=mode, rank=RANK, scale=PERTURBATION_SCALE, ensemble_size=ENSEMBLE_SIZE
        )
        
        # 5. Compute Stability Metrics
        print("Computing stability metrics...")
        point_stability = compute_point_stability(model, ensemble, X)
        
        # 5b. Compute Boundary Stability
        print("Evaluating boundary sensitivity...")
        boundary_threshold = 0.05
        boundary_stab, boundary_idx = compute_boundary_stability(model, ensemble, X, threshold=boundary_threshold)
        
        # 6. Extract Regions and Mass (passing cell_ids for purity)
        regions = extract_regions(model, X, cell_ids=cell_ids)
        
        # 7. Compute Region Stability
        region_stabilities_map = compute_region_stability(model, ensemble, X, point_stability)
        
        all_results[mode] = {
            "region_mass": [r["mass"] for r in regions],
            "region_stability": [region_stabilities_map[r["region_id"]] for r in regions],
            "region_purity": [r.get("purity", 0.0) for r in regions],
            "point_stability": point_stability.tolist(),
            "boundary_stability": boundary_stab.tolist(),
            "boundary_indices": boundary_idx.tolist(),
            "X": X.tolist()
        }
        
    # Save combined results
    json_results = {m: {k: v for k, v in all_results[m].items() if k != "X"} for m in modes}
    with open("results/logs/probe_results.json", "w") as f:
        json.dump(json_results, f)
        
    print("\nResults saved to results/logs/probe_results.json")
    
    # 8. Visualizations
    
    # Point Stability Heatmaps (Side-by-Side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, mode in enumerate(modes):
        ps = np.array(all_results[mode]["point_stability"])
        sc = axes[i].scatter(X[:, 0], X[:, 1], c=ps, cmap='viridis', s=1)
        fig.colorbar(sc, ax=axes[i], label='Stability')
        axes[i].set_title(f'Global Point Stability ({mode})')
    plt.tight_layout()
    plt.savefig('results/figures/stability_heatmap_comp.png')
    plt.close()
    
    # Boundary Stability Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, mode in enumerate(modes):
        b_idx = np.array(all_results[mode]["boundary_indices"])
        b_stab = np.array(all_results[mode]["boundary_stability"])
        if len(b_idx) > 0:
            sc = axes[i].scatter(X[b_idx, 0], X[b_idx, 1], c=b_stab, cmap='magma', s=2)
            fig.colorbar(sc, ax=axes[i], label='Stability')
        axes[i].set_title(f'Boundary Stability ({mode})\nthresh={boundary_threshold}')
    plt.tight_layout()
    plt.savefig('results/figures/boundary_stability_comp.png')
    plt.close()

    # Region Stability Histograms
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, mode in enumerate(modes):
        axes[i].hist(all_results[mode]["region_stability"], bins=20, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Region Stability Dist ({mode})')
        axes[i].set_xlabel('Stability')
        axes[i].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('results/figures/region_stability_hist_comp.png')
    plt.close()
    
    # Mass vs Stability Scatter
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, mode in enumerate(modes):
        axes[i].scatter(all_results[mode]["region_mass"], all_results[mode]["region_stability"], alpha=0.5)
        axes[i].set_xscale('log')
        axes[i].set_title(f'Mass vs Stability ({mode})')
        axes[i].set_xlabel('Mass')
        axes[i].set_ylabel('Stability')
        axes[i].grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('results/figures/mass_vs_stability_comp.png')
    plt.close()

    # Purity vs Stability Scatter (Validation Check)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, mode in enumerate(modes):
        axes[i].scatter(all_results[mode]["region_stability"], all_results[mode]["region_purity"], alpha=0.5, color='orange')
        axes[i].set_title(f'Stability vs Purity ({mode})')
        axes[i].set_xlabel('Stability')
        axes[i].set_ylabel('Purity (Dominant Cell Frac)')
        axes[i].grid(True, ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('results/figures/purity_vs_stability_comp.png')
    plt.close()
    
    print("Comparison figures saved to results/figures/")

if __name__ == "__main__":
    run_experiment()
