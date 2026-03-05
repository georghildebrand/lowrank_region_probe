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
from probes.lowrank_ensemble import generate_lowrank_ensemble
from probes.stability_metrics import compute_point_stability, compute_region_stability
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
    INPUT_DIM = 10 # Default for non-2D analysis
    HIDDEN_WIDTH = 64
    RANK = 2
    ENSEMBLE_SIZE = 64
    PERTURBATION_SCALE = 0.05
    DATASET_SIZE = 10000
    TRAINING_STEPS = 3000
    
    # Create directories
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # 1. Generate Dataset (Using synthetic polytopes for stability check)
    # We'll use 2D for visualization but the logic supports higher dims
    X, y, cell_ids = generate_synthetic_polytopes(n_samples=DATASET_SIZE, dim=2)
    
    # 2. Train Base Model
    # Note: Using dim=2 here to allow Heatmap visualization
    model = MLP(input_dim=2, hidden_dim=HIDDEN_WIDTH)
    print("Training base model...")
    train_model(model, X, y, steps=TRAINING_STEPS)
    
    # 3. Compute Base Gate Patterns
    base_patterns = model.gate_pattern(X)
    
    # 4. Generate Low-Rank Ensemble
    print("Generating low-rank ensemble...")
    ensemble = generate_lowrank_ensemble(model, rank=RANK, scale=PERTURBATION_SCALE, ensemble_size=ENSEMBLE_SIZE)
    
    # 5. Compute Stability Metrics
    print("Computing stability metrics...")
    point_stability = compute_point_stability(model, ensemble, X)
    
    # 6. Extract Regions and Mass
    regions = extract_regions(model, X)
    
    # 7. Compute Region Stability
    region_stabilities_map = compute_region_stability(model, ensemble, X, point_stability)
    
    # Prepare results for JSON
    results = {
        "region_mass": [r["mass"] for r in regions],
        "region_stability": [region_stabilities_map[r["region_id"]] for r in regions],
        "point_stability_distribution": point_stability.tolist()
    }
    
    with open("results/logs/probe_results.json", "w") as f:
        json.dump(results, f)
        
    print("Results saved to results/logs/probe_results.json")
    
    # 8. Visualizations
    
    # Stability Heatmap
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X[:, 0], X[:, 1], c=point_stability.numpy(), cmap='viridis', s=1)
    plt.colorbar(sc, label='Stability')
    plt.title('Point Stability Heatmap')
    plt.savefig('results/figures/stability_heatmap.png')
    plt.close()
    
    # Region Stability Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(results["region_stability"], bins=20, color='skyblue', edgecolor='black')
    plt.title('Region Stability Distribution')
    plt.xlabel('Stability')
    plt.ylabel('Count')
    plt.savefig('results/figures/region_stability_hist.png')
    plt.close()
    
    # Mass vs Stability
    plt.figure(figsize=(8, 6))
    plt.scatter(results["region_mass"], results["region_stability"], alpha=0.5)
    plt.xscale('log')
    plt.title('Region Mass vs Stability')
    plt.xlabel('Mass (Number of Points)')
    plt.ylabel('Stability')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('results/figures/mass_vs_stability.png')
    plt.close()
    
    print("Figures saved to results/figures/")

if __name__ == "__main__":
    run_experiment()
