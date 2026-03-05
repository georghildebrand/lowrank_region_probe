import torch
import numpy as np
from collections import defaultdict

def compute_point_stability(base_model, ensemble, X):
    """
    Computes stability(x) = average fraction of gates that remain unchanged 
    across the ensemble (Hamming similarity).
    """
    base_pattern = base_model.gate_pattern(X) # [N, hidden]
    
    hamming_similarities = torch.zeros(X.shape[0], device=X.device)
    
    for model in ensemble:
        pattern = model.gate_pattern(X)
        # Fraction of gates that are identical for each point
        matches = (pattern == base_pattern).float().mean(dim=1)
        hamming_similarities += matches
        
    stability = hamming_similarities / len(ensemble)
    return stability

def compute_region_stability(base_model, ensemble, X, point_stability=None):
    if point_stability is None:
        point_stability = compute_point_stability(base_model, ensemble, X)
        
    base_patterns = base_model.gate_pattern(X)
    region_data = defaultdict(list)
    
    for i in range(X.shape[0]):
        pattern_tuple = tuple(base_patterns[i].tolist())
        region_data[pattern_tuple].append(point_stability[i].item())
        
    region_stabilities = {}
    for pattern_tuple, stabilities in region_data.items():
        region_stabilities[pattern_tuple] = np.mean(stabilities)
        
    return region_stabilities

def compute_boundary_sensitivity(point_stability):
    return 1.0 - point_stability

def identify_boundary_points(model, X, threshold=0.1):
    """
    Selects points where distance to nearest ACTIVE hyperplane < threshold.
    Active hyperplanes are those associated with neurons that change sign 
    across the dataset X.
    """
    with torch.no_grad():
        W = model.fc1.weight # [hidden, input]
        b = model.fc1.bias   # [hidden]
        
        # 1. Identify Active Hyperplanes
        # Pre-activations for all points: [N, hidden]
        z_all = X @ W.T + b
        gates = (z_all > 0)
        
        # A neuron is active if it is not always 0 AND not always 1
        is_active = (gates.any(dim=0) & (~gates.all(dim=0)))
        active_indices = torch.where(is_active)[0]
        
        if len(active_indices) == 0:
            return torch.tensor([], dtype=torch.long), torch.zeros(X.shape[0])
            
        W_active = W[active_indices]
        b_active = b[active_indices]
        z_active = z_all[:, active_indices]
        
        # 2. Geometric distances to active hyperplanes
        norms = torch.norm(W_active, dim=1) # [num_active]
        distances = torch.abs(z_active) / (norms + 1e-12) # [N, num_active]
        
        # distance_i = min_j distance_{i,j}
        min_distances, _ = torch.min(distances, dim=1) # [N]
        
        boundary_indices = torch.where(min_distances < threshold)[0]
        return boundary_indices, min_distances

def compute_boundary_stability(base_model, ensemble, X, threshold=0.1):
    """
    Computes stability specifically for points near boundaries.
    """
    boundary_indices, _ = identify_boundary_points(base_model, X, threshold)
    
    if len(boundary_indices) == 0:
        return torch.tensor([]), boundary_indices
        
    X_boundary = X[boundary_indices]
    stability_boundary = compute_point_stability(base_model, ensemble, X_boundary)
    
    return stability_boundary, boundary_indices
