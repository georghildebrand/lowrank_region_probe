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
    Selects points where distance to nearest hyperplane < threshold.
    distance_j = |w_j·x + b_j| / ||w_j||
    """
    with torch.no_grad():
        W = model.fc1.weight # [hidden, input]
        b = model.fc1.bias   # [hidden]
        
        # Pre-activations: z = X @ W.T + b  -> [N, hidden]
        z = X @ W.T + b
        
        # Normalization factor: ||w_j||
        norms = torch.norm(W, dim=1) # [hidden]
        
        # Geometric distances to all hyperplanes
        distances = torch.abs(z) / (norms + 1e-12) # [N, hidden]
        
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
