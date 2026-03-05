import torch
import numpy as np
from collections import defaultdict

def compute_point_stability(base_model, ensemble, X):
    """
    Computes stability(x) = fraction of ensemble models with same gate pattern as base.
    """
    base_pattern = base_model.gate_pattern(X) # [N, hidden]
    
    match_counts = torch.zeros(X.shape[0])
    
    for model in ensemble:
        pattern = model.gate_pattern(X)
        # Check if entire pattern matches (all gates)
        matches = torch.all(pattern == base_pattern, dim=1)
        match_counts += matches.float()
        
    stability = match_counts / len(ensemble)
    return stability

def compute_region_stability(base_model, ensemble, X, point_stability=None):
    """
    Computes mean stability per region.
    Regions are defined by base gate patterns.
    """
    if point_stability is None:
        point_stability = compute_point_stability(base_model, ensemble, X)
        
    base_patterns = base_model.gate_pattern(X)
    
    # Group points by gate pattern
    # Convert bool tensor to tuple of bits for hashing
    region_data = defaultdict(list)
    
    for i in range(X.shape[0]):
        pattern_tuple = tuple(base_patterns[i].tolist())
        region_data[pattern_tuple].append(point_stability[i].item())
        
    region_stabilities = {}
    for pattern_tuple, stabilities in region_data.items():
        region_stabilities[pattern_tuple] = np.mean(stabilities)
        
    return region_stabilities

def compute_boundary_sensitivity(point_stability):
    """
    boundary_sensitivity(x) = 1 - stability(x)
    """
    return 1.0 - point_stability
