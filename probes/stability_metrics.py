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
