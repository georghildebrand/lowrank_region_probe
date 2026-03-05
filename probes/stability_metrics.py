import torch
import numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

def compute_point_hamming_stability(base_model, ensemble, X):
    """
    Current Hamming similarity (fraction of gates identical).
    """
    base_pattern = base_model.gate_pattern(X)
    similarities = torch.zeros(X.shape[0], device=X.device)
    
    for model in ensemble:
        pattern = model.gate_pattern(X)
        matches = (pattern == base_pattern).float().mean(dim=1)
        similarities += matches
        
    return similarities / len(ensemble)

def compute_point_exact_stability(base_model, ensemble, X):
    """
    Exact pattern match stability (Boolean matches).
    """
    base_pattern = base_model.gate_pattern(X)
    matches_count = torch.zeros(X.shape[0], device=X.device)
    
    for model in ensemble:
        pattern = model.gate_pattern(X)
        matches = torch.all(pattern == base_pattern, dim=1).float()
        matches_count += matches
        
    return matches_count / len(ensemble)

def identify_boundary_points(model, X, threshold=0.01, boundary_mode="all", k=64):
    """
    Identifies points near polytope boundaries.
    Modes:
    - 'all': min distance to any hyperplane.
    - 'active': min distance to neurons that flip across the whole dataset X.
    - 'locally_active': min distance to neurons that flip within k-NN neighborhood.
    """
    with torch.no_grad():
        W = model.fc1.weight
        b = model.fc1.bias
        z_all = X @ W.T + b # [N, hidden]
        
        if boundary_mode == "all":
            active_indices = torch.arange(W.shape[0])
        elif boundary_mode == "active":
            gates = (z_all > 0)
            is_active = (gates.any(dim=0) & (~gates.all(dim=0)))
            active_indices = torch.where(is_active)[0]
        elif boundary_mode == "locally_active":
            # Very heavy: check sign flips in kNN neighborhoods
            X_np = X.cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=k).fit(X_np)
            indices = nbrs.kneighbors(X_np, return_distance=False) # [N, k]
            
            gates = (z_all > 0).cpu().numpy() # [N, hidden]
            
            # For each point, find which neurons flip in its neighborhood
            # local_gates: [N, k, hidden]
            local_flips = np.zeros((X.shape[0], W.shape[0]), dtype=bool)
            for i in range(X.shape[0]):
                neighbor_gates = gates[indices[i]] # [k, hidden]
                # Flips if neighbor gates are not all identical to point gate
                # OR just if any pair in neighbors differ
                # Correction: "hyperplanes that change sign within neighborhood"
                local_flips[i] = neighbor_gates.any(axis=0) & (~neighbor_gates.all(axis=0))
            
            # Now compute distance only to those locally active hyperplanes per point
            norms = torch.norm(W, dim=1) # [hidden]
            distances_raw = torch.abs(z_all) / (norms + 1e-12) # [N, hidden]
            
            local_flips_torch = torch.from_numpy(local_flips).to(X.device)
            # Mask inactive entries with infinity
            distances_masked = torch.where(local_flips_torch, distances_raw, torch.tensor(float('inf')).to(X.device))
            
            min_distances, _ = torch.min(distances_masked, dim=1)
            # Handle points with no locally active hyperplanes (stable deep interiors)
            min_distances[torch.isinf(min_distances)] = threshold + 1.0 # Far away
            
            boundary_indices = torch.where(min_distances < threshold)[0]
            return boundary_indices, min_distances
        else:
            raise ValueError(f"Unknown boundary_mode: {boundary_mode}")

        if len(active_indices) == 0:
            return torch.tensor([], dtype=torch.long), torch.zeros(X.shape[0])
            
        W_act = W[active_indices]
        z_act = z_all[:, active_indices]
        norms = torch.norm(W_act, dim=1)
        distances = torch.abs(z_act) / (norms + 1e-12)
        min_distances, _ = torch.min(distances, dim=1)
        
        boundary_indices = torch.where(min_distances < threshold)[0]
        return boundary_indices, min_distances

def compute_boundary_stability(base_model, ensemble, X, threshold=0.01, boundary_mode="all"):
    """Legacy wrapper for point stability at boundaries."""
    indices, _ = identify_boundary_points(base_model, X, threshold=threshold, boundary_mode=boundary_mode)
    if len(indices) == 0:
        return torch.tensor([]), indices
    stab = compute_point_exact_stability(base_model, ensemble, X[indices])
    return stab, indices
