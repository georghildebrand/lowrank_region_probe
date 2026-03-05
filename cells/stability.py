import torch
import numpy as np

def compute_cell_stability_summaries(cells, point_exact_stab, point_ham_stab, boundary_indices=None):
    """
    Aggregates point-level stability into cell-level statistics.
    """
    Summaries = []
    
    # Use a set for fast lookup of boundary points
    bound_set = set(boundary_indices.tolist()) if boundary_indices is not None else set()
    
    for cell in cells:
        idx = cell["member_indices"]
        
        c_exact = point_exact_stab[idx]
        c_ham = point_ham_stab[idx]
        
        # Intersection with boundary indices
        cell_members_set = set(idx)
        boundary_idx_in_cell = list(cell_members_set.intersection(bound_set))
        
        summary = {
            "cell_index": cell["cell_index"],
            "exact_mean": c_exact.mean().item(),
            "exact_std": c_exact.std().item() if len(c_exact) > 1 else 0.0,
            "ham_mean": c_ham.mean().item(),
            "ham_std": c_ham.std().item() if len(c_ham) > 1 else 0.0,
            "mass": cell["mass"],
            "boundary_fraction": len(boundary_idx_in_cell) / cell["mass"]
        }
        
        if len(boundary_idx_in_cell) > 0:
            summary["exact_stability_boundaryband"] = point_exact_stab[boundary_idx_in_cell].mean().item()
        else:
            summary["exact_stability_boundaryband"] = 1.0 # Or NaN? manual implies mean over band
            
        Summaries.append(summary)
        
    return Summaries
