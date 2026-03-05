import numpy as np

def score_structural_cells(cell_summaries):
    """
    Computes structural cell score:
    Score = mean_exact_stability * log(1 + mass) * (1 - boundary_fraction)
    """
    scored_cells = []
    for s in cell_summaries:
        score = s["exact_mean"] * np.log1p(s["mass"]) * (1.0 - s["boundary_fraction"])
        
        c = s.copy()
        c["structural_score"] = score
        scored_cells.append(c)
        
    # Sort by score descending
    scored_cells = sorted(scored_cells, key=lambda x: x["structural_score"], reverse=True)
    return scored_cells

def select_top_cells(scored_cells, top_k=20, min_mass=10):
    """
    Selects top candidates with thresholding.
    """
    selection = [c for c in scored_cells if c["mass"] >= min_mass]
    return selection[:top_k]
