import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_cell_adjacency(X, cell_ids, k=16):
    """
    Builds an adjacency graph between cells based on point proximity.
    """
    N = X.shape[0]
    # Filter points that actually belong to a valid cell (cell_id >= 0)
    valid_mask = cell_ids >= 0
    X_valid = X[valid_mask]
    ids_valid = cell_ids[valid_mask]
    map_back = np.where(valid_mask)[0]
    
    if len(X_valid) < 2:
        return []
        
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X_valid))).fit(X_valid)
    # Get indices of neighbors for everyone
    distances, indices = nbrs.kneighbors(X_valid)
    
    edges = set()
    for i in range(len(X_valid)):
        u_cell = ids_valid[i]
        for j in indices[i, 1:]: # Skip self
            v_cell = ids_valid[j]
            if u_cell != v_cell:
                # Store as sorted tuple for un-ordered uniqueness
                edge = tuple(sorted((int(u_cell), int(v_cell))))
                edges.add(edge)
                
    return sorted(list(edges))

def get_cell_degrees(cells, edges):
    """Computes adjacency degree per cell."""
    degrees = {c["cell_index"]: 0 for c in cells}
    for u, v in edges:
        if u in degrees: degrees[u] += 1
        if v in degrees: degrees[v] += 1
    return degrees
