import numpy as np
import torch

def generate_synthetic_polytopes(n_samples=10000, dim=2):
    """
    Generates a dataset consisting of convex polytope regions in 2D.
    For simplicity, we use 4 quadrants.
    """
    X = np.random.uniform(-1, 1, (n_samples, dim))
    
    cell_id = np.zeros(n_samples, dtype=int)
    # Assign cell IDs based on quadrants
    # Cell 0: x > 0, y > 0
    # Cell 1: x < 0, y > 0
    # Cell 2: x < 0, y < 0
    # Cell 3: x > 0, y < 0
    
    cell_id[(X[:, 0] >= 0) & (X[:, 1] >= 0)] = 0
    cell_id[(X[:, 0] < 0) & (X[:, 1] >= 0)] = 1
    cell_id[(X[:, 0] < 0) & (X[:, 1] < 0)] = 2
    cell_id[(X[:, 0] >= 0) & (X[:, 1] < 0)] = 3
    
    # Define labels (simple checkerboard)
    y = (cell_id % 2).astype(float)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    cell_id_tensor = torch.tensor(cell_id, dtype=torch.long)
    
    return X_tensor, y_tensor, cell_id_tensor
