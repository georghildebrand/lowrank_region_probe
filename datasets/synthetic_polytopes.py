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


def generate_soft_checkerboard(n_samples=10000, softness=0.0, seed=None):
    """
    2D checkerboard with tunable boundary softness.

    softness=0 reproduces the hard quadrant checkerboard exactly.
    softness>0 draws labels y ~ Bernoulli(sigmoid(signed_margin / softness)),
    signed_margin = min(|x1|,|x2|) * (2*y_hard - 1): a coin flip on the
    quadrant boundary, (near-)deterministic deep inside a quadrant.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n_samples, 2))

    cell_id = np.zeros(n_samples, dtype=int)
    cell_id[(X[:, 0] >= 0) & (X[:, 1] >= 0)] = 0
    cell_id[(X[:, 0] < 0) & (X[:, 1] >= 0)] = 1
    cell_id[(X[:, 0] < 0) & (X[:, 1] < 0)] = 2
    cell_id[(X[:, 0] >= 0) & (X[:, 1] < 0)] = 3

    y_hard = (cell_id % 2).astype(float)
    if softness <= 0:
        y = y_hard
    else:
        margin = np.minimum(np.abs(X[:, 0]), np.abs(X[:, 1]))
        signed_margin = margin * (2 * y_hard - 1)
        p_one = 1.0 / (1.0 + np.exp(-signed_margin / softness))
        y = (rng.uniform(size=n_samples) < p_one).astype(float)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        torch.tensor(cell_id, dtype=torch.long),
    )
