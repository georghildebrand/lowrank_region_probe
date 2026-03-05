import numpy as np
import torch

def generate_circle_dataset(n_samples=10000, radius=0.7):
    """
    Generate points: u in [-1,1]^2, label = (||u|| < r)
    """
    X = np.random.uniform(-1, 1, (n_samples, 2))
    dist = np.linalg.norm(X, axis=1)
    y = (dist < radius).astype(float)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    return X_tensor, y_tensor
