import torch
import copy
import numpy as np

def generate_perturbation_ensemble(base_model, ensemble_size=64, perturbation_config=None):
    """
    Generates an ensemble of models with perturbed first-layer weights.
    Supports low-rank/full-rank and different bias perturbation families.
    """
    if perturbation_config is None:
        perturbation_config = {
            "mode": "lowrank",
            "rank": 2,
            "scale": 0.05,
            "family": "centroid_preserving",
            "data_centroid": None
        }
    
    mode = perturbation_config.get("mode", "lowrank")
    rank = perturbation_config.get("rank", 2)
    scale = perturbation_config.get("scale", 0.05)
    family = perturbation_config.get("family", "centroid_preserving")
    data_centroid = perturbation_config.get("data_centroid", None)
    
    ensemble = []
    hidden_dim, input_dim = base_model.fc1.weight.shape
    W0 = base_model.fc1.weight.data.clone()
    b0 = base_model.fc1.bias.data.clone()
    
    if family == "centroid_preserving" and data_centroid is None:
        # If not provided, assume zero if needed, but better to raise or calculate
        # For robustness, we will default to zero but log a warning if possible
        data_centroid = torch.zeros(input_dim)

    for _ in range(ensemble_size):
        perturbed_model = copy.deepcopy(base_model)
        
        # 1. Generate Perturbation Direction
        if mode == "lowrank":
            A = torch.randn(rank, input_dim)
            B = torch.randn(hidden_dim, rank)
            perturbation = B @ A
        elif mode == "fullrank":
            perturbation = torch.randn(hidden_dim, input_dim)
        else:
            raise ValueError(f"mode must be 'lowrank' or 'fullrank', got {mode}")
            
        # 2. Scale perturbation to exact Frobenius norm
        pert_norm = torch.norm(perturbation)
        if pert_norm > 0:
            perturbation = (perturbation / pert_norm) * scale * torch.norm(W0)
        
        # 3. Apply weight perturbation
        W_new = W0 + perturbation
        
        # 4. Handle Bias according to Family
        if family == "weights_only":
            b_new = b0
        elif family == "weights_and_bias":
            # Match bias perturbation norm relative to b0
            # If b0 is zero, use a small scale relative to W0 norm
            b_pert = torch.randn(hidden_dim)
            b0_norm = torch.norm(b0)
            if b0_norm > 1e-6:
                b_pert = (b_pert / torch.norm(b_pert)) * scale * b0_norm
            else:
                b_pert = (b_pert / torch.norm(b_pert)) * scale * (torch.norm(W0) / np.sqrt(hidden_dim))
            b_new = b0 + b_pert
        elif family == "centroid_preserving":
            # b_new = b0 + (W0 - W_new) @ centroid
            # This preserves the signed offset w·c + b
            b_new = b0 + torch.mv((W0 - W_new), data_centroid)
        else:
            raise ValueError(f"Unknown family: {family}")
            
        with torch.no_grad():
            perturbed_model.fc1.weight.copy_(W_new)
            perturbed_model.fc1.bias.copy_(b_new)
            
        ensemble.append(perturbed_model)
        
    return ensemble
