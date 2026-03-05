import torch
import copy

def generate_perturbation_ensemble(base_model, mode="lowrank", rank=2, scale=0.05, ensemble_size=64):
    """
    Generates an ensemble of models with perturbed first-layer weights.
    Adjusts biases to preserve the distance of the hyperplanes to the origin.
    mode: "lowrank" (B@A) or "fullrank" (Gaussian noise).
    """
    ensemble = []
    
    hidden_dim, input_dim = base_model.fc1.weight.shape
    W0 = base_model.fc1.weight.data.clone()
    b0 = base_model.fc1.bias.data.clone()
    
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
            raise ValueError("mode must be 'lowrank' or 'fullrank'")
            
        # 2. Scale perturbation to exact Frobenius norm
        perturbation = (perturbation / torch.norm(perturbation)) * scale * torch.norm(W0)
        
        # 3. Apply weight perturbation
        W_new = W0 + perturbation
        
        # 4. Correct Bias to prevent "Origin Collapse"
        # Distance to origin is d = -b / ||w||. To keep d constant: b_new = b0 * (||W_new|| / ||W0||)
        norm_W0 = torch.norm(W0, dim=1) + 1e-12
        norm_W_new = torch.norm(W_new, dim=1)
        b_new = b0 * (norm_W_new / norm_W0)
        
        with torch.no_grad():
            perturbed_model.fc1.weight.copy_(W_new)
            perturbed_model.fc1.bias.copy_(b_new)
            
        ensemble.append(perturbed_model)
        
    return ensemble
