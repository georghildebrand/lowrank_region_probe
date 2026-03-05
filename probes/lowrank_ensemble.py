import torch
import copy

def generate_lowrank_ensemble(base_model, rank=2, scale=0.05, ensemble_size=64):
    """
    Generates an ensemble of models with low-rank perturbations to the first layer's weights.
    ΔW = scale * (B @ A)
    """
    ensemble = []
    
    hidden_dim, input_dim = base_model.fc1.weight.shape
    W0 = base_model.fc1.weight.data
    
    for _ in range(ensemble_size):
        # Create a copy of the base model
        perturbed_model = copy.deepcopy(base_model)
        
        # Generate low-rank perturbation
        A = torch.randn(rank, input_dim)
        B = torch.randn(hidden_dim, rank)
        
        # Normalize to ensure consistent perturbation norm
        perturbation = B @ A
        # Using the specified scale to adjust the perturbation relative to some norm
        # or directly as weight magnitude. 
        # Instructions say: "scale * (B @ A)" and "Ensure perturbation norm is approximately consistent."
        perturbation = (perturbation / torch.norm(perturbation)) * scale * torch.norm(W0)
        
        # Apply perturbation
        with torch.no_grad():
            perturbed_model.fc1.weight.add_(perturbation)
            
        ensemble.append(perturbed_model)
        
    return ensemble
