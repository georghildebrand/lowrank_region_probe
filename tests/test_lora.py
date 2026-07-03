import copy
import torch
import pytest
from models.deep_mlp import DeepMLP
from models.lora import forward_with_delta, lora_finetune


def _small_model(seed=0):
    torch.manual_seed(seed)
    return DeepMLP(input_dim=16, hidden_dims=(8, 8, 4))


def test_forward_with_delta_zeros_equals_model():
    """forward_with_delta with zero delta must equal model(x) exactly."""
    model = _small_model()
    x = torch.randn(10, 16)
    for layer in range(len(model.fcs)):
        W, _ = model.layer_weight(layer)
        out_dim, in_dim = W.shape
        delta = torch.zeros(out_dim, in_dim)
        out_delta = forward_with_delta(model, x, layer, delta)
        out_model = model(x)
        assert torch.allclose(out_delta, out_model, atol=1e-6), (
            f"layer {layer}: delta-zero forward differs from model forward"
        )


def test_lora_finetune_improves_accuracy():
    """lora_finetune on a trivially learnable task should raise accuracy.
    Fine-tune the output layer (closest to label) with a clear linear rule
    and more steps to ensure reliable convergence."""
    torch.manual_seed(42)
    model = DeepMLP(input_dim=16, hidden_dims=(8, 8, 4))
    X = torch.randn(400, 16)
    # Linear rule: label = sign(X[:,0])
    y = (X[:, 0] > 0).float().unsqueeze(1)

    # Initial accuracy before fine-tuning (expect ~50% for random model)
    with torch.no_grad():
        init_acc = ((model(X) > 0).float() == y).float().mean().item()

    # Fine-tune last hidden layer (closest to output) with more steps
    last_layer = len(model.fcs) - 1
    delta, ft_acc = lora_finetune(model, layer=last_layer, X=X, y=y,
                                  rank=1, steps=500, lr=5e-2, seed=0)
    assert ft_acc > 0.6, (
        f"fine-tune did not reach >60%: init={init_acc:.3f} final={ft_acc:.3f}"
    )


def test_lora_finetune_does_not_change_base_weights():
    """Base model weights must be unchanged after lora_finetune."""
    model = _small_model()
    state_before = {k: v.clone() for k, v in model.state_dict().items()}

    X = torch.randn(100, 16)
    y = (X[:, 0] > 0).float().unsqueeze(1)
    lora_finetune(model, layer=1, X=X, y=y, rank=1, steps=50, seed=0)

    for k, v_before in state_before.items():
        assert torch.equal(model.state_dict()[k], v_before), (
            f"weight {k} was modified by lora_finetune"
        )


def test_lora_delta_rank():
    """Returned delta must have matrix rank <= requested rank."""
    model = _small_model()
    X = torch.randn(100, 16)
    y = (X[:, 0] > 0).float().unsqueeze(1)
    for rank in [1, 2]:
        delta, _ = lora_finetune(model, layer=0, X=X, y=y,
                                 rank=rank, steps=100, seed=0)
        actual_rank = torch.linalg.matrix_rank(delta).item()
        assert actual_rank <= rank, (
            f"delta rank {actual_rank} > requested rank {rank}"
        )
