"""
LoRA rank/steps sweep: does fragility-prediction strengthen as the adapter converges?

Base task: mnist_even_odd. LoRA fine-tune onto mnist_lt5 (same X, new labels).
Probe stays rank-1; sweep over LORA_RANKS x LORA_STEPS_LIST. Seed=0 only.
"""
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from experiments.common import train_model
from models.deep_mlp import DeepMLP
from models.lora import forward_with_delta, lora_finetune
from datasets.image_binary import load_binary_dataset
from probes.layer_stability import evaluate_layer_stability
from analysis.conditioning import partial_spearman

SEED = 0
LAYERS = [0, 1, 2]
PROBE_RANK = 1
PROBE_SCALE = 0.1
LORA_RANKS = [1, 4, 8]
LORA_STEPS_LIST = [300, 1000]
LORA_LR = 1e-2
HIDDEN_DIMS = (256, 128, 64)
N_TRAIN = 20000
N_EVAL = 10000
ENSEMBLE = 64

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_TRAIN, N_EVAL, ENSEMBLE = 2000, 1000, 8
    LORA_RANKS, LORA_STEPS_LIST = [1, 4], [50, 100]


def main():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # Load both splits — X must align
    X_tr_a, y_tr_a, X_ev_a, y_ev_a = load_binary_dataset(
        "mnist_even_odd", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)
    X_tr_b, y_tr_b, X_ev_b, y_ev_b = load_binary_dataset(
        "mnist_lt5", n_train=N_TRAIN, n_eval=N_EVAL, seed=SEED)

    assert torch.equal(X_tr_a, X_tr_b), "Train X alignment violated"
    assert torch.equal(X_ev_a, X_ev_b), "Eval X alignment violated"

    X_tr, X_eval = X_tr_a, X_ev_a
    y_lt5_tr = y_tr_b

    # Train base model once
    torch.manual_seed(SEED)
    base = DeepMLP(784, HIDDEN_DIMS)
    base, base_acc = train_model(base, X_tr_a, y_tr_a, max_epochs=60, target_acc=0.99)
    print(f"[seed={SEED}] base model acc={base_acc:.3f}")

    all_rows = []

    for layer in LAYERS:
        # Probe cached once per layer (independent of LoRA rank/steps)
        probe = evaluate_layer_stability(base, X_eval, layer=layer,
                                         rank=PROBE_RANK, scale=PROBE_SCALE,
                                         ensemble_size=ENSEMBLE)
        frag_low = 1.0 - probe["hamming_low"]
        frag_full = 1.0 - probe["hamming_full"]
        dist = probe["dist"]

        with torch.no_grad():
            logit_before = base(X_eval).squeeze(1).numpy()

        for lora_rank in LORA_RANKS:
            for lora_steps in LORA_STEPS_LIST:
                delta, ft_acc = lora_finetune(base, layer, X_tr, y_lt5_tr,
                                              rank=lora_rank, steps=lora_steps,
                                              lr=LORA_LR, seed=SEED)
                with torch.no_grad():
                    logit_after = forward_with_delta(base, X_eval, layer, delta).squeeze(1).numpy()
                change = np.abs(logit_after - logit_before)

                rho_low = spearmanr(frag_low, change)[0]
                rho_full = spearmanr(frag_full, change)[0]
                rho_dist = spearmanr(-dist, change)[0]
                rho_low_given_dist = partial_spearman(frag_low, change, dist)
                rho_full_given_dist = partial_spearman(frag_full, change, dist)

                row = {
                    "layer": layer, "lora_rank": lora_rank, "lora_steps": lora_steps,
                    "rho_low": float(rho_low), "rho_full": float(rho_full),
                    "rho_dist": float(rho_dist),
                    "rho_low_given_dist": float(rho_low_given_dist),
                    "rho_full_given_dist": float(rho_full_given_dist),
                    "ft_acc": float(ft_acc), "base_acc": float(base_acc),
                    "delta_fro": float(delta.norm()),
                }
                all_rows.append(row)
                print(f"  layer={layer} rank={lora_rank} steps={lora_steps}"
                      f"  partial_r1={rho_low_given_dist:+.3f}"
                      f"  partial_full={rho_full_given_dist:+.3f}"
                      f"  ft_acc={ft_acc:.3f}"
                      f"  |delta|={delta.norm():.3f}")

    # Save results
    config = {
        "seed": SEED, "layers": LAYERS, "probe_rank": PROBE_RANK,
        "probe_scale": PROBE_SCALE, "lora_ranks": LORA_RANKS,
        "lora_steps_list": LORA_STEPS_LIST, "lora_lr": LORA_LR,
        "hidden_dims": list(HIDDEN_DIMS), "n_train": N_TRAIN,
        "n_eval": N_EVAL, "ensemble": ENSEMBLE, "smoke": SMOKE,
    }
    with open("results/logs/lora_sweep_results.json", "w") as f:
        json.dump({"config": config, "results": all_rows}, f, indent=2)

    # Figure: 2-panel
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    layer_colors = {0: "#2196F3", 1: "#FF9800", 2: "#4CAF50"}
    for layer in LAYERS:
        for steps, ls in zip(LORA_STEPS_LIST, ["-", "--"]):
            rows = [r for r in all_rows if r["layer"] == layer and r["lora_steps"] == steps]
            x = [r["lora_rank"] for r in rows]
            axes[0].plot(x, [r["ft_acc"] for r in rows],
                         color=layer_colors[layer], ls=ls,
                         marker="o", label=f"L{layer} s={steps}")
            axes[1].plot(x, [r["rho_low_given_dist"] for r in rows],
                         color=layer_colors[layer], ls=ls,
                         marker="o", label=f"L{layer} s={steps}")
    for ax, title in zip(axes, ["ft_acc vs lora_rank", "partial rho(frag_r1|dist) vs lora_rank"]):
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("lora_rank")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("LoRA sweep: probe fragility prediction vs adapter rank/steps")
    fig.tight_layout()
    fig.savefig("results/figures/lora_sweep.png", dpi=150)
    plt.close(fig)

    # Summary table
    print("\n==== LORA SWEEP SUMMARY ====")
    print(f"{'layer':>5}  {'rank':>5}  {'steps':>6}  {'ft_acc':>7}  "
          f"{'partial_r1|dist':>16}  {'partial_full|dist':>17}  {'|delta|':>8}")
    for row in sorted(all_rows, key=lambda r: (r["layer"], r["lora_rank"], r["lora_steps"])):
        print(f"{row['layer']:>5}  {row['lora_rank']:>5}  {row['lora_steps']:>6}"
              f"  {row['ft_acc']:>7.3f}  {row['rho_low_given_dist']:>+16.3f}"
              f"  {row['rho_full_given_dist']:>+17.3f}  {row['delta_fro']:>8.3f}")
    print("\nSaved results/logs/lora_sweep_results.json + figure.")


if __name__ == "__main__":
    main()
