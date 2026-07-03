"""
Experiment 9: multi-layer LoRA prediction.

All 3 layers fine-tuned simultaneously. Per-layer attribution via ablation:
delta_logit_l = |forward_with_delta(base, X_eval, l, deltas[l]) - base(X_eval)|.
Tests whether layer-2 fragility predicts its attribution once the adapter converges.
"""
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from experiments.common import train_model
from models.deep_mlp import DeepMLP
from models.lora import forward_with_delta, forward_with_all_deltas, lora_finetune_multilayer
from datasets.image_binary import load_binary_dataset
from probes.layer_stability import evaluate_layer_stability
from analysis.conditioning import partial_spearman

SEEDS = [0, 1, 2]
LAYERS = [0, 1, 2]
PROBE_RANK = 1
PROBE_SCALE = 0.1
LORA_RANK = 1
LORA_STEPS = 1000
LORA_LR = 1e-2
HIDDEN_DIMS = (256, 128, 64)
N_TRAIN = 20000
N_EVAL = 10000
ENSEMBLE = 64

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    SEEDS = [0]
    N_TRAIN, N_EVAL, ENSEMBLE, LORA_STEPS = 2000, 1000, 8, 100


def main():
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    all_rows = []

    for seed in SEEDS:
        X_tr_a, y_tr_a, X_ev_a, y_ev_a = load_binary_dataset(
            "mnist_even_odd", n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)
        X_tr_b, y_tr_b, X_ev_b, y_ev_b = load_binary_dataset(
            "mnist_lt5", n_train=N_TRAIN, n_eval=N_EVAL, seed=seed)

        assert torch.equal(X_tr_a, X_tr_b), "Train X alignment violated"
        assert torch.equal(X_ev_a, X_ev_b), "Eval X alignment violated"

        X_tr, X_eval = X_tr_a, X_ev_a
        y_lt5_tr = y_tr_b

        torch.manual_seed(seed)
        base = DeepMLP(784, HIDDEN_DIMS)
        base, base_acc = train_model(base, X_tr_a, y_tr_a, max_epochs=60, target_acc=0.99)
        print(f"[seed={seed}] base acc={base_acc:.3f}")

        # Fine-tune all layers simultaneously
        deltas, ft_acc = lora_finetune_multilayer(base, layers=LAYERS, X=X_tr,
                                                   y=y_lt5_tr, rank=LORA_RANK,
                                                   steps=LORA_STEPS, lr=LORA_LR,
                                                   seed=seed)
        print(f"  multi-layer ft_acc={ft_acc:.3f}")

        with torch.no_grad():
            logit_before = base(X_eval).squeeze(1).numpy()

        for layer in LAYERS:
            # Per-layer attribution: ablation — only this layer's delta active
            with torch.no_grad():
                logit_ablation = forward_with_delta(
                    base, X_eval, layer, deltas[layer]).squeeze(1).numpy()
            change = np.abs(logit_ablation - logit_before)

            # Probe: random rank-1 and full-rank fragility
            probe = evaluate_layer_stability(base, X_eval, layer=layer,
                                             rank=PROBE_RANK, scale=PROBE_SCALE,
                                             ensemble_size=ENSEMBLE)
            frag_low = 1.0 - probe["hamming_low"]
            frag_full = 1.0 - probe["hamming_full"]
            dist = probe["dist"]

            rho_low = float(spearmanr(frag_low, change)[0])
            rho_full = float(spearmanr(frag_full, change)[0])
            rho_dist = float(spearmanr(-dist, change)[0])
            rho_low_given_dist = float(partial_spearman(frag_low, change, dist))
            rho_full_given_dist = float(partial_spearman(frag_full, change, dist))

            row = {
                "seed": seed, "layer": layer,
                "rho_low": rho_low, "rho_full": rho_full, "rho_dist": rho_dist,
                "rho_low_given_dist": rho_low_given_dist,
                "rho_full_given_dist": rho_full_given_dist,
                "ft_acc": float(ft_acc), "base_acc": float(base_acc),
                "delta_fro": float(deltas[layer].norm()),
                "mean_change": float(change.mean()),
            }
            all_rows.append(row)
            print(f"  layer={layer}  partial_r1={rho_low_given_dist:+.3f}"
                  f"  partial_full={rho_full_given_dist:+.3f}"
                  f"  rho_dist={rho_dist:+.3f}"
                  f"  mean_change={change.mean():.4f}")

    config = {
        "seeds": SEEDS, "layers": LAYERS, "probe_rank": PROBE_RANK,
        "probe_scale": PROBE_SCALE, "lora_rank": LORA_RANK, "lora_steps": LORA_STEPS,
        "lora_lr": LORA_LR, "hidden_dims": list(HIDDEN_DIMS),
        "n_train": N_TRAIN, "n_eval": N_EVAL, "ensemble": ENSEMBLE, "smoke": SMOKE,
    }
    with open("results/logs/multilayer_lora_results.json", "w") as f:
        json.dump({"config": config, "results": all_rows}, f, indent=2)

    # Figure
    metrics = ["rho_low_given_dist", "rho_full_given_dist", "rho_dist"]
    labels = ["partial rho(r1|dist)", "partial rho(full|dist)", "rho(-dist)"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(LAYERS))
    w = 0.22
    offsets = np.linspace(-w, w, 3)
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        means, stds = [], []
        for layer in LAYERS:
            vals = [r[metric] for r in all_rows if r["layer"] == layer]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.bar(x + offsets[i], means, w, yerr=stds, label=label,
               color=color, capsize=4, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"layer {l}" for l in LAYERS])
    ax.set_ylabel("Spearman rho")
    ax.set_title("Multi-layer LoRA: per-layer ablation attribution vs probe fragility")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/multilayer_lora.png", dpi=150)
    plt.close(fig)

    print("\n==== MULTILAYER LORA SUMMARY (mean over seeds) ====")
    print(f"{'layer':>5}  {'partial r1|dist':>15}  {'partial full|dist':>17}"
          f"  {'rho(-dist)':>10}  {'ft_acc':>6}  {'mean_change':>11}")
    for layer in LAYERS:
        rows = [r for r in all_rows if r["layer"] == layer]
        def m(k): return np.mean([r[k] for r in rows])
        print(f"{layer:>5}  {m('rho_low_given_dist'):>+15.3f}"
              f"  {m('rho_full_given_dist'):>+17.3f}"
              f"  {m('rho_dist'):>+10.3f}"
              f"  {m('ft_acc'):>6.3f}"
              f"  {m('mean_change'):>11.4f}")
    print("\nSaved results/logs/multilayer_lora_results.json + figure.")


if __name__ == "__main__":
    main()
