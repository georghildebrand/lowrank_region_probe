"""
Directional probe experiment: does probing in the learned LoRA direction
predict fine-tune behavior change better than random rank-1 or full-rank?

Base task: mnist_even_odd; fine-tune task: mnist_lt5 (same X, new labels).
Directional probe uses ±(top singular direction of learned delta) as the
perturbation, norm-matched to random probes.
"""
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from experiments.common import train_model
from models.deep_mlp import DeepMLP
from models.lora import forward_with_delta, lora_finetune, dominant_direction
from datasets.image_binary import load_binary_dataset
from probes.layer_stability import evaluate_layer_stability, evaluate_layer_stability_directional
from analysis.conditioning import partial_spearman

SEEDS = [0, 1, 2]
LAYERS = [0, 1, 2]
PROBE_RANK = 1
PROBE_SCALE = 0.1
LORA_RANK = 1
LORA_STEPS = 300
LORA_LR = 1e-2
HIDDEN_DIMS = (256, 128, 64)
N_TRAIN = 20000
N_EVAL = 10000
ENSEMBLE = 64
ENSEMBLE_DIR = 2

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    SEEDS = [0]
    N_TRAIN, N_EVAL, ENSEMBLE, LORA_STEPS = 2000, 1000, 8, 50


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
        print(f"[seed={seed}] base model acc={base_acc:.3f}")

        for layer in LAYERS:
            # Random probes (rank-1 and full-rank)
            probe = evaluate_layer_stability(base, X_eval, layer=layer,
                                             rank=PROBE_RANK, scale=PROBE_SCALE,
                                             ensemble_size=ENSEMBLE)
            frag_low = 1.0 - probe["hamming_low"]
            frag_full = 1.0 - probe["hamming_full"]
            dist = probe["dist"]

            with torch.no_grad():
                logit_before = base(X_eval).squeeze(1).numpy()

            # LoRA fine-tune to get the learned delta
            delta, ft_acc = lora_finetune(base, layer, X_tr, y_lt5_tr,
                                          rank=LORA_RANK, steps=LORA_STEPS,
                                          lr=LORA_LR, seed=seed)

            with torch.no_grad():
                logit_after = forward_with_delta(base, X_eval, layer, delta).squeeze(1).numpy()
            change = np.abs(logit_after - logit_before)
            delta_fro = float(delta.norm())

            # Directional probe in the learned LoRA direction
            if delta_fro < 1e-9:
                print(f"  WARNING: delta near zero at seed={seed} layer={layer}, skipping directional probe")
                rho_directional = float("nan")
                partial_rho_directional = float("nan")
            else:
                direction = dominant_direction(delta)
                dprobe = evaluate_layer_stability_directional(
                    base, X_eval, layer, direction, scale=PROBE_SCALE,
                    ensemble_size=ENSEMBLE_DIR)
                frag_dir = 1.0 - dprobe["hamming_dir"]
                rho_directional = float(spearmanr(frag_dir, change)[0])
                partial_rho_directional = float(partial_spearman(frag_dir, change, dist))

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
                "rho_directional": rho_directional,
                "partial_rho_directional": partial_rho_directional,
                "ft_acc": float(ft_acc), "base_acc": float(base_acc),
                "delta_fro": delta_fro,
            }
            all_rows.append(row)
            print(f"  layer={layer}"
                  f"  partial_r1={rho_low_given_dist:+.3f}"
                  f"  partial_full={rho_full_given_dist:+.3f}"
                  f"  partial_dir={partial_rho_directional:+.3f}"
                  f"  ft_acc={ft_acc:.3f}")

    config = {
        "seeds": SEEDS, "layers": LAYERS, "probe_rank": PROBE_RANK,
        "probe_scale": PROBE_SCALE, "lora_rank": LORA_RANK,
        "lora_steps": LORA_STEPS, "lora_lr": LORA_LR,
        "hidden_dims": list(HIDDEN_DIMS), "n_train": N_TRAIN,
        "n_eval": N_EVAL, "ensemble": ENSEMBLE, "ensemble_dir": ENSEMBLE_DIR,
        "smoke": SMOKE,
    }
    with open("results/logs/directional_probe_results.json", "w") as f:
        json.dump({"config": config, "results": all_rows}, f, indent=2)

    # Figure: grouped bar per layer, 7 correlations
    metrics = ["rho_low", "rho_full", "rho_dist",
               "rho_low_given_dist", "rho_full_given_dist",
               "rho_directional", "partial_rho_directional"]
    labels = ["rho(frag_r1)", "rho(frag_full)", "rho(-dist)",
              "partial r1|dist", "partial full|dist",
              "rho(frag_dir)", "partial dir|dist"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#795548",
              "#F44336", "#E91E63"]
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(LAYERS))
    w = 0.11
    offsets = np.linspace(-3 * w, 3 * w, 7)
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        means, stds = [], []
        for layer in LAYERS:
            vals = [r[metric] for r in all_rows if r["layer"] == layer
                    and not (r[metric] != r[metric])]  # skip NaN
            means.append(np.mean(vals) if vals else float("nan"))
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        ax.bar(x + offsets[i], means, w, yerr=stds, label=label,
               color=color, capsize=3, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"layer {l}" for l in LAYERS])
    ax.set_ylabel("Spearman rho")
    ax.set_title("Directional probe: learned LoRA direction vs random rank-1 vs full-rank")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/directional_probe.png", dpi=150)
    plt.close(fig)

    print("\n==== DIRECTIONAL PROBE SUMMARY (mean over seeds) ====")
    print(f"{'layer':>5}  {'partial r1':>10}  {'partial full':>12}  "
          f"{'partial dir':>11}  {'ft_acc':>6}")
    for layer in LAYERS:
        rows = [r for r in all_rows if r["layer"] == layer]
        def m(key):
            vals = [r[key] for r in rows if r[key] == r[key]]
            return np.mean(vals) if vals else float("nan")
        print(f"{layer:>5}  {m('rho_low_given_dist'):>+10.3f}"
              f"  {m('rho_full_given_dist'):>+12.3f}"
              f"  {m('partial_rho_directional'):>+11.3f}"
              f"  {m('ft_acc'):>6.3f}")
    print("\nSaved results/logs/directional_probe_results.json + figure.")


if __name__ == "__main__":
    main()
