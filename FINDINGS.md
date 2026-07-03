# Findings

Eight experiments testing the hypothesis: *stable ReLU gate regions under
low-rank weight perturbations correspond to structural partitions learned
from data.*

**Verdict: not supported at scale — the sign inverts.** Trained networks are
consistently MORE fragile than label-shuffled controls under rank-1
perturbation on real data and at depth. The effect is robustly
rank-specific, which validates the probe itself; what it detects is
boundary commitment, not region robustness.

## Method (shared across experiments)

- Perturb first-layer (or one chosen layer's) weights: `ΔW = scale · (B @ A)`
  (rank-r) vs full-rank Gaussian control, both scaled to the exact Frobenius
  norm `scale · ||W0||` — norm-matched, so only structure differs.
- Bias correction is centroid-preserving: `b_new = b0 + (W0 − W_new) @ c`
  with `c` the mean of the layer's input activations, so the signed offset
  at the data centroid is invariant.
- Confound controls:
  - **Distance conditioning**: partial Spearman ρ(stability, mode | distance
    to nearest gate hyperplane) — removes the trivial "far points survive
    anything" effect.
  - **Label-shuffle control**: identical-init model trained on permuted
    labels — isolates *learned* geometry from *any converged* geometry.
    Geometry bonus = trained ρ − shuffled ρ.
  - **Rank-specificity control**: rank-8 must behave like full-rank noise.
- Two stability metrics:
  - **Pointwise exact**: fraction of ensemble members where the point's full
    gate pattern is unchanged.
  - **Region identity** (stricter): fraction of a point's base-cell
    co-members that still share its gate pattern under perturbation — does
    the *partition* survive, not just the point.

## Experiment 1 — Rank × scale sweep (2D synthetic polytope)

`make sweep` — 3 seeds × ranks {1,2,4,8} × scales {0.02,0.05,0.1}.

Low-rank advantage exists and survives distance conditioning, but only at
very low rank:

| rank | mean partial ρ |
|---|---|
| 1 | +0.289 |
| 2 | +0.191 |
| 4 | +0.076 |
| 8 | +0.023 (≈ noise) |

## Experiment 2 — Label-shuffle control (2D polytope + 5D GMM)

`make label-shuffle`, `make gmm` — pointwise metric, rank=1.

| dataset | scale | trained ρ | shuffled ρ | bonus | Wilcoxon |
|---|---|---|---|---|---|
| 2D polytope | 0.1 | +0.522 | +0.223 | **+0.298** | p=0.00012 |
| 5D GMM | 0.1 | +0.783 | +0.740 | +0.043 | p=0.00085 |

Significant on both, but the bonus collapses from +57% of the trained effect
(2D) to +5.5% (5D). The 2D ground truth is itself a polytope; the GMM has
soft curved boundaries.

## Experiment 3 — Capacity-ratio sweep (5D GMM)

`make capacity-ratio` — hidden_dim ∈ {8..256} at input_dim=5.

Prediction (bonus grows as capacity shrinks) was **wrong**: bonus goes
*negative* at low hidden_dim (−0.108 at hidden=8) despite similar accuracy.
Capacity-constrained trained networks commit hyperplanes tightly to true
class boundaries and become MORE rank-1-fragile than arbitrary shuffled
geometry.

## Experiment 4 — Region-identity re-scoring (2D + 5D)

`make region-identity` — same design as Experiment 2, stricter metric.

| dataset | scale=0.1 pointwise bonus | region-identity bonus |
|---|---|---|
| 2D polytope | +0.298 | +0.089 |
| 5D GMM | +0.043 | **−0.095 (reversed)** |

Wilcoxon p=0.756 overall — **the geometry bonus does not survive the
partition-level metric.** Learned geometry helps individual gate patterns
persist, not whole regions.

## Experiment 5 — Real data, layer-wise (MNIST / Fashion-MNIST)

`make real-data` — 3-hidden-layer MLP (784→256→128→64), binary splits,
single-layer perturbation, 3 seeds, rank {1,8}, scale 0.1.

Geometry bonus (trained − shuffled partial ρ), rank=1:

| split | layer 0 | layer 1 | layer 2 (hamming) | layer 2 (region-id) |
|---|---|---|---|---|
| MNIST even/odd | +0.107 | −0.042 | **−0.245** | −0.103 |
| Fashion easy | +0.015 | −0.081 | −0.005 | −0.025 |
| Fashion hard | +0.043 | −0.091 | **−0.258** | −0.265 |

Rank-8 control ≈ 0 everywhere. Trained acc ≥ 0.99, shuffled memorization
acc ≈ 0.90.

## Experiment 6 — LoRA prediction (MNIST, layer-wise)

`make lora-prediction` — base model trained on even/odd; each layer LoRA
fine-tuned (rank-1, frozen base, 300 steps) onto digit<5 with the *same*
inputs; per-point |Δlogit| on eval correlated with probe fragility.
3 seeds × 3 layers.

Mean over seeds:

| layer | ρ(frag_r1) | ρ(frag_full) | ρ(−dist) | partial ρ(r1\|dist) | partial ρ(full\|dist) | ft_acc |
|---|---|---|---|---|---|---|
| 0 | −0.106 | −0.115 | −0.041 | −0.100 | −0.108 | 0.884 |
| 1 | +0.011 | +0.028 | −0.028 | +0.044 | +0.061 | 0.761 |
| 2 | −0.420 | −0.393 | −0.559 | **+0.190** | **+0.223** | 0.615 |

Two results:

1. **Fragility-given-distance does predict fine-tune susceptibility at the
   deep layer.** Raw correlations at layer 2 are dominated by a distance
   confound with the *opposite* sign — far-from-boundary points carry large
   logits and LoRA shifts them most (ρ(−dist)=−0.559). Conditioned on
   distance, fragile points change *more*, positive in all 3 seeds
   (+0.260/+0.176/+0.133).
2. **But the prediction is NOT rank-1-specific.** The full-rank probe's
   partial correlation is as large or larger (+0.223 vs +0.190, every
   seed). Generic gate fragility at matched distance predicts where LoRA
   moves the model; the perturbation *structure* adds nothing predictive.

Caveats: rank-1 LoRA on this relabeling is weak at depth (layer-2 ft_acc
0.49–0.84, seed-dependent), so layer-2 |Δlogit| partly reflects an
under-trained adapter; layers 0–1 fine-tune better yet show no fragility
signal at all. Experiments 7–8 follow up on both caveats.

## Experiment 7 — Directional probe (learned LoRA direction vs random)

`make directional-probe` — same base task as Exp 6 but the probe direction
is extracted from the *actual learned delta* via SVD top singular direction
(unit Frobenius). The ±direction ensemble (2 members, norm-matched) replaces
the 64-member random ensemble for the directional signal. 3 seeds × 3 layers.

Mean over seeds:

| layer | partial ρ(r1\|dist) | partial ρ(full\|dist) | partial ρ(dir\|dist) |
|---|---|---|---|
| 0 | −0.100 | −0.108 | **−0.220** |
| 1 | +0.044 | +0.061 | **−0.142** |
| 2 | +0.190 | +0.223 | **−0.066** |

**Result: directional probe consistently ANTI-predicts behavior change.**
Across 8/9 seed-layer cells, `partial_dir` is negative while `partial_r1`
and `partial_full` are positive at the deep layer. Gates that are *stable*
under perturbation in the learned LoRA direction tend to have their logits
change *more*, not less.

Interpretation: the optimizer finds a weight direction that avoids flipping
the gates of the most fragile (boundary-proximate) points — these points are
left structurally intact while the logit surface shifts under them. The
LoRA-direction probe is detecting gate stability in the *specifically chosen*
subspace, which turns out to be the subspace the optimizer treats as safe to
perturb without gate disruption. This is a meaningful null: if the LoRA
direction were purely isotropic, partial_dir would be near zero like random
rank-8; that it is *negative* shows the optimizer actively selects a
direction along which fragile-point gates are protected.

## Experiment 8 — LoRA rank/steps sweep

`make lora-sweep` — Exp 6 design repeated across LoRA ranks {1,4,8} ×
steps {300,1000}, probe rank=1 fixed, seed=0 only.

Selected rows (full table in `results/logs/lora_sweep_results.json`):

| layer | rank | steps | ft_acc | partial ρ(r1\|dist) | partial ρ(full\|dist) |
|---|---|---|---|---|---|
| 0 | 1 | 300 | 0.873 | −0.003 | −0.042 |
| 0 | 8 | 1000 | **0.982** | +0.022 | −0.006 |
| 1 | 1 | 300 | 0.749 | −0.006 | −0.003 |
| 1 | 4 | 300 | 0.929 | +0.013 | +0.001 |
| 2 | 1 | 300 | 0.492 | +0.260 | +0.340 |
| 2 | 8 | 1000 | **0.492** | +0.260 | +0.340 |

Two findings:

1. **Layer 2 LoRA is stuck at chance.** ft_acc=0.492 (≈ balanced chance)
   for every (rank, steps) combination; `|delta|` grows with rank but
   correlations are *identical across all 18 cells at layer 2*. The
   fine-tuning optimizer cannot move the decision surface via layer 2 alone
   for this task. Correlations at layer 2 in Exp 6 reflect |Δlogit| from a
   never-converged adapter, not a meaningful fine-tuning signal.
2. **Fragility signal does NOT strengthen as adapter converges.** Layers 0–1
   improve substantially with rank/steps (0.749→0.982 ft_acc), but
   partial ρ(r1|dist) stays near zero (range −0.046 to +0.022). No evidence
   that a better-converged adapter reveals fragility as a predictor.

## Interpretation

One mechanism explains all five perturbation experiments:

1. **The probe is real**: every effect — positive or negative — is specific
   to rank-1 perturbations and vanishes by rank 8 at matched norm. Gate
   stability responds to perturbation *structure*, not magnitude.
2. **Training buys boundary commitment, not region robustness.** Where the
   true partition is natively polytope-shaped (2D checkerboard), learned
   regions do survive coherent low-rank tilt better than arbitrary ones. On
   soft-boundary data (GMM, images) and at depth, trained representations
   sit tightly against their decision surfaces, and a coherent rank-1 tilt
   breaks them *more* easily than the arbitrary geometry of a
   label-shuffled control.
3. **Inverted reading**: rank-1 *fragility*, not stability, is the
   signature of learned structure in realistic settings. Deep layers of
   trained networks are maximally rank-1-sensitive — consistent with why
   low-rank adapters (LoRA-style) steer trained networks so effectively.
4. **The LoRA prediction (Exp 6) does not survive scrutiny** (Experiments 7–8):
   the +0.190 partial correlation at layer 2 collapses when the directional
   probe shows negative signal (Exp 7), and the rank/steps sweep shows the
   layer-2 adapter never converges (ft_acc=0.492 = chance across all rank
   and step counts, Exp 8). The Exp 6 signal was a residual distance
   confound from a stuck adapter, not a structural LoRA connection.
5. **The optimizer selects directions that protect fragile points' gates.**
   Directional-probe partial ρ is negative (Exp 7): gates stable in the
   learned LoRA direction are *more* likely to have logits change. The
   fine-tuning optimizer avoids gate disruption at fragile (boundary-close)
   points as a by-product of its own convergence dynamics, not by design.

## Open follow-ups

- Layer 2 is gated for single-layer LoRA on this task; try fine-tuning
  all layers simultaneously, or use a task where layer-2-only adaptation
  can converge, to get a meaningful |Δlogit| signal at depth.
- The negative directional-probe effect (Exp 7) is a new phenomenon worth
  isolating: is it specific to gradient-descent LoRA, or does any rank-1
  update to a trained network preferentially avoid fragile-point gate flips?
- "Polytopeness" gradient: interpolate ground-truth boundary softness and
  show the bonus decays continuously.
- Region-identity metric at layers 0–1 on real data is starved by
  `min_mass` filtering (256/128 gates → few cells with ≥10 members);
  needs a coarser cell definition (e.g. top-k active gates) to be
  informative there.
