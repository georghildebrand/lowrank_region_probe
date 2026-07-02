# Findings

Five experiments testing the hypothesis: *stable ReLU gate regions under
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

## Interpretation

One mechanism explains all five experiments:

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

## Open follow-ups

- Probe fragility as a *predictor* of where low-rank fine-tuning changes
  behavior (the LoRA connection, the most promising direction).
- "Polytopeness" gradient: interpolate ground-truth boundary softness and
  show the bonus decays continuously.
- Region-identity metric at layers 0–1 on real data is starved by
  `min_mass` filtering (256/128 gates → few cells with ≥10 members);
  needs a coarser cell definition (e.g. top-k active gates) to be
  informative there.
