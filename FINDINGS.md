# Findings

Eleven experiments testing the hypothesis: *stable ReLU gate regions under
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

## Experiment 9 — Multi-layer LoRA prediction

`make multilayer-lora` — all three layers fine-tuned simultaneously (rank=1
LoRA on each, 1000 steps), then per-layer attribution measured via ablation:
|forward_with_delta(base, X, l, deltas[l]) − base(X)|. Does layer-l fragility
predict layer-l's contribution to the full multi-layer fine-tune? 3 seeds.

Mean over seeds:

| layer | partial ρ(r1\|dist) | partial ρ(full\|dist) | ρ(−dist) | ft_acc | mean \|Δlogit\| |
|---|---|---|---|---|---|
| 0 | −0.030 | −0.037 | −0.010 | 0.964 | 4.857 |
| 1 | +0.004 | −0.007 | −0.031 | 0.964 | 3.676 |
| 2 | +0.014 | +0.043 | −0.222 | 0.964 | 1.421 |

Per-seed at layer 2 (the strongest single-layer signal in Exp 6): +0.046
(seed 0), −0.004 (seed 1), −0.001 (seed 2) — no consistent sign across seeds.

**Result: no meaningful signal at full convergence.** Multi-layer training
lifts ft_acc to 0.964 (adapter converges, unlike Exp 8's stuck layer-2
single-layer adapter). With a converged adapter, partial ρ(r1|dist) collapses
to noise at every layer.

Note on the smoke run artifact: a 100-step partial-convergence run (seed=0
only, ft_acc=0.877) showed layer-2 partial_r1=+0.307 — a strong false positive.
At partial convergence the fine-tune has not yet reached fragile (boundary-
proximate) points; the ablation signal is dominated by the safer, boundary-far
points that happen to be fragile in training, spuriously inflating the
correlation. Once the adapter converges fully (0.964) and has moved all
relevant points, the spurious signal vanishes. This confirms: the Exp 6
positive signal was a partial-convergence artifact from the stuck layer-2
adapter, not a structural LoRA connection.

## Experiment 10 — Polytopeness gradient

`make polytopeness` — Exp 3 (label-shuffle) design repeated across a
boundary-softness gradient on the 2D checkerboard. New dataset
`generate_soft_checkerboard(softness)`: labels drawn
`y ~ Bernoulli(sigmoid(signed_margin / softness))`, so softness=0 is the
hard checkerboard and larger values blur the quadrant boundaries.
3 seeds × 5 softness levels × 3 scales; geometry bonus = trained −
shuffled partial ρ, identical-init control.

(Methods note: this experiment seeds `torch.manual_seed(seed)` directly
before EACH model construction, making the identical-init control exact.
The original `run_label_shuffle.py`/`run_gmm.py` seed after construction,
so their shuffled models actually had different inits — that does not
invalidate the Exp 3/4 control (any-converged-geometry comparison), but
the "identical init" description was only aspirational there.)

Mean over seeds and scales:

| softness | geometry bonus | acc_trained |
|---|---|---|
| 0.0 | **+0.119** | 1.000 |
| 0.05 | −0.072 | 0.936 |
| 0.1 | −0.037 | 0.879 |
| 0.2 | −0.045 | 0.789 |
| 0.4 | −0.059 | 0.683 |

**Result: a cliff, not a gradient.** The prediction was continuous decay;
instead the bonus exists only at exactly softness=0 and collapses to a
small *negative* value at the first nonzero softness, staying flat
thereafter. The rank-1 geometry bonus is not proportional to "how
polytope-like" the data is — it requires an exactly-hard piecewise
partition. The moment the optimal decision boundary has any soft margin,
the trained network's geometry behaves like the soft-boundary regimes
(GMM, images): slightly more fragile than the shuffled control.

## Experiment 11 — Function-weighted gate flips

`make functional-flips` — per point, TWO stability metrics from the SAME
rank-1/full-rank perturbation ensemble: per-gate Hamming stability, and
functional stability (−mean |Δlogit|). Trained vs identical-init
label-shuffled, on 2D checkerboard and 5D GMM. Rescue prediction: trained
networks' gate flips are benign — fragile in Hamming, robust in function.
(Metric note: bonuses here use per-gate Hamming, not the exact-pattern
metric of Exp 3/10 — numbers are not directly comparable across
experiments.)

Mean over 3 seeds × 2 scales:

| dataset | bonus (Hamming) | bonus (functional) | per-flip excess trained | per-flip excess shuffled | mean flips t/s |
|---|---|---|---|---|---|
| checkerboard2d | +0.003 | −0.010 | **0.758** | 0.061 | 2.05 / 0.81 |
| gmm5d | +0.002 | +0.044 | 0.225 | 0.155 | 3.06 / 2.15 |

`per-flip excess` = flip-attributable |Δlogit| per flipped gate, with the
smooth no-flip |Δlogit| baseline at the same point subtracted (the naive
|Δlogit|/flips estimator is confounded by the within-region component and
was replaced after review).

**Result: the rescue fails, inverted.** The functional geometry bonus is
noise on both datasets (signs inconsistent across seeds). And the
benign-flip prediction is contradicted outright: on the checkerboard,
trained networks' flips carry **~12× more** function change per flip than
shuffled controls (0.758 vs 0.061, trained > shuffled in 6/6 cells), and
trained networks also flip *more* gates on average (2.05 vs 0.81). GMM
shows the same direction more weakly (4/6 cells). Trained gate flips are
functionally *loaded*, not benign: training concentrates the function on
exactly the gates the probe flips.

## Interpretation

One mechanism explains all six LoRA experiments:

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
4. **The LoRA prediction (Exp 6) does not survive scrutiny** (Experiments 7–9):
   the +0.190 partial correlation at layer 2 collapses when the directional
   probe shows negative signal (Exp 7), and the rank/steps sweep shows the
   layer-2 single-layer adapter never converges (ft_acc=0.492 = chance, Exp 8).
   Multi-layer LoRA achieves full convergence (ft_acc=0.964, Exp 9) but
   partial ρ collapses to noise — confirming the Exp 6 positive was a
   partial-convergence artifact from a stuck adapter, not a structural signal.
5. **The optimizer selects directions that protect fragile points' gates.**
   Directional-probe partial ρ is negative (Exp 7): gates stable in the
   learned LoRA direction are *more* likely to have logits change. The
   fine-tuning optimizer avoids gate disruption at fragile (boundary-close)
   points as a by-product of its own convergence dynamics, not by design.
6. **Fragility does not predict per-layer adaptation attribution at convergence.**
   Even with a fully converged multi-layer adapter and layer-resolved ablation
   signal (Exp 9), partial ρ(r1|dist) is indistinguishable from zero. Random
   gate fragility carries no information about which layer a converged
   fine-tuning adapter uses to reshape the logit surface. The LoRA connection
   hypothesis is fully refuted.
7. **The geometry bonus is a threshold effect, not a gradient** (Exp 10).
   The positive bonus requires an exactly-hard piecewise partition; the
   smallest boundary softness (0.05) kills it entirely and flips its sign.
   There is no continuum from "polytope data" to "soft data" — the original
   hypothesis holds on a measure-zero corner of data space.
8. **Gate flips in trained networks are functionally loaded** (Exp 11).
   The last rescue path — trained networks fragile in Hamming but robust in
   function — is inverted: per flipped gate, trained networks change their
   output ~12× more than shuffled controls on the checkerboard. Training
   does not build redundancy around its gates; it concentrates function ON
   them. This is the same boundary-commitment mechanism seen from a third
   angle: committed gates are few, load-bearing, and rank-1-reachable.

## Open follow-ups

- The negative directional-probe effect (Exp 7) is a new phenomenon worth
  isolating: is it specific to gradient-descent LoRA, or does any rank-1
  update to a trained network preferentially avoid fragile-point gate flips?
- The softness cliff (Exp 10) deserves a finer sweep between 0 and 0.05 to
  locate the transition — is it sharp in softness, or does it track the
  probe scale (bonus dies when label noise width exceeds the perturbation's
  boundary displacement)?
- Per-flip functional loading (Exp 11) suggests a positive-framing follow-up:
  committed gates as a *sparse function skeleton* — can the top-k
  highest-per-flip-impact gates alone reconstruct most of the decision
  surface?
- Region-identity metric at layers 0–1 on real data is starved by
  `min_mass` filtering (256/128 gates → few cells with ≥10 members);
  needs a coarser cell definition (e.g. top-k active gates) to be
  informative there.
