# Low-Rank Region Probe

## Goal
This project implements an experiment to detect **stable ReLU gate regions under low-rank weight perturbations**.

## Concept
Low-rank perturbations behave like **global deformation fields** on the ReLU polytope complex. Regions that remain stable under these perturbations may correspond to **structural partitions learned from data**.

## Method
1. Train a base MLP model on a dataset.
2. Sample an ensemble of low-rank perturbations ($\Delta W = scale \cdot (B @ A)$).
3. Measure gate-pattern stability across the ensemble.
4. Identify stable regions and analyze the relationship between region mass and stability.

## Metrics
- **Point Stability**: Fraction of ensemble models where the point's gate pattern remains identical to the base model.
- **Region Stability**: Mean point stability within a region (defined by a unique gate pattern).
- **Boundary Sensitivity**: $1 - \text{Stability}$.

## Running Experiments

| Target | Experiment |
|---|---|
| `make run` | Original structural-cell probe (2D polytope) |
| `make sweep` | Rank × scale × seed sweep, distance-conditioned lowrank vs fullrank |
| `make label-shuffle` | Trained vs label-shuffled control (2D polytope) |
| `make gmm` | 5D Gaussian-mixture replication |
| `make capacity-ratio` | Geometry bonus vs hidden_dim/input_dim ratio |
| `make region-identity` | Re-scoring with partition-level (region-identity) metric |
| `make real-data` | MNIST / Fashion-MNIST, 3-layer MLP, layer-wise probe |
| `make lora-prediction` | Probe fragility vs where LoRA fine-tune changes behavior |
| `make directional-probe` | Probe in learned LoRA direction vs random rank-1 vs full-rank |
| `make lora-sweep` | LoRA rank × steps sweep — does fragility signal strengthen as adapter converges? |
| `make clean` | Remove previous results |

Tests: `conda run -n plora python3 -m pytest tests/`

## Results

See **[FINDINGS.md](FINDINGS.md)**. Short version: the original hypothesis
is not supported at scale — on real data and at depth, trained networks are
*more* fragile under rank-1 perturbation than label-shuffled controls, and
the effect is strictly rank-specific. The probe detects boundary
commitment, not region robustness.

## Scope
This repository explores **stability under weight perturbation**, not training dynamics.
