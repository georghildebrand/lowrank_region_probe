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
To run the main experiment probe:
```bash
make run
```

To clean previous results:
```bash
make clean
```

## Scope
This repository explores **stability under weight perturbation**, not training dynamics. Stable regions under low-rank perturbations are candidates for **structural ReLU polytopes**.
