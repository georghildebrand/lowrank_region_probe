"""
Layer-local stability under single-layer perturbation.

Perturb ONLY layer l's weights (rank-controlled, norm-matched, centroid-
preserving w.r.t. that layer's input activation centroid). Measure gate
stability at layer l with its inputs held fixed — layers below are
unperturbed so the input activations are exact.
"""
import numpy as np
import torch

from cells.ensemble_batched import generate_perturbation_batch, batched_gate_patterns
from cells.regions import hash_patterns
from probes.region_identity import region_identity_from_hashes
from analysis.conditioning import min_hyperplane_distance


def evaluate_layer_stability(model, X, layer, rank, scale, ensemble_size=64, min_mass=10):
    H, Z = model.layer_states(X)[layer]          # input acts + preacts at layer
    W0, b0 = model.layer_weight(layer)
    base_gates = Z > 0                            # [N, out]
    base_hash = hash_patterns(base_gates)
    centroid = H.mean(dim=0)
    dist = min_hyperplane_distance(W0, b0, H).cpu().numpy()

    out = {"dist": dist}
    for mode, suffix in [("lowrank", "low"), ("fullrank", "full")]:
        W, b = generate_perturbation_batch(W0, b0, ensemble_size, rank, scale,
                                           mode, centroid)
        gates = batched_gate_patterns(H, W, b)    # [E, N, out]
        # Hamming: fraction of gates matching base, averaged over ensemble
        matches = (gates == base_gates[None]).float().mean(dim=2)  # [E, N]
        out[f"hamming_{suffix}"] = matches.mean(dim=0).cpu().numpy()
        # Region identity on hashed patterns per ensemble member
        pert_hashes = [hash_patterns(gates[e]) for e in range(ensemble_size)]
        ri, _ = region_identity_from_hashes(base_hash, pert_hashes, min_mass=min_mass)
        out[f"ri_{suffix}"] = ri
    return out


def evaluate_layer_stability_directional(model, X, layer, direction, scale,
                                         ensemble_size=2):
    """Gate stability under +/- perturbation along a fixed direction.
    Returns {"dist", "hamming_dir"} — no region-identity (not needed downstream)."""
    H, Z = model.layer_states(X)[layer]
    W0, b0 = model.layer_weight(layer)
    base_gates = Z > 0
    centroid = H.mean(dim=0)
    dist = min_hyperplane_distance(W0, b0, H).cpu().numpy()

    W, b = generate_perturbation_batch(W0, b0, ensemble_size, 1, scale,
                                       "directional", centroid, direction=direction)
    gates = batched_gate_patterns(H, W, b)                       # [E, N, out]
    matches = (gates == base_gates[None]).float().mean(dim=2)    # [E, N]
    return {"dist": dist, "hamming_dir": matches.mean(dim=0).cpu().numpy()}
