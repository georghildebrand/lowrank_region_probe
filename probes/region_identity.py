"""
Region-identity stability: does the PARTITION survive perturbation, not just
each point's own gate pattern?

For point i with base cell C_i (points sharing i's base-model gate pattern),
under perturbed model m:
    survival_i(m) = (|{j in C_i : pert_hash[j] == pert_hash[i]}| - 1) / (|C_i| - 1)
Score_i = mean_m survival_i(m).
1.0 = cell always stays together; 0.0 = point always separates from all
co-members. Points in cells below min_mass get NaN (no meaningful co-members).
"""
import numpy as np
from collections import defaultdict

from cells.regions import hash_patterns


def region_identity_from_hashes(base_hash, pert_hashes, min_mass=10):
    N = len(base_hash)
    groups = defaultdict(list)
    for i in range(N):
        groups[base_hash[i]].append(i)

    cell_ids = np.full(N, -1, dtype=int)
    cells = []
    for members in sorted(groups.values(), key=len, reverse=True):
        if len(members) >= min_mass:
            cell_ids[members] = len(cells)
            cells.append(np.array(members))

    scores = np.zeros(N, dtype=float)
    for pert in pert_hashes:
        for members in cells:
            h_m = pert[members]
            counts = defaultdict(int)
            for h in h_m:
                counts[h] += 1
            co = np.array([counts[h] for h in h_m], dtype=float)
            scores[members] += (co - 1.0) / max(len(members) - 1, 1)
    scores /= max(len(pert_hashes), 1)
    scores[cell_ids < 0] = np.nan
    return scores, cell_ids


def compute_region_identity_stability(base_model, ensemble, X, min_mass=10):
    base_hash = hash_patterns(base_model.gate_pattern(X))
    pert_hashes = [hash_patterns(m.gate_pattern(X)) for m in ensemble]
    return region_identity_from_hashes(base_hash, pert_hashes, min_mass=min_mass)
