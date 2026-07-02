"""Shared distance-conditioning helpers for stability experiments."""
import numpy as np
import torch
from scipy.stats import spearmanr


def min_hyperplane_distance(W, b, X):
    """Geometric distance |w.x + b| / ||w||, min over units. X:[N,d], W:[h,d], b:[h]."""
    with torch.no_grad():
        z = X @ W.T + b
        norms = torch.norm(W, dim=1)
        return torch.min(torch.abs(z) / (norms + 1e-12), dim=1)[0]


def partial_spearman_mode_given_distance(s_a, s_b, dist):
    """Partial Spearman rho(stability, mode | distance), paired sample.

    ``dist`` must be the **shared** distance array covering the same spatial
    points for both modes — i.e. the same values are used for the s_a half and
    the s_b half of the concatenated sample.  This shared-distance contract is
    what makes rho(mode, dist) = 0 by construction, reducing the full partial-
    correlation formula to the simpler ``rho_ym / sqrt(1 - rho_yd^2)``.
    Passing per-mode distance arrays (where the two halves differ) would
    violate this assumption and silently produce wrong results.

    NaN entries (from region-identity min_mass filtering) are dropped pairwise.
    """
    s_a, s_b, dist = np.asarray(s_a, float), np.asarray(s_b, float), np.asarray(dist, float)
    valid = ~(np.isnan(s_a) | np.isnan(s_b) | np.isnan(dist))
    s_a, s_b, dist = s_a[valid], s_b[valid], dist[valid]

    y = np.concatenate([s_a, s_b])
    m = np.concatenate([np.ones_like(s_a), np.zeros_like(s_b)])
    d = np.concatenate([dist, dist])
    rho_ym = spearmanr(y, m)[0]
    rho_yd = spearmanr(y, d)[0]
    return rho_ym / np.sqrt(max(1.0 - rho_yd ** 2, 1e-9))
