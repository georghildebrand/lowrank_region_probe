"""
5D Gaussian mixture dataset with known latent partition.
4 clusters in R^5, well-separated, checkerboard labels.
"""
import numpy as np
import torch


def generate_gaussian_mixture(n_samples=10000, dim=5, n_clusters=4, separation=3.0, seed=None):
    rng = np.random.default_rng(seed)

    # Cluster centers: vertices of a hypercube scaled by separation
    centers = rng.choice([-1, 1], size=(n_clusters, dim)) * separation

    n_per = n_samples // n_clusters
    counts = [n_per] * n_clusters
    counts[-1] += n_samples - sum(counts)  # remainder to last cluster

    X_parts, cluster_ids = [], []
    for i, (c, n) in enumerate(zip(centers, counts)):
        pts = rng.normal(loc=c, scale=1.0, size=(n, dim))
        X_parts.append(pts)
        cluster_ids.extend([i] * n)

    X = np.vstack(X_parts)
    cluster_ids = np.array(cluster_ids, dtype=np.int64)

    # Checkerboard labels: cluster index parity
    y = (cluster_ids % 2).astype(float)

    # Shuffle so clusters are interleaved in the sample index
    perm = rng.permutation(n_samples)
    X, y, cluster_ids = X[perm], y[perm], cluster_ids[perm]

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        torch.tensor(cluster_ids, dtype=torch.long),
    )
