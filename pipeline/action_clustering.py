"""K-means clustering over motion feature vectors -> latent pseudo-actions."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def cluster_actions(
    all_features: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster rows of ``all_features`` with KMeans.

    These cluster IDs are **weak pseudo-action labels**, not true robot
    motor commands or language-aligned skills.

    Returns:
        labels: (n_samples,) int latent_action per row
        distances: (n_samples,) Euclidean distance to assigned center
    """
    n_samples = all_features.shape[0]
    if n_samples == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    k = int(max(1, min(n_clusters, n_samples)))
    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    )
    labels = km.fit_predict(all_features).astype(np.int32)
    centers = km.cluster_centers_
    dists = np.linalg.norm(all_features - centers[labels], axis=1)
    return labels, dists.astype(np.float64)
