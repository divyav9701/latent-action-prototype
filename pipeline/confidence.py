"""
Proxy confidence scores for weak pseudo-labels.

These scores are **heuristic** and do not certify correctness for robotics.
"""

from __future__ import annotations

import numpy as np


def _normalize_inv_rank(x: np.ndarray) -> np.ndarray:
    """Higher when x is smaller; map to [0, 1] via percentile rank inversion."""
    if x.size == 0:
        return x
    order = np.argsort(np.argsort(x))
    ranks = order.astype(np.float64) / max(x.size - 1, 1)
    return 1.0 - ranks


def _triangle_score(values: np.ndarray, low: float, mid: float, high: float) -> np.ndarray:
    """Piecewise linear: 0 at low, 1 at mid, 0 at high (clamped)."""
    v = values.astype(np.float64)
    out = np.zeros_like(v)
    m1 = (v >= low) & (v < mid)
    m2 = (v >= mid) & (v <= high)
    m3 = v > high
    out[m1] = (v[m1] - low) / max(mid - low, 1e-9)
    out[m2] = (high - v[m2]) / max(high - mid, 1e-9)
    out[m3] = 0.0
    return np.clip(out, 0.0, 1.0)


def compute_confidence(
    features: np.ndarray,
    cluster_distances: np.ndarray,
    clip_row_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute motion, consistency, smoothness, and overall confidence.

    Args:
        features: (N, D) motion feature rows (same order as pairs).
        cluster_distances: (N,) distance to assigned KMeans center.
        clip_row_indices: (N,) integer clip id per row (same clip = same id).

    Returns:
        motion_score, consistency_score, smoothness_score, confidence (each (N,))
    """
    n = features.shape[0]
    if n == 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z, z, z

    mean_mag = features[:, 2]
    p10, p50, p90 = np.percentile(mean_mag, [10, 50, 90])
    low = float(max(p10, 1e-6))
    mid = float(max(p50, low * 1.01))
    high = float(max(p90, mid * 1.01))
    motion_score = _triangle_score(mean_mag, low, mid, high)

    consistency_score = _normalize_inv_rank(cluster_distances.astype(np.float64))

    # Smoothness: low feature delta to temporal neighbors within the same clip
    smoothness = np.ones(n, dtype=np.float64)
    if n > 1:
        deltas = np.linalg.norm(np.diff(features, axis=0), axis=1)
        same_clip_next = clip_row_indices[1:] == clip_row_indices[:-1]
        forward = np.zeros(n, dtype=np.float64)
        backward = np.zeros(n, dtype=np.float64)
        # forward[i] uses delta between i and i+1
        for i in range(n - 1):
            if same_clip_next[i]:
                d = float(deltas[i])
                s = 1.0 / (1.0 + d)
                forward[i] = s
                backward[i + 1] = s
        smoothness = 0.5 * (forward + backward)
        alone = (forward + backward) < 1e-9
        smoothness[alone] = 0.5

    confidence = (
        0.4 * motion_score + 0.4 * consistency_score + 0.2 * smoothness
    )
    confidence = np.clip(confidence, 0.0, 1.0)
    motion_score = np.clip(motion_score, 0.0, 1.0)
    consistency_score = np.clip(consistency_score, 0.0, 1.0)
    smoothness = np.clip(smoothness, 0.0, 1.0)

    return motion_score, consistency_score, smoothness, confidence
