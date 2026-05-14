"""Dense optical flow features for consecutive frame pairs."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def _circular_mean_angle(angles: np.ndarray, weights: np.ndarray) -> float:
    """Weighted circular mean of angles in radians."""
    if angles.size == 0:
        return 0.0
    s = np.sum(weights * np.sin(angles))
    c = np.sum(weights * np.cos(angles))
    return float(math.atan2(s, c))


def compute_flow_features(frame_paths: list[str | Path]) -> np.ndarray:
    """
    For each consecutive frame pair, compute Farneback optical flow stats.

    Returns array of shape (n_pairs, 6) with columns:
    mean_dx, mean_dy, mean_magnitude, std_magnitude,
    dominant_angle (radians), percent_moving_pixels
    """
    if len(frame_paths) < 2:
        return np.zeros((0, 6), dtype=np.float64)

    feats: list[np.ndarray] = []
    prev_gray: np.ndarray | None = None

    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            prev_gray = None
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        prev_gray = gray

        dx = flow[..., 0].astype(np.float64)
        dy = flow[..., 1].astype(np.float64)
        mag = np.sqrt(dx * dx + dy * dy)

        mean_dx = float(np.mean(dx))
        mean_dy = float(np.mean(dy))
        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))

        move_thresh = max(1e-3, 0.05 * (np.percentile(mag, 95) + 1e-6))
        mask = mag > move_thresh
        pct_moving = float(np.mean(mask))

        angles = np.arctan2(dy, dx)
        if np.any(mask):
            dom_angle = _circular_mean_angle(angles[mask], mag[mask])
        else:
            dom_angle = float(math.atan2(mean_dy, mean_dx))

        feats.append(
            np.array(
                [mean_dx, mean_dy, mean_mag, std_mag, dom_angle, pct_moving],
                dtype=np.float64,
            )
        )

    if not feats:
        return np.zeros((0, 6), dtype=np.float64)
    return np.stack(feats, axis=0)
