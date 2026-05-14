"""
CFA Demosaicing Correlation Signal.

Real cameras: R-noise and B-noise share scene structure → moderate positive
channel noise correlation (rb_corr ≈ 0.15–0.45). AI generators produce
channels jointly (over-correlated) or independently (under-correlated).

Authentic band: [0.05, 0.75]. Deviations in either direction score towards 1.

Score [0, 1]: 0=authentic, 1=fake.

Source: UAIC uaic-fraud-detection/poc/signals/tier1_physics.py — cfa_correlation_score()
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cfa_correlation_score(img_array: np.ndarray) -> dict:
    """Channel noise cross-correlation via Wiener-filter residuals.

    Args:
        img_array: RGB image array, uint8 or float32, shape (H, W, 3) or (H, W).

    Returns:
        score    float [0–1]  high = likely fake
        rb_corr  float        Pearson r between R-noise and B-noise residuals
        rg_corr  float        Pearson r between R-noise and G-noise residuals
        rb_score float        per-pair anomaly score for R/B channels
        rg_score float        per-pair anomaly score for R/G channels
    """
    try:
        from scipy.signal import wiener           # type: ignore
        from scipy.ndimage import uniform_filter  # type: ignore

        img = _ensure_rgb_float(img_array)
        R   = img[:, :, 0].astype(np.float64)
        G   = img[:, :, 1].astype(np.float64)
        B   = img[:, :, 2].astype(np.float64)

        # Remove JPEG 8×8 blocking structure before residual extraction
        R_hp = R - uniform_filter(R, size=8)
        G_hp = G - uniform_filter(G, size=8)
        B_hp = B - uniform_filter(B, size=8)

        r_noise = R_hp - wiener(R_hp, mysize=5)
        g_noise = G_hp - wiener(G_hp, mysize=5)
        b_noise = B_hp - wiener(B_hp, mysize=5)

        rb_corr = _pearson(r_noise.flatten(), b_noise.flatten())
        rg_corr = _pearson(r_noise.flatten(), g_noise.flatten())

        def _pair_score(corr: float) -> float:
            if corr > 0.75:
                return float(np.clip((corr - 0.75) / 0.25, 0.0, 1.0))
            if corr < 0.05:
                return float(np.clip((0.05 - corr) / 0.15, 0.0, 1.0))
            return 0.0

        rb_score = _pair_score(rb_corr)
        rg_score = _pair_score(rg_corr)
        score    = float(max(rb_score, rg_score))

        return {
            "score":    round(score,    4),
            "rb_corr":  round(rb_corr,  4),
            "rg_corr":  round(rg_corr,  4),
            "rb_score": round(rb_score, 4),
            "rg_score": round(rg_score, 4),
        }

    except Exception as exc:
        logger.warning("cfa_correlation_score failed: %s", exc)
        return {"score": None, "rb_corr": 0.0, "rg_corr": 0.0, "rb_score": 0.0, "rg_score": 0.0}


def _ensure_rgb_float(img_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(img_array, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    a_c = a - a.mean();  b_c = b - b.mean()
    denom = float(np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum()))
    return 0.0 if denom < 1e-10 else float(np.dot(a_c, b_c) / denom)
