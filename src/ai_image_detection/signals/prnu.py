"""
PRNU Noise Floor Anomaly Signal (FFT-based).

AI generators concentrate energy in mid-frequencies (over-smooth textures)
and suppress high-frequencies (lack of sensor noise). Real scanned documents
have balanced mid/high frequency energy.

Score [0, 1]: 0=authentic, 1=fake.

Source: UAIC uaic-fraud-detection/poc/signals/tier1_physics.py — prnu_anomaly_score()
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def prnu_anomaly_score(img_array: np.ndarray) -> dict:
    """FFT spatial frequency band ratio for PRNU anomaly detection.

    Band definitions (% of max radial frequency):
        Low:  0–8%    DC + near-DC structure
        Mid:  8–40%   content frequencies (AI over-represents)
        High: 40–100% texture, noise, fine detail (AI under-represents)

    Args:
        img_array: RGB or grayscale image array, uint8 or float32.

    Returns:
        score      float [0–1]  high = likely fake
        mid_ratio  float        fraction of power in mid-frequency band
        high_ratio float        fraction of power in high-frequency band
        fft_score  float        raw combined FFT score (== score)
    """
    try:
        gray = _to_gray_float(img_array).astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
        h, w = gray.shape

        if h < 16 or w < 16:
            return {"score": None, "mid_ratio": 0.0, "high_ratio": 0.0, "fft_score": None}

        fft_shift = np.fft.fftshift(np.fft.fft2(gray))
        power     = (np.abs(fft_shift) ** 2).astype(np.float64)

        cy, cx = h / 2.0, w / 2.0
        ys     = (np.arange(h) - cy) / (h / 2.0)
        xs     = (np.arange(w) - cx) / (w / 2.0)
        xv, yv = np.meshgrid(xs, ys)
        r_map  = np.sqrt(xv ** 2 + yv ** 2)

        total_energy = float(power.sum()) + 1e-10
        mid_ratio    = float(power[(r_map > 0.08) & (r_map <= 0.40)].sum() / total_energy)
        high_ratio   = float(power[r_map > 0.40].sum()                    / total_energy)

        mid_score  = float(np.clip((mid_ratio  - 0.50) / 0.20, 0.0, 1.0))
        high_score = float(np.clip((0.20 - high_ratio) / 0.15, 0.0, 1.0))
        fft_score  = float(np.clip(0.6 * mid_score + 0.4 * high_score, 0.0, 1.0))

        return {
            "score":      round(fft_score, 4),
            "mid_ratio":  round(mid_ratio,  4),
            "high_ratio": round(high_ratio, 4),
            "fft_score":  round(fft_score,  4),
        }

    except Exception as exc:
        logger.warning("prnu_anomaly_score failed: %s", exc)
        return {"score": None, "mid_ratio": 0.0, "high_ratio": 0.0, "fft_score": None, "error": str(exc)}


def _to_gray_float(img_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(img_array, dtype=np.float64)
    if arr.ndim == 3:
        arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2] if arr.shape[2] >= 3 else arr[:, :, 0]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr
