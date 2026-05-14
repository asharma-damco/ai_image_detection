"""
Error Level Analysis (ELA) Signal.

Re-compresses the image at JPEG quality 85 and measures per-pixel differences.
Authentic JPEG images show uniformly low error (already at compression floor).
Edited regions and AI-generated images show elevated or irregular error patterns.

Score [0, 1]: 0=authentic, 1=fake.

Source: UAIC uaic-fraud-detection/poc/signals/tier1_physics.py — ela_anomaly_score()
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def ela_anomaly_score(img_pil: Image.Image) -> dict:
    """Error Level Analysis at JPEG quality 85.

    Args:
        img_pil: PIL Image (any mode).

    Returns:
        score         float [0–1]   high = likely fake
        heatmap       np.ndarray    uint8 grayscale ELA delta map (H, W)
        ela_mean      float         mean per-pixel error
        cv            float         coefficient of variation
        hotspot_ratio float         fraction of pixels > mean + 2×std
    """
    try:
        QUALITY = 85
        rgb     = img_pil.convert("RGB")

        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=QUALITY)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        orig_arr   = np.array(rgb,          dtype=np.float32)
        recomp_arr = np.array(recompressed, dtype=np.float32)

        delta         = np.abs(orig_arr - recomp_arr).mean(axis=2)  # (H, W)
        mean_d        = float(delta.mean())
        std_d         = float(delta.std())
        cv            = float(std_d / mean_d) if mean_d > 1e-6 else 0.0
        hotspot_ratio = float((delta > mean_d + 2.0 * std_d).mean())

        max_d   = delta.max()
        heatmap = (delta / max_d * 255.0).astype(np.uint8) if max_d > 0 else np.zeros_like(delta, dtype=np.uint8)

        # Smoothness anomaly: low ela_mean → AI-generated (too smooth)
        smooth_score  = float(np.clip((0.06 - mean_d) / 0.04, 0.0, 1.0))
        # Hotspot anomaly: high hotspot_ratio → inpainted region
        hotspot_score = float(np.clip((hotspot_ratio - 0.04) / 0.10, 0.0, 1.0))
        score         = float(np.clip(0.5 * smooth_score + 0.5 * hotspot_score, 0.0, 1.0))

        return {
            "score":         round(score, 4),
            "heatmap":       heatmap,
            "ela_mean":      round(mean_d, 4),
            "cv":            round(cv, 4),
            "hotspot_ratio": round(hotspot_ratio, 4),
        }

    except Exception as exc:
        logger.warning("ela_anomaly_score failed: %s", exc)
        return {
            "score":         0.5,
            "heatmap":       np.zeros((8, 8), dtype=np.uint8),
            "ela_mean":      0.0,
            "cv":            0.0,
            "hotspot_ratio": 0.0,
        }
