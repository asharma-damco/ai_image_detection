"""
DCT Coefficient Benford's Law Signal.

Tests whether the first significant digit of non-zero JPEG AC coefficients
follows Benford's Law. Natural images conform closely; AI-generated images
deviate because generators learn priors rather than physical scene statistics.

Requires raw JPEG bytes or a file path for meaningful results.
Falls back to a pixel-domain blockiness score when a decoded array is passed.

Score [0, 1]: 0=authentic, 1=fake.

Source: UAIC uaic-fraud-detection/poc/signals/tier1_physics.py — dct_benford_score()
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

_BENFORD: np.ndarray = np.array(
    [math.log10(1.0 + 1.0 / d) for d in range(1, 10)],
    dtype=np.float64,
)


def dct_benford_score(img_source: Union[str, bytes, np.ndarray]) -> dict:
    """DCT coefficient statistics and Benford's Law test.

    Args:
        img_source: File path (str), raw JPEG bytes, or decoded image array.
                    Raw JPEG path/bytes required for valid coefficient extraction.

    Returns:
        score            float [0–1]   high = likely fake
        chi2_stat        float         chi-square statistic (8 df)
        p_value          float         p-value
        zero_pct         float         fraction of AC coefficients == 0
        compressed_input bool          True if raw JPEG coefficients were used
    """
    try:
        from scipy.stats import chi2 as _scipy_chi2  # type: ignore

        if isinstance(img_source, (str, bytes)):
            try:
                import jpegio  # type: ignore
            except ImportError:
                logger.warning("dct_benford_score: jpegio not installed — pip install jpegio")
                return {"score": 0.5, "chi2_stat": 0.0, "p_value": 1.0, "zero_pct": 0.0, "compressed_input": False}

            _cleanup = False
            if isinstance(img_source, bytes):
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tmp.write(img_source)
                tmp.close()
                jpeg_path = tmp.name
                _cleanup  = True
            else:
                jpeg_path = img_source

            try:
                jpeg       = jpegio.read(jpeg_path)
                ac_all_raw = jpeg.coef_arrays[0].flatten().astype(np.float64)
            finally:
                if _cleanup:
                    os.unlink(jpeg_path)

            compressed_input = True

        else:
            # Decoded array: use pixel-domain blockiness as fallback
            block_score = _blockiness_score(_to_gray_float(img_source))
            return {"score": round(block_score, 4), "chi2_stat": 0.0, "p_value": 1.0, "zero_pct": 0.0, "compressed_input": False}

        zero_pct = float((ac_all_raw == 0).mean())
        nonzero  = np.abs(ac_all_raw[ac_all_raw != 0])

        if len(nonzero) < 100:
            zero_score = float(np.clip((zero_pct - 0.60) / 0.30, 0.0, 1.0))
            return {"score": round(zero_score, 4), "chi2_stat": 0.0, "p_value": 1.0, "zero_pct": round(zero_pct, 4), "compressed_input": compressed_input}

        log10_vals  = np.floor(np.log10(nonzero + 1e-12)).astype(int)
        first_digit = np.clip((nonzero / np.power(10.0, log10_vals.astype(float))).astype(int), 1, 9)

        observed  = np.array([float(np.sum(first_digit == d)) for d in range(1, 10)])
        expected  = _BENFORD * len(nonzero)
        chi2_stat = float(np.sum((observed - expected) ** 2 / (expected + 1e-10)))
        p_value   = float(1.0 - _scipy_chi2.cdf(chi2_stat, df=8))

        chi2_score = float(np.clip(chi2_stat / 30.0, 0.0, 1.0))
        zero_score = float(np.clip((zero_pct - 0.60) / 0.30, 0.0, 1.0))
        score      = float(np.clip(0.6 * chi2_score + 0.4 * zero_score, 0.0, 1.0))

        return {
            "score":            round(score, 4),
            "chi2_stat":        round(chi2_stat, 4),
            "p_value":          round(p_value, 4),
            "zero_pct":         round(zero_pct, 4),
            "compressed_input": compressed_input,
        }

    except Exception as exc:
        logger.warning("dct_benford_score failed: %s", exc)
        return {"score": 0.5, "chi2_stat": 0.0, "p_value": 1.0, "zero_pct": 0.0, "compressed_input": False}


def _blockiness_score(gray: np.ndarray) -> float:
    """JPEG blocking artifact strength (fallback for decoded arrays).

    Real JPEG scans: strong block boundaries (ratio ≈ 1.5–4.0) → score → 0
    AI / uncompressed: weak boundaries (ratio ≈ 0.8–1.2) → score → 1
    """
    h, w = gray.shape
    g    = gray.astype(np.float64)
    if g.max() > 1.5:
        g = g / 255.0

    boundary_cols = list(range(8, w, 8))
    interior_cols = [c for c in range(1, w - 1) if c % 8 != 0]

    if not boundary_cols or not interior_cols:
        return 0.5

    bdry_diff = float(np.abs(g[:, boundary_cols] - g[:, [c - 1 for c in boundary_cols]]).mean())
    intr_diff = float(np.abs(np.diff(g[:, interior_cols], axis=1)).mean())
    ratio     = bdry_diff / (intr_diff + 1e-10)
    return float(np.clip((1.5 - ratio) / 1.0, 0.0, 1.0))


def _to_gray_float(img_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(img_array, dtype=np.float64)
    if arr.ndim == 3:
        arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2] if arr.shape[2] >= 3 else arr[:, :, 0]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr
