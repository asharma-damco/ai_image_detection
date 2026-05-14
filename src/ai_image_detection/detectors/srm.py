"""
SRM Filter Bank Forensic Signal.

Implements the Stochastic Residual Map (SRM) forensic signal using 5 fixed
analytical kernels (Fridrich & Kodovsky 2012). Requires no GPU and no training.

Key insight: AI inpainting / diffusion editing alters local noise residual
statistics. Authentic photos have spatially consistent residuals from the
camera sensor pipeline. Edited regions show discontinuities at the edit boundary.

Source: UAIC uaic-fraud-detection/poc/srm_adapter.py
"""

import numpy as np
import cv2
from PIL import Image

from .base import BaseDetector


# ── SRM kernels (Fridrich & Kodovsky 2012, Table 1 subset) ────────────────────
SRM_KERNELS: dict[str, np.ndarray] = {
    "hp3": np.array([
        [-1,  2, -1], [ 2, -4,  2], [-1,  2, -1],
    ], dtype=np.float32) / 4.0,

    "hp5v": np.array([
        [ 0,  0,  0,  0,  0], [ 0,  0,  0,  0,  0],
        [-1,  2, -2,  2, -1], [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=np.float32) / 4.0,

    "hp5h": np.array([
        [ 0,  0, -1,  0,  0], [ 0,  0,  2,  0,  0],
        [ 0,  0, -2,  0,  0], [ 0,  0,  2,  0,  0],
        [ 0,  0, -1,  0,  0],
    ], dtype=np.float32) / 4.0,

    "sq3": np.array([
        [-1,  0,  0,  0,  1], [ 0, -1,  0,  1,  0],
        [ 0,  0,  0,  0,  0], [ 0,  1,  0, -1,  0],
        [ 1,  0,  0,  0, -1],
    ], dtype=np.float32) / 4.0,

    "edge5": np.array([
        [-1, -1, -1, -1, -1], [-1,  1,  1,  1, -1],
        [-1,  1,  8,  1, -1], [-1,  1,  1,  1, -1],
        [-1, -1, -1, -1, -1],
    ], dtype=np.float32) / 8.0,
}


def is_available() -> tuple[bool, str]:
    try:
        import numpy  # noqa: F401
        import cv2    # noqa: F401
        return True, "ok"
    except ImportError as e:
        return False, f"Missing dependency: {e}"


class SRMAnalyzer(BaseDetector):
    """
    SRM filter bank forensic analysis for AI-edited image detection.

    Steps:
      1. Convert to grayscale
      2. Apply each SRM kernel via 2D convolution
      3. Compute local block variance of residual maps
      4. High variance = noise inconsistency = manipulation signature
      5. Aggregate across kernels → single anomaly score [0, 1]
    """

    def __init__(self, block_size: int = 16):
        """
        Args:
            block_size: Size of local variance blocks. 16 is a good default
                        for balanced spatial resolution vs. score stability.
        """
        self.kernels    = SRM_KERNELS
        self.block_size = block_size

    # ── BaseDetector interface ─────────────────────────────────────────────────

    def predict(self, img: Image.Image, **kwargs) -> dict:
        """Convenience wrapper that maps compute_anomaly_score to the BaseDetector API."""
        result = self.compute_anomaly_score(img)
        score  = result["score"]
        label  = "Fake" if score >= 0.60 else "Real"
        return {
            "label":      label,
            "confidence": round(abs(score - 0.5) * 2, 4),
            "score":      score,
            **result,
        }

    # ── Core signal API ────────────────────────────────────────────────────────

    def extract_residuals(self, pil_image: Image.Image) -> np.ndarray:
        """Apply all SRM kernels; return stacked residual maps (H, W, N_kernels)."""
        gray = np.array(pil_image.convert("L"), dtype=np.float32) / 255.0
        residuals = [
            cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT)
            for kernel in self.kernels.values()
        ]
        return np.stack(residuals, axis=-1)

    def compute_anomaly_score(self, pil_image: Image.Image) -> dict:
        """
        Compute the SRM anomaly score.

        Returns:
            score        float [0–1]  higher = more anomalous / likely manipulated
            residual_map np.ndarray (H,W) float32 — combined heatmap for display
            detail       dict — per-kernel stats
        """
        residuals = self.extract_residuals(pil_image)
        H, W, N   = residuals.shape
        bs        = self.block_size

        if H < bs or W < bs:
            # Image too small for block-variance analysis; return neutral score.
            return {"score": 0.5, "residual_map": np.zeros((H, W), dtype=np.float32), "detail": {}}

        detail: dict       = {}
        kernel_scores: list[float] = []
        combined_map = np.zeros((H, W), dtype=np.float32)

        for ki, name in enumerate(self.kernels):
            rmap    = residuals[:, :, ki]
            var_map = self._local_variance_map(rmap, bs)
            p95_var = float(np.percentile(var_map, 95))
            raw_score = min(p95_var / 0.010, 1.0)
            kernel_scores.append(raw_score)
            detail[name] = {
                "mean_var":     round(float(np.mean(var_map)), 6),
                "p95_var":      round(p95_var, 6),
                "contribution": round(raw_score, 4),
            }
            up_h = (H // bs) * bs;  up_w = (W // bs) * bs
            var_cropped = var_map[:up_h // bs, :up_w // bs]
            var_full    = np.kron(var_cropped, np.ones((bs, bs), dtype=np.float32))
            pad_h = H - var_full.shape[0];  pad_w = W - var_full.shape[1]
            if pad_h > 0 or pad_w > 0:
                var_full = np.pad(var_full, ((0, pad_h), (0, pad_w)), mode="edge")
            combined_map += var_full

        combined_map /= N
        map_max = combined_map.max()
        if map_max > 0:
            combined_map = combined_map / map_max

        weights   = {"hp3": 1.0, "hp5v": 1.0, "hp5h": 1.0, "sq3": 0.8, "edge5": 1.2}
        w_total   = sum(weights[n] for n in self.kernels)
        score     = sum(weights[n] * s for n, s in zip(self.kernels, kernel_scores)) / w_total

        return {
            "score":        round(float(score), 4),
            "residual_map": combined_map,
            "detail":       detail,
        }

    def detect_jpeg_grid_inconsistency(self, pil_image: Image.Image) -> dict:
        """
        Detect JPEG 8×8 DCT block grid inconsistencies.

        AI editing tools re-save only modified regions, introducing a boundary
        where the 8×8 JPEG quantization grid alignment changes.

        Returns:
            score   float [0–1]  higher = more inconsistency = likely edited
            heatmap np.ndarray (H,W) float32 [0,1] — block-level inconsistency map
        """
        ycbcr     = pil_image.convert("YCbCr")
        y_channel = np.array(ycbcr)[:, :, 0].astype(np.float32)
        H, W      = y_channel.shape
        block_size = 8
        blocks_h   = H // block_size;  blocks_w = W // block_size

        if blocks_h < 2 or blocks_w < 2:
            return {"score": 0.0, "heatmap": np.zeros((H, W), dtype=np.float32)}

        hf_energy = np.zeros((blocks_h, blocks_w), dtype=np.float32)
        for bi in range(blocks_h):
            for bj in range(blocks_w):
                block = y_channel[bi * block_size:(bi+1)*block_size, bj*block_size:(bj+1)*block_size]
                dct_block = cv2.dct(block)
                hf_energy[bi, bj] = float(np.mean(np.abs(dct_block[4:, 4:])))

        inconsistency_map = np.zeros_like(hf_energy)
        for bi in range(blocks_h):
            for bj in range(blocks_w):
                r0, r1 = max(0, bi-1), min(blocks_h, bi+2)
                c0, c1 = max(0, bj-1), min(blocks_w, bj+2)
                inconsistency_map[bi, bj] = float(np.var(hf_energy[r0:r1, c0:c1]))

        p95   = float(np.percentile(inconsistency_map, 95))
        score = min(p95 / 15.0, 1.0)

        norm_map = inconsistency_map / (inconsistency_map.max() + 1e-8)
        heatmap  = np.kron(norm_map.astype(np.float32), np.ones((block_size, block_size), dtype=np.float32))
        pad_h    = H - heatmap.shape[0];  pad_w = W - heatmap.shape[1]
        if pad_h > 0 or pad_w > 0:
            heatmap = np.pad(heatmap, ((0, pad_h), (0, pad_w)), mode="edge")

        return {"score": round(float(score), 4), "heatmap": heatmap[:H, :W]}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _local_variance_map(self, arr: np.ndarray, block_size: int) -> np.ndarray:
        H, W   = arr.shape
        rows   = H // block_size;  cols = W // block_size
        result = np.zeros((rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                block = arr[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                result[i, j] = float(np.var(block))
        return result


# ── Display helpers ───────────────────────────────────────────────────────────

def residual_map_to_heatmap(residual_map: np.ndarray, colormap: int = cv2.COLORMAP_HOT) -> np.ndarray:
    """Convert normalised float32 residual map [0,1] to BGR uint8 heatmap."""
    uint8_map = (residual_map * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(uint8_map, colormap)


def overlay_heatmap_on_image(pil_image: Image.Image, heatmap_bgr: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Alpha-blend a BGR heatmap onto a PIL image."""
    orig_rgb    = np.array(pil_image.convert("RGB"))
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    if heatmap_rgb.shape[:2] != orig_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended = cv2.addWeighted(orig_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return Image.fromarray(blended.astype(np.uint8))
