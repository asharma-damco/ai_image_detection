"""
TruFor Pixel-Level Forgery Localization Adapter.

Wraps the TruFor model (github.com/grip-unina/TruFor) into a clean adapter.
TruFor combines NoisePrint++ (camera fingerprint anomaly) with RGB features
in a transformer fusion architecture.

Output
------
    integrity_score  float 0–1  (0=tampered, 1=authentic)
    localization_map np.ndarray (H,W) float32 — pixel forgery probability
    confidence_map   np.ndarray (H,W) float32 — prediction reliability
    detection        str — "authentic" | "uncertain" | "tampered"
    _fallback        bool — True if SRM proxy was used instead of TruFor

Setup (one-time)
----------------
    cd weights/
    git clone https://github.com/grip-unina/TruFor.git
    # Download trufor.pth.tar (~300 MB) per the TruFor README
    # Place at: weights/TruFor/weights/trufor.pth.tar

Source: UAIC uaic-fraud-detection/poc/trufor_adapter.py
"""

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..config import TRUFOR_DIR, TRUFOR_WEIGHTS

try:
    import cv2
    _COLORMAP_TRUFOR = getattr(cv2, "COLORMAP_MAGENTA", cv2.COLORMAP_PINK)
except (ImportError, AttributeError):
    _COLORMAP_TRUFOR = None

# ── Availability check ────────────────────────────────────────────────────────

def is_available() -> tuple[bool, str]:
    if not TRUFOR_DIR.exists():
        return False, (
            f"TruFor repo not found at `{TRUFOR_DIR}`.\n\n"
            "Setup:\n  cd weights/\n  git clone https://github.com/grip-unina/TruFor.git"
        )
    if not TRUFOR_WEIGHTS.exists():
        return False, (
            f"TruFor weights not found at `{TRUFOR_WEIGHTS}`.\n\n"
            "Download trufor.pth.tar from the TruFor repo releases and save to:\n"
            f"  {TRUFOR_WEIGHTS}\n\nFile size: ~300 MB."
        )
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "PyTorch not installed. Run: pip install torch torchvision"
    return True, "ok"


# ── Model cache ───────────────────────────────────────────────────────────────
_trufor_model_cache: dict = {}


def _load_trufor_model(weights_path: Path, device: str):
    cache_key = f"{weights_path}::{device}"
    if cache_key in _trufor_model_cache:
        return _trufor_model_cache[cache_key]

    import torch

    trufor_src = str(TRUFOR_DIR)
    # Verify the clone is the real TruFor package before inserting into sys.path.
    if not (TRUFOR_DIR / "networks").is_dir():
        raise RuntimeError(
            f"TruFor 'networks' directory missing in {TRUFOR_DIR}.\n"
            "The clone may be incomplete. Remove and re-clone:\n"
            "  cd weights/  &&  git clone https://github.com/grip-unina/TruFor.git"
        )
    if trufor_src not in sys.path:
        sys.path.insert(0, trufor_src)

    try:
        from networks.trainer import Trainer  # type: ignore[import]
        import argparse

        opt = argparse.Namespace(
            model="trufor",
            weights=str(weights_path),
            gpu_ids=[],
            isTrain=False,
            phase="test",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = Trainer(opt)
            trainer.eval()

        _trufor_model_cache[cache_key] = trainer
        return trainer

    except Exception as e:
        raise RuntimeError(
            f"Failed to load TruFor model: {e}\n\n"
            "Make sure TruFor is cloned at weights/TruFor/ and requirements are installed:\n"
            "  pip install -r weights/TruFor/requirements.txt"
        ) from e


# ── Mask statistics ───────────────────────────────────────────────────────────

def _compute_mask_stats(loc_map: np.ndarray, H: int, W: int) -> dict:
    """Compute spatial statistics from a TruFor localization map."""
    loc_map    = np.asarray(loc_map, dtype=np.float32)
    binary_mask = loc_map > 0.5
    mask_mean   = float(loc_map.mean())

    p90         = float(np.percentile(loc_map, 90))
    top10_pixels = loc_map[loc_map >= p90]
    mask_top10_mean = float(top10_pixels.mean()) if len(top10_pixels) > 0 else 0.0

    activated = np.argwhere(binary_mask)
    if len(activated) < 10:
        spatial_conc = 0.30
    else:
        rows_norm = activated[:, 0] / max(H, 1)
        cols_norm = activated[:, 1] / max(W, 1)
        spatial_conc = float(np.sqrt(np.var(rows_norm) + np.var(cols_norm)) / np.sqrt(2))

    text_row_start     = int(0.40 * H)
    portrait_col_start = int(0.70 * W)
    text_zone_act      = float(loc_map[text_row_start:, :].mean())
    portrait_zone_act  = float(loc_map[:, portrait_col_start:].mean())

    return {
        "localization_mask":           binary_mask,
        "mask_mean":                   round(mask_mean, 4),
        "mask_top10_mean":             round(mask_top10_mean, 4),
        "mask_spatial_concentration":  round(spatial_conc, 4),
        "text_zone_activation":        round(text_zone_act, 4),
        "portrait_zone_activation":    round(portrait_zone_act, 4),
    }


# ── SRM fallback ──────────────────────────────────────────────────────────────

class _FallbackAnalyzer:
    """Uses SRM residuals to approximate a localization signal when TruFor is unavailable."""

    def analyze(self, pil_image: Image.Image) -> dict:
        try:
            from .srm import SRMAnalyzer
            srm    = SRMAnalyzer(block_size=32)
            srm_r  = srm.compute_anomaly_score(pil_image)
            rmap   = srm_r["residual_map"]
            score  = srm_r["score"]
            integrity = round(1.0 - score, 4)
            detection = "tampered" if score >= 0.60 else "uncertain" if score >= 0.40 else "authentic"

            H, W = rmap.shape
            stats = _compute_mask_stats(rmap, H, W)
            return {
                "integrity_score":             integrity,
                "localization_map":            rmap,
                "confidence_map":              np.ones((H, W), dtype=np.float32) * 0.4,
                "detection":                   detection,
                "_fallback":                   True,
                "_fallback_reason":            "TruFor not available — SRM residual used as proxy",
                "noiseprint_map":              rmap,
                "localization_mask":           stats["localization_mask"],
                "mask_mean":                   stats["mask_mean"],
                "mask_top10_mean":             stats["mask_top10_mean"],
                "mask_spatial_concentration":  stats["mask_spatial_concentration"],
                "text_zone_activation":        stats["text_zone_activation"],
                "portrait_zone_activation":    stats["portrait_zone_activation"],
            }
        except Exception as e:
            H, W  = pil_image.size[1], pil_image.size[0]
            empty = np.zeros((H, W), dtype=np.float32)
            return {
                "integrity_score":             0.5,
                "localization_map":            empty,
                "confidence_map":              empty,
                "detection":                   "uncertain",
                "_fallback":                   True,
                "_fallback_reason":            f"TruFor and SRM both unavailable: {e}",
                "noiseprint_map":              empty,
                "localization_mask":           np.zeros((H, W), dtype=bool),
                "mask_mean":                   0.0,
                "mask_top10_mean":             0.0,
                "mask_spatial_concentration":  0.30,
                "text_zone_activation":        0.0,
                "portrait_zone_activation":    0.0,
            }


# ── Main adapter ──────────────────────────────────────────────────────────────

class TruForAnalyzer:
    """
    Adapter wrapping TruFor pixel-level forgery detection.

    analyze(pil_image) → integrity_score, localization_map, confidence_map, detection
    analyze_roi(pil_image, roi_tuple) → same, localization map embedded in full-image coords
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.weights_path = weights_path or TRUFOR_WEIGHTS
        self.device       = device
        self._model       = None
        self._fallback    = _FallbackAnalyzer()

    def _get_model(self):
        if self._model is None:
            ok, _ = is_available()
            if not ok:
                return None
            try:
                self._model = _load_trufor_model(self.weights_path, self.device)
            except Exception:
                self._model = None
        return self._model

    def analyze(self, pil_image: Image.Image) -> dict:
        model = self._get_model()
        if model is None:
            return self._fallback.analyze(pil_image)
        return self._run_trufor(model, pil_image)

    def analyze_roi(self, pil_image: Image.Image, roi: tuple) -> dict:
        """Run TruFor on a crop; embed result back into full-image coordinates."""
        x, y, w, h = roi
        iw, ih = pil_image.size
        x  = max(0, int(x));  y  = max(0, int(y))
        x2 = min(iw, x + int(w));  y2 = min(ih, y + int(h))
        w  = x2 - x;  h = y2 - y

        if w < 32 or h < 32:
            result = self.analyze(pil_image)
            result["_roi_note"] = "ROI too small (<32px) — ran on full image"
            return result

        crop   = pil_image.crop((x, y, x2, y2))
        result = self.analyze(crop)

        import cv2
        full_loc  = np.zeros((ih, iw), dtype=np.float32)
        full_conf = np.zeros((ih, iw), dtype=np.float32)

        loc_map  = result["localization_map"]
        conf_map = result["confidence_map"]

        if loc_map.shape  != (h, w):
            loc_map  = cv2.resize(loc_map,  (w, h), interpolation=cv2.INTER_LINEAR)
        if conf_map.shape != (h, w):
            conf_map = cv2.resize(conf_map, (w, h), interpolation=cv2.INTER_LINEAR)

        full_loc[y:y2, x:x2]  = loc_map
        full_conf[y:y2, x:x2] = conf_map

        result["localization_map"] = full_loc
        result["confidence_map"]   = full_conf
        result["_roi"]             = {"x": x, "y": y, "w": w, "h": h}
        return result

    def _run_trufor(self, model, pil_image: Image.Image) -> dict:
        import torch
        import cv2

        try:
            img_rgb = np.array(pil_image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            H, W    = img_bgr.shape[:2]

            img_tensor = torch.from_numpy(
                img_bgr.transpose(2, 0, 1).astype(np.float32)
            ).unsqueeze(0) / 255.0

            _np_activation: dict = {}

            def _hook_fn(module, inp, out):
                _np_activation["map"] = out.detach().cpu()

            _hook = None
            _NP_KEYWORDS = ("noiseprint", "noisep", "np_enc", "noise_enc", "np_branch")
            for _name, _mod in model.named_modules():
                if any(kw in _name.lower() for kw in _NP_KEYWORDS):
                    _hook = _mod.register_forward_hook(_hook_fn)
                    break

            with torch.no_grad():
                try:
                    output = model.test(img_tensor)
                except AttributeError:
                    output = model(img_tensor)

            if _hook is not None:
                _hook.remove()

            if isinstance(output, dict):
                integrity  = float(output.get("score", 0.5))
                loc_map_t  = output.get("map",  output.get("localization", None))
                conf_map_t = output.get("conf", output.get("confidence",   None))
            elif isinstance(output, (tuple, list)) and len(output) >= 2:
                integrity  = float(output[0]) if not hasattr(output[0], "shape") else float(output[0].mean())
                loc_map_t  = output[1] if len(output) > 1 else None
                conf_map_t = output[2] if len(output) > 2 else None
            else:
                integrity  = float(output) if not hasattr(output, "shape") else float(output.mean())
                loc_map_t  = conf_map_t = None

            def _to_map(t, H, W):
                if t is None:
                    return np.full((H, W), 0.5, dtype=np.float32)
                if hasattr(t, "cpu"):
                    t = t.cpu().numpy()
                t = np.squeeze(t)
                if t.ndim == 0:
                    return np.full((H, W), float(t), dtype=np.float32)
                t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
                if t_norm.shape != (H, W):
                    t_norm = cv2.resize(t_norm.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                return t_norm.astype(np.float32)

            loc_map  = _to_map(loc_map_t,  H, W)
            conf_map = _to_map(conf_map_t, H, W)

            np_raw        = _np_activation.get("map")
            noiseprint_map = _to_map(np_raw, H, W) if np_raw is not None else loc_map

            integrity = float(np.clip(integrity, 0.0, 1.0))
            detection = "tampered" if integrity < 0.40 else "uncertain" if integrity < 0.65 else "authentic"

            stats = _compute_mask_stats(loc_map, H, W)

            return {
                "integrity_score":             round(integrity, 4),
                "localization_map":            loc_map,
                "confidence_map":              conf_map,
                "detection":                   detection,
                "_fallback":                   False,
                "noiseprint_map":              noiseprint_map,
                "localization_mask":           stats["localization_mask"],
                "mask_mean":                   stats["mask_mean"],
                "mask_top10_mean":             stats["mask_top10_mean"],
                "mask_spatial_concentration":  stats["mask_spatial_concentration"],
                "text_zone_activation":        stats["text_zone_activation"],
                "portrait_zone_activation":    stats["portrait_zone_activation"],
            }

        except Exception as e:
            result = self._fallback.analyze(pil_image)
            result["_trufor_error"] = str(e)
            return result


# ── Display helpers ───────────────────────────────────────────────────────────

def localization_map_to_heatmap(loc_map: np.ndarray, colormap=None) -> np.ndarray:
    """Convert float32 localization map [0,1] to BGR uint8 heatmap."""
    import cv2
    cmap     = colormap if colormap is not None else (_COLORMAP_TRUFOR or cv2.COLORMAP_PINK)
    uint8map = (loc_map * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(uint8map, cmap)


def overlay_localization_on_image(
    pil_image: Image.Image,
    loc_map: np.ndarray,
    alpha: float = 0.50,
    colormap=None,
) -> Image.Image:
    """Alpha-blend TruFor localization heatmap onto the original image."""
    import cv2
    orig_rgb    = np.array(pil_image.convert("RGB"))
    heatmap_bgr = localization_map_to_heatmap(loc_map, colormap=colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    if heatmap_rgb.shape[:2] != orig_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended = cv2.addWeighted(orig_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return Image.fromarray(blended.astype(np.uint8))
