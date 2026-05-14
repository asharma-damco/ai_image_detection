"""
Explainability visualisation utilities for DualBranchModel.

Two visualisations:

1. patch_verdict_grid()
   Divides the image into overlapping patches, runs inference on each,
   and paints cells RED (fake) or GREEN (real) with opacity proportional
   to confidence. Shows WHERE the model suspects manipulation.

2. dct_artifact_map()
   Renders the high-frequency DCT map extracted from the image.
   AI-generated images show unnaturally smooth, regular frequency patterns.
   Real photos show organic noise patterns.

Source: UAIC uaic-fraud-detection/src/backend/app/ml/explainability.py
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ..detectors.dual_branch import DualBranchModel
from ..preprocessing.dct import extract_dct_high_freq
from ..preprocessing.image import preprocess_image

PATCH_FLAG_THRESHOLD = 0.50


def patch_verdict_grid(
    model: DualBranchModel,
    img: Image.Image,
    device: torch.device,
    overall_label: str,
    patch_size: int = 112,
    stride: int = 56,
    alpha: float = 0.45,
) -> tuple[Image.Image, dict]:
    """Run per-patch inference and overlay a colour-coded verdict grid.

    Overall = FAKE → RED for suspicious patches (fake_prob > 0.5).
    Overall = REAL → GREEN tint proportional to real confidence.

    Returns:
        (overlaid_image, stats_dict)
    """
    W, H    = img.size
    base    = img.convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    patch_results: list[dict] = []

    model.eval()
    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch  = img.crop((x, y, x + patch_size, y + patch_size))
                rgb_t, dct_t = preprocess_image(patch)
                logits = model(rgb_t.to(device), dct_t.to(device))
                probs  = F.softmax(logits, dim=1)[0]
                fake_p = float(probs[0]);  real_p = float(probs[1])
                patch_results.append({"x": x, "y": y, "fake_prob": round(fake_p, 3), "real_prob": round(real_p, 3)})

                if overall_label == "Fake":
                    colour = (220, 38, 38, int(alpha * fake_p * 255)) if fake_p > PATCH_FLAG_THRESHOLD else (0, 0, 0, 0)
                else:
                    colour = (22, 163, 74, int(alpha * 0.6 * real_p * 255))

                draw.rectangle([x, y, x + patch_size, y + patch_size], fill=colour, outline=(255, 255, 255, 30), width=1)

    result     = Image.alpha_composite(base, overlay).convert("RGB")
    fake_probs = [p["fake_prob"] for p in patch_results]
    flagged    = [p for p in patch_results if p["fake_prob"] > PATCH_FLAG_THRESHOLD]

    stats = {
        "total_patches":   len(patch_results),
        "flagged_patches": len(flagged),
        "clean_patches":   len(patch_results) - len(flagged),
        "mean_fake_prob":  round(float(np.mean(fake_probs)), 3) if fake_probs else 0.0,
        "max_fake_prob":   round(float(np.max(fake_probs)),  3) if fake_probs else 0.0,
        "overall_label":   overall_label,
    }
    return result, stats


def dct_artifact_map(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Produce grayscale and coloured visualisations of the DCT high-frequency map.

    AI images: unnaturally smooth / regular DCT maps.
    Real photos: irregular, noisy DCT maps.

    Returns:
        (gray_pil, color_pil) — both resized to match original image dimensions.
    """
    W, H = img.size

    dct_array = extract_dct_high_freq(img)   # (1, 224, 224)
    dct_2d    = dct_array[0]                  # (224, 224)

    dct_norm  = (dct_2d - dct_2d.min()) / (dct_2d.max() - dct_2d.min() + 1e-8)
    dct_uint8 = np.uint8(dct_norm * 255)
    dct_resized = cv2.resize(dct_uint8, (W, H), interpolation=cv2.INTER_LINEAR)

    gray_pil  = Image.fromarray(dct_resized, mode="L").convert("RGB")
    color_bgr = cv2.applyColorMap(dct_resized, cv2.COLORMAP_INFERNO)
    color_pil = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))

    return gray_pil, color_pil


def build_evidence_panel(
    model: DualBranchModel,
    img: Image.Image,
    device: torch.device,
    label: str,
    roi: dict | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, dict]:
    """Convenience wrapper returning (original, patch_grid, dct_color, stats).

    If roi is supplied, panels are computed on the crop then pasted back.
    """
    target       = img
    paste_coords = None

    if roi:
        x, y, w, h  = roi["x"], roi["y"], roi["width"], roi["height"]
        target       = img.crop((x, y, x + w, y + h))
        paste_coords = (x, y)

    grid, stats = patch_verdict_grid(model, target, device, overall_label=label)
    _, dct_color = dct_artifact_map(target)

    if paste_coords:
        full_grid = img.copy().convert("RGB");  full_grid.paste(grid, paste_coords)
        full_dct  = img.copy().convert("RGB");  full_dct.paste(dct_color, paste_coords)
        return img, full_grid, full_dct, stats

    return img, grid, dct_color, stats
