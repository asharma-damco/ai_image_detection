"""
Central configuration for the ai_image_detection framework.

All thresholds, weight presets, and path defaults live here.
Override DUAL_BRANCH_WEIGHTS via the DUAL_BRANCH_WEIGHTS_PATH env variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Package root / weights directory ──────────────────────────────────────────
# Path: src/ai_image_detection → src → project root
_PKG_ROOT   = Path(__file__).parent.parent.parent
WEIGHTS_DIR = _PKG_ROOT / "weights"

# Default weights base — use env vars to override (see scripts/download_weights.py)
_UAIC_POC = WEIGHTS_DIR

# Default checkpoint — override with DUAL_BRANCH_WEIGHTS_PATH env var
DUAL_BRANCH_WEIGHTS: Path = Path(
    os.environ.get(
        "DUAL_BRANCH_WEIGHTS_PATH",
        str(_UAIC_POC / "dtc_rgb_model_v1.pth"),
    )
)

# TruFor repo / weights
TRUFOR_DIR:     Path = Path(os.environ.get("TRUFOR_DIR",     str(_UAIC_POC / "TruFor")))
TRUFOR_WEIGHTS: Path = TRUFOR_DIR / "weights" / "trufor.pth.tar"

# UniversalFakeDetect repo
UFD_REPO:    Path = Path(os.environ.get("UFD_REPO",    str(_UAIC_POC / "UniversalFakeDetect")))
UFD_WEIGHTS: Path = UFD_REPO / "pretrained_weights" / "fc_weights.pth"

# YOLO damage model
YOLO_MODEL_DIR:      Path = Path(os.environ.get("YOLO_MODEL_DIR", str(_UAIC_POC / "yolo_damage_detect")))
YOLO_CACHED_WEIGHTS: Path = YOLO_MODEL_DIR / "trained.pt"
YOLO_GITHUB_URL = (
    "https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector"
    "/raw/main/trained.pt"
)

# ── Verdict thresholds ─────────────────────────────────────────────────────────
THRESH_AUTHENTIC  = 0.35
THRESH_SUSPICIOUS = 0.60  # >= this → "Likely Fraudulent"

# ── Confidence spread thresholds ───────────────────────────────────────────────
SPREAD_HIGH   = 0.20  # signals spread < 0.20 → HIGH confidence
SPREAD_MEDIUM = 0.40  # signals spread < 0.40 → MEDIUM confidence

# ── DualBranchModel inference thresholds ───────────────────────────────────────
FULL_IMAGE_THRESHOLD = 0.10   # fake_prob > threshold → FAKE (sensitive)
PATCH_THRESHOLD      = 0.50
PATCH_SIZE           = 224
PATCH_STRIDE         = 112
PATCH_TOP_K          = 5

# ── Ensemble weight presets ────────────────────────────────────────────────────

# Vehicle damage pipeline
# yolo_damage intentionally excluded — it drives ROI only, not fraud scoring.
# Redistributed its 0.10 weight proportionally across fraud signals.
VEHICLE_DAMAGE_WEIGHTS: dict[str, float] = {
    "trufor":      0.35,
    "dual_branch": 0.28,
    "clip_ufd":    0.22,
    "rigid":       0.10,
    "srm":         0.05,
}                          # sum = 1.00

# Document fraud pipeline
DOCUMENT_FRAUD_WEIGHTS: dict[str, float] = {
    "trufor":   0.25,   # was 0.30
    "srm":      0.20,   # unchanged
    "siglip2":  0.23,   # +0.03 from freed weak signals
    "dire":     0.18,   # +0.03 from freed weak signals
    "rigid":    0.10,   # unchanged
    # demoted — weak signal on diffusion-era generators
    "ela":      0.02,   # was 0.05
    # demoted — weak signal on diffusion-era generators
    "prnu":     0.02,   # was 0.05
}                       # sum = 1.00

# ID card pipeline
ID_CARD_WEIGHTS: dict[str, float] = {
    "trufor":      0.25,   # was 0.30
    "srm":         0.20,   # unchanged
    "dire":        0.19,   # +0.04 from freed weak signals
    "dual_branch": 0.15,   # was 0.20
    "rigid":       0.14,   # +0.04 from freed weak signals
    # demoted — weak signal on diffusion-era generators
    "ela":         0.02,   # was 0.05
    # demoted — weak signal on diffusion-era generators
    "prnu":        0.02,   # was 0.05
    # demoted — weak signal on diffusion-era generators
    "dct_benford": 0.01,   # was 0.03
    "cfa":         0.02,   # unchanged
}                          # sum = 1.00

# ── Signal human-readable labels ───────────────────────────────────────────────
SIGNAL_LABELS: dict[str, str] = {
    "dual_branch":                "DualBranchModel (RGB+DCT)",
    "clip_ufd":                   "CLIP / UniversalFakeDetect",
    "trufor":                     "TruFor (pixel-level forgery)",
    "srm":                        "SRM Filter Bank",
    "siglip2":                    "SigLIP-2 So400m",
    "prnu":                       "PRNU Noise Floor",
    "ela":                        "Error Level Analysis (ELA)",
    "dct_benford":                "DCT / Benford's Law",
    "cfa":                        "CFA Demosaicing Correlation",
    "dinov2":                     "DINOv2 ViT-L/14 Feature Anomaly",
    "noiseprint_localizer":       "NoisePrint++ Tamper Localizer",
    "mask_spatial_concentration": "TruFor Spatial Concentration",
    "metadata":                   "EXIF Metadata Forensics",
}

# ── Inference constants ────────────────────────────────────────────────────────
# Centralised here so detectors don't carry magic numbers in their source.

IMAGE_MIN_SIZE        = 64     # minimum input dimension for any pipeline
DINO_INPUT_SIZE       = 518    # DINOv2 ViT-L/14 preferred input (37×37 patches)
DINO_FALLBACK_SIZE    = 224    # fallback if 518 causes position-embed error
NOISE_STD             = 0.05   # Gaussian noise std in normalised pixel space (RIGID)
RIGID_N_PERTURBATIONS = 3      # perturbation count for RIGID (was 10 — CPU too slow)
DIRE_INVERSION_STEPS  = 10     # DDIM inversion steps for DIRE (was 20 — halves latency)
SRM_FAKE_THRESHOLD    = 0.60   # SRM score >= this → "Fake"
SRM_VAR_NORM          = 0.010  # p95 variance normalisation factor for SRM kernels
