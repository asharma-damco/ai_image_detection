"""
DualBranchModel — EfficientNet-B0 (RGB) + DCT CNN detector.

Architecture
------------
  RGB Branch  : EfficientNet-B0 (ImageNet pretrained) → 1280-d features
  DCT Branch  : 3-layer CNN (1→16→32→64 channels)    →  256-d features
  Fusion      : Learnable gating (soft weights)       →  512-d fused
  Classifier  : FC 512→128→2 logits  (0=Fake, 1=Real)

Inference modes
---------------
  predict()         — full-image (224×224)
  predict_roi()     — ROI crop then full-image inference
  predict_patches() — sliding-window top-k aggregation

Sources: UAIC uaic-fraud-detection/src/backend/app/ml/model.py + inference.py
         PIMA onboarding_poc/cu_poc/ai_fraud_detector.py (unified)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from ..config import (
    DUAL_BRANCH_WEIGHTS,
    FULL_IMAGE_THRESHOLD,
    PATCH_SIZE,
    PATCH_STRIDE,
    PATCH_THRESHOLD,
    PATCH_TOP_K,
)
from ..preprocessing.image import preprocess_image
from .base import BaseDetector


# ── Model components ──────────────────────────────────────────────────────────

class _EfficientNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)  # [B, 1280]


class _DCTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # [B, 256]


class _FeatureFusion(nn.Module):
    def __init__(self, dim_rgb: int = 1280, dim_dct: int = 256):
        super().__init__()
        self.fc_rgb = nn.Linear(dim_rgb, 512)
        self.fc_dct = nn.Linear(dim_dct, 512)
        self.gate = nn.Sequential(
            nn.Linear(512 * 2, 512), nn.ReLU(),
            nn.Linear(512, 2), nn.Softmax(dim=1),
        )

    def forward(self, rgb_feat: torch.Tensor, dct_feat: torch.Tensor) -> torch.Tensor:
        rgb_proj = self.fc_rgb(rgb_feat)
        dct_proj = self.fc_dct(dct_feat)
        combined = torch.cat([rgb_proj, dct_proj], dim=1)
        weights  = self.gate(combined)                          # [B, 2]
        return weights[:, 0:1] * rgb_proj + weights[:, 1:2] * dct_proj  # [B, 512]


class DualBranchModel(nn.Module):
    """Raw PyTorch model — use DualBranchDetector for the full inference API."""

    def __init__(self):
        super().__init__()
        self.rgb_encoder = _EfficientNetEncoder()
        self.dct_encoder = _DCTEncoder()
        self.fusion      = _FeatureFusion()
        self.classifier  = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2),
        )

    def forward(self, rgb: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_encoder(rgb)
        dct_feat = self.dct_encoder(dct)
        fused    = self.fusion(rgb_feat, dct_feat)
        return self.classifier(fused)  # [B, 2] logits


def load_model(
    weights_path: Path | str = DUAL_BRANCH_WEIGHTS,
    device: Optional[torch.device] = None,
) -> DualBranchModel:
    """Load DualBranchModel from a checkpoint or plain state_dict file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model      = DualBranchModel()
    checkpoint = torch.load(str(weights_path), map_location=device)
    state      = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── High-level detector ────────────────────────────────────────────────────────

class DualBranchDetector(BaseDetector):
    """
    Full-featured wrapper around DualBranchModel.

    Supports full-image, ROI-crop, and sliding-window patch inference.
    Model is lazy-loaded on first call to predict().
    """

    def __init__(
        self,
        weights_path: Path | str = DUAL_BRANCH_WEIGHTS,
        threshold: float = FULL_IMAGE_THRESHOLD,
        device: Optional[torch.device] = None,
    ):
        self.weights_path = Path(weights_path)
        self.threshold    = threshold
        self.device       = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: Optional[DualBranchModel] = None

    def _get_model(self) -> DualBranchModel:
        if self._model is None:
            self._model = load_model(self.weights_path, self.device)
        return self._model

    # ── BaseDetector interface ─────────────────────────────────────────────────

    def predict(self, img: Image.Image, threshold: Optional[float] = None) -> dict:
        """Full-image inference.

        Returns:
            label         "Real" | "Fake"
            confidence    float — probability of predicted class
            probabilities {"Real": float, "Fake": float}
            score         float — fake probability (for ensemble consumption)
        """
        thr   = threshold if threshold is not None else self.threshold
        model = self._get_model()

        rgb_t, dct_t = preprocess_image(img)
        rgb_t = rgb_t.to(self.device)
        dct_t = dct_t.to(self.device)

        with torch.no_grad():
            logits = model(rgb_t, dct_t)
            probs  = F.softmax(logits, dim=1)[0]

        fake_prob = probs[0].item()
        real_prob = probs[1].item()

        if fake_prob > thr:
            label      = "Fake"
            confidence = fake_prob
        else:
            label      = "Real"
            confidence = real_prob

        return {
            "label":         label,
            "confidence":    round(confidence, 4),
            "probabilities": {"Real": round(real_prob, 4), "Fake": round(fake_prob, 4)},
            "score":         round(fake_prob, 4),
        }

    def predict_patches(
        self,
        img: Image.Image,
        threshold: float = PATCH_THRESHOLD,
        patch_size: int  = PATCH_SIZE,
        stride: int      = PATCH_STRIDE,
        top_k: int       = PATCH_TOP_K,
    ) -> dict:
        """Sliding-window patch prediction with top-k aggregation.

        More robust for large images — aggregates local predictions
        rather than relying on a single global crop.

        Returns:
            label        "Real" | "Fake"
            confidence   float
            top_k_score  float — mean fake_prob of top-k patches
            patch_count  int
        """
        model = self._get_model()
        img_resized = img.resize((512, 512))
        W, H        = img_resized.size

        # Collect all patches first, then run a single batched forward pass.
        rgb_list: list[torch.Tensor] = []
        dct_list: list[torch.Tensor] = []
        for i in range(0, W - patch_size + 1, stride):
            for j in range(0, H - patch_size + 1, stride):
                patch = img_resized.crop((i, j, i + patch_size, j + patch_size))
                rgb_t, dct_t = preprocess_image(patch)
                rgb_list.append(rgb_t)
                dct_list.append(dct_t)

        if not rgb_list:
            return {"label": "Real", "confidence": 0.5,
                    "probabilities": {"Real": 0.5, "Fake": 0.5},
                    "score": 0.5, "top_k_score": 0.5, "patch_count": 0}

        rgb_batch = torch.cat(rgb_list, dim=0).to(self.device)
        dct_batch = torch.cat(dct_list, dim=0).to(self.device)
        fake_scores: list[float] = []

        model.eval()
        with torch.no_grad():
            logits = model(rgb_batch, dct_batch)
            probs  = F.softmax(logits, dim=1)
            fake_scores = probs[:, 0].tolist()

        fake_scores_sorted = sorted(fake_scores, reverse=True)
        k           = min(top_k, len(fake_scores_sorted))
        top_k_score = float(np.mean(fake_scores_sorted[:k]))
        real_score  = 1.0 - top_k_score
        label       = "Fake" if top_k_score > threshold else "Real"
        confidence  = top_k_score if label == "Fake" else real_score

        return {
            "label":         label,
            "confidence":    round(confidence, 4),
            "probabilities": {"Real": round(real_score, 4), "Fake": round(top_k_score, 4)},
            "score":         round(top_k_score, 4),
            "top_k_score":   round(top_k_score, 4),
            "patch_count":   len(fake_scores),
        }
