"""RIGID — DINOv2 Perturbation Sensitivity detector.

Reference: arXiv 2411.19117 — "RIGID: Training-free and Model-agnostic
Image Forgery Detection".

Core idea: real image patch embeddings shift MORE under Gaussian pixel-space
perturbation than AI-generated ones, because generated images lie closer to
the model's learned manifold (lower local Lipschitz constant).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import DINO_FALLBACK_SIZE, DINO_INPUT_SIZE, NOISE_STD, RIGID_N_PERTURBATIONS
from .base import BaseDetector
from .dino_probe import _DINO_MEAN, _DINO_STD, _load_dino_model

logger = logging.getLogger(__name__)

_RIGID_SIZE    = DINO_INPUT_SIZE     # 518 — 37×37 patches at ViT-L/14 patch size 14
_FALLBACK_SIZE = DINO_FALLBACK_SIZE  # 224 — fallback if 518 causes position-embed error
_THRESHOLD     = 0.12
_N_PERTURB     = RIGID_N_PERTURBATIONS   # 3 (was 10 — 30–60 s per pass on CPU)
_NOISE_STD     = NOISE_STD              # 0.05 in normalised [0,1] pixel space


class RIGIDDetector(BaseDetector):
    """Training-free AI image detector via DINOv2 patch perturbation sensitivity.

    Reuses the module-level DINOv2 singleton from dino_probe — no second load.
    High sensitivity score → likely real; low score → likely AI-generated.
    """

    def __init__(
        self,
        n_perturbations: int = _N_PERTURB,
        noise_std: float = _NOISE_STD,
        device: str | None = None,
    ) -> None:
        self.n_perturbations = n_perturbations
        self.noise_std = noise_std
        self._device_override = device
        self._model = None
        self._device = None

    def _get_model(self):
        """Return (model, device), loading via dino_probe singleton."""
        if self._model is not None:
            return self._model, self._device
        model, device = _load_dino_model()
        if model is None:
            return None, None
        self._model = model
        self._device = self._device_override or device
        return self._model, self._device

    def _to_tensor(self, img: Image.Image, size: int, device: str) -> torch.Tensor:
        """Resize, normalise, and send to device. Returns (1, 3, size, size)."""
        img_r = img.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.array(img_r, dtype=np.float32) / 255.0
        mean = np.array(_DINO_MEAN, dtype=np.float32)
        std  = np.array(_DINO_STD,  dtype=np.float32)
        arr  = (arr - mean) / std
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return t.to(device)

    def _forward_patches(self, model, tensor: torch.Tensor) -> torch.Tensor:
        """Run DINOv2 forward and return patch tokens [num_patches, embed_dim]."""
        with torch.no_grad():
            out = model.forward_features(tensor)
        if isinstance(out, dict):
            patches = out.get("x_norm_patchtokens")
        else:
            patches = out[:, 1:, :]   # strip CLS
        if patches is None:
            raise RuntimeError("forward_features returned no patch tokens")
        return patches.squeeze(0)   # [num_patches, embed_dim]

    def _extract_patch_features(self, img: Image.Image) -> torch.Tensor:
        """Extract DINOv2 patch embeddings at 518×518.

        Returns:
            Tensor [num_patches, embed_dim] on the model's device.
        """
        model, device = self._get_model()
        if model is None:
            raise RuntimeError("DINOv2 unavailable")

        try:
            tensor = self._to_tensor(img, _RIGID_SIZE, device)
            return self._forward_patches(model, tensor)
        except Exception:
            # Position-embed interpolation may fail on very old torch builds
            tensor = self._to_tensor(img, _FALLBACK_SIZE, device)
            return self._forward_patches(model, tensor)

    def _perturbation_sensitivity(self, img: Image.Image) -> float:
        """Mean cosine distance between original and Gaussian-perturbed patch embeddings.

        Gaussian noise (std=noise_std) is added in normalised pixel space before
        DINOv2 normalization. Averaged over n_perturbations and all patches.

        Note: 10 forward passes at 518×518 with ViT-L/14 takes ~30–60 s on CPU.
        """
        model, device = self._get_model()
        if model is None:
            raise RuntimeError("DINOv2 unavailable")

        # Probe input size — fall back to 224 if 518 fails
        try:
            size = _RIGID_SIZE
            orig_tensor = self._to_tensor(img, size, device)
            self._forward_patches(model, orig_tensor)
        except Exception:
            size = _FALLBACK_SIZE
            orig_tensor = self._to_tensor(img, size, device)

        orig_patches = self._forward_patches(model, orig_tensor)   # [P, D]
        orig_norm    = F.normalize(orig_patches, dim=-1)            # [P, D]

        # Pre-generate all noise tensors at once; iterate for DINOv2 forward
        # (DINOv2 forward_features does not support batch > 1 across all builds).
        # n_perturbations is intentionally small (default 3) for CPU practicality.
        noise_batch = torch.randn(
            self.n_perturbations, *orig_tensor.shape[1:],
            device=orig_tensor.device, dtype=orig_tensor.dtype,
        ) * self.noise_std   # [N, 3, H, W]

        distances: list[float] = []
        with torch.no_grad():
            for i in range(self.n_perturbations):
                perturbed = orig_tensor + noise_batch[i : i + 1]
                perturbed_patches = self._forward_patches(model, perturbed)   # [P, D]
                perturbed_norm    = F.normalize(perturbed_patches, dim=-1)
                cos_dist = (1.0 - (orig_norm * perturbed_norm).sum(dim=-1)).mean().item()
                distances.append(cos_dist)

        return float(np.mean(distances))

    def predict(
        self,
        img: Image.Image,
        threshold: float = _THRESHOLD,
        **kwargs,
    ) -> dict:
        """Run RIGID detection.

        Args:
            img: Input PIL image (any size; resized internally to 518×518).
            threshold: Sensitivity below this → "likely_ai".

        Returns:
            score           float  — mean cosine distance (higher = more likely real)
            verdict         str    — "likely_ai" | "likely_real" | "unavailable"
            label           str    — "Fake" | "Real" | "Unknown"
            confidence      float  — probability of predicted class
            n_perturbations int    — number of perturbations used
            threshold       float  — threshold used
        """
        try:
            score = self._perturbation_sensitivity(img)
        except Exception as exc:
            logger.warning("RIGID: sensitivity computation failed: %s", exc)
            return {
                "score": None,
                "verdict": "unavailable",
                "label": "Unknown",
                "confidence": 0.0,
                "n_perturbations": self.n_perturbations,
                "threshold": threshold,
            }

        verdict = "likely_ai" if score < threshold else "likely_real"
        label   = "Fake" if verdict == "likely_ai" else "Real"
        # Sigmoid centred on threshold: low sensitivity → high fake probability
        fake_prob  = float(1.0 / (1.0 + np.exp((score - threshold) * 50)))
        confidence = round(fake_prob if label == "Fake" else 1.0 - fake_prob, 4)

        return {
            "score":           round(score, 6),
            "verdict":         verdict,
            "label":           label,
            "confidence":      confidence,
            "n_perturbations": self.n_perturbations,
            "threshold":       threshold,
        }

    # Alias so task-spec name works alongside BaseDetector's predict()
    detect = predict


if __name__ == "__main__":
    import sys

    det = RIGIDDetector.__new__(RIGIDDetector)
    det.n_perturbations = 10
    det.noise_std = 0.05
    det._device_override = None
    det._model = None
    det._device = None

    # Simulate DINOv2 unavailable
    det._get_model = lambda: (None, None)
    r = det.predict(Image.new("RGB", (64, 64)))
    assert r["score"] is None
    assert r["verdict"] == "unavailable"
    print("RIGID scaffold OK")
    sys.exit(0)
