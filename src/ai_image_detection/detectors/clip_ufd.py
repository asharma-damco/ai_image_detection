"""
UniversalFakeDetect (CLIP ViT-L/14) Adapter.

Architecture: CLIP ViT-L/14 backbone + nn.Linear(768, 1) fine-tune head.
Trained on CVPR 2023 synthetic image detection dataset.
Output: single logit → sigmoid() → fake probability [0, 1].

Setup (one-time):
    cd weights/
    git clone https://github.com/Yuheng-Li/UniversalFakeDetect.git
    # weights are included at UniversalFakeDetect/pretrained_weights/fc_weights.pth

Source: UAIC uaic-fraud-detection/poc/universal_fake_detect_adapter.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image

from ..config import UFD_REPO, UFD_WEIGHTS
from .base import BaseDetector

_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

_PREPROCESS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
])


class UFDNotInstalledError(RuntimeError):
    pass


def _ensure_repo() -> None:
    if not UFD_REPO.exists():
        raise UFDNotInstalledError(
            f"UniversalFakeDetect repo not found at:\n  {UFD_REPO}\n\n"
            "Run:\n  cd weights/\n  git clone https://github.com/Yuheng-Li/UniversalFakeDetect.git"
        )
    # Verify the repo is the real UFD package (not just any directory at that path).
    if not (UFD_REPO / "models").is_dir():
        raise UFDNotInstalledError(
            f"'models' directory missing in {UFD_REPO}.\n"
            "The clone may be incomplete. Remove and re-clone:\n"
            "  cd weights/\n  git clone https://github.com/Yuheng-Li/UniversalFakeDetect.git"
        )
    if str(UFD_REPO) not in sys.path:
        sys.path.insert(0, str(UFD_REPO))


def is_available() -> bool:
    try:
        _ensure_repo()
        return True
    except UFDNotInstalledError:
        return False


class UFDAdapter(BaseDetector):
    """
    Loads UniversalFakeDetect (CLIP ViT-L/14) and exposes predict() / predict_roi().
    Model is lazy-loaded on first call.
    """

    MODEL_VERSION = "ufd-clip-vitl14"

    def __init__(self, device: Optional[torch.device] = None):
        self.device   = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model   = None
        self._loaded  = False

    def load(self) -> "UFDAdapter":
        if self._loaded:
            return self
        _ensure_repo()

        from models import get_model  # type: ignore[import]
        model = get_model("CLIP:ViT-L/14")

        if UFD_WEIGHTS.exists():
            state_dict = torch.load(str(UFD_WEIGHTS), map_location=self.device)
            model.fc.load_state_dict(state_dict)

        self._model  = model.to(self.device).eval()
        self._loaded = True
        return self

    def predict(self, img: Image.Image, threshold: float = 0.5, **kwargs) -> dict:
        """
        Classify a full image.

        Args:
            img:       PIL RGB image (any size).
            threshold: fake probability >= threshold → label = "Fake".

        Returns:
            label, confidence, probabilities, score
        """
        if not self._loaded:
            self.load()

        tensor = _PREPROCESS(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit     = self._model(tensor)
            fake_prob = torch.sigmoid(logit).item()

        real_prob  = 1.0 - fake_prob
        label      = "Fake" if fake_prob >= threshold else "Real"
        confidence = fake_prob if label == "Fake" else real_prob

        return {
            "label":         label,
            "confidence":    round(confidence, 4),
            "probabilities": {"Real": round(real_prob, 4), "Fake": round(fake_prob, 4)},
            "score":         round(fake_prob, 4),
            "model_version": self.MODEL_VERSION,
            "threshold":     threshold,
        }

    @property
    def weights_found(self) -> bool:
        return UFD_WEIGHTS.exists()
