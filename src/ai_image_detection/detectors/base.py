"""Base interface that all ML-model detectors implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class BaseDetector(ABC):
    """Common contract for all detectors.

    Every detector must implement predict() which accepts a PIL Image and
    returns a dict containing at minimum:
        score       float  fake probability [0, 1]  (0=authentic, 1=fake)
        label       str    "Real" | "Fake"
        confidence  float  probability of the predicted class [0, 1]
    """

    @abstractmethod
    def predict(self, img: Image.Image, **kwargs) -> dict:
        """Run inference on a PIL Image. Returns result dict."""

    def predict_roi(self, img: Image.Image, roi: dict, **kwargs) -> dict:
        """Crop to roi {"x","y","width","height"} then predict."""
        x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
        crop = img.crop((x, y, x + w, y + h))
        result = self.predict(crop, **kwargs)
        result["roi"] = roi
        return result

    @property
    def name(self) -> str:
        return self.__class__.__name__
