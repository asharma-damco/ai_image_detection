"""
YOLOv11m Car Damage Detection Adapter.

Model: ReverendBayes/YOLO11m-Car-Damage-Detector (CarDD_COCO fine-tune, ~20M params)
Damage classes: dent, scratch, crack, shattered glass, broken lamp, flat tire

Weights are downloaded once from GitHub and cached at weights/yolo_damage_detect/trained.pt.

Setup:
    pip install ultralytics
    Weights download automatically on first load.

Source: UAIC uaic-fraud-detection/poc/yolo_damage_adapter.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..config import YOLO_CACHED_WEIGHTS, YOLO_GITHUB_URL, YOLO_MODEL_DIR

DAMAGE_CLASSES = {"dent", "scratch", "crack", "shattered glass", "broken lamp", "flat tire"}


def is_available() -> tuple[bool, str]:
    """Return (True, 'cached'|'github') or (False, error_message)."""
    try:
        import ultralytics  # noqa: F401
        return (True, "cached") if YOLO_CACHED_WEIGHTS.exists() else (True, "github")
    except ImportError:
        return False, "ultralytics not installed. Run: pip install ultralytics"


def _download_weights() -> Path:
    import urllib.request
    YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = YOLO_CACHED_WEIGHTS
    try:
        urllib.request.urlretrieve(YOLO_GITHUB_URL, dest)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download YOLO11m weights.\nURL: {YOLO_GITHUB_URL}\nError: {e}\n\n"
            f"Manual download: save trained.pt from the GitHub repo to {dest}"
        ) from e
    return dest


class DamageDetector:
    """
    YOLOv11m damage detector.

    detect(img) → list of bbox dicts sorted by confidence descending.
    best_roi(img) / composite_roi(img) → ROI dict for DualBranch follow-up.
    """

    MODEL_VERSION = "yolo11m-car-damage-reverend-bayes"

    def __init__(self, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self._model: Any   = None
        self._loaded        = False
        self._mode: str     = "not loaded"

    def load(self) -> "DamageDetector":
        if self._loaded:
            return self
        from ultralytics import YOLO  # type: ignore[import]
        weights_path = YOLO_CACHED_WEIGHTS if YOLO_CACHED_WEIGHTS.exists() else _download_weights()
        self._model  = YOLO(str(weights_path))
        self._mode   = "cached" if YOLO_CACHED_WEIGHTS.exists() else "github"
        self._loaded = True
        return self

    def detect(self, img: Image.Image) -> list[dict]:
        """
        Run damage detection on a PIL image.

        Returns:
            list of dicts sorted by confidence (highest first):
            {"bbox": [x,y,w,h], "bbox_xyxy": [x1,y1,x2,y2],
             "confidence": float, "label": str, "class_id": int}
        """
        if not self._loaded:
            self.load()

        results    = self._model.predict(source=img.convert("RGB"), conf=self.conf_threshold, verbose=False)
        detections: list[dict] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                cls_id          = int(box.cls[0].item())
                label           = result.names.get(cls_id, str(cls_id))
                conf            = float(box.conf[0].item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append({
                    "bbox":       [x1, y1, x2 - x1, y2 - y1],
                    "bbox_xyxy":  [x1, y1, x2, y2],
                    "confidence": conf,
                    "label":      label,
                    "class_id":   cls_id,
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def best_roi(self, img: Image.Image) -> Optional[dict]:
        """Return highest-confidence detection as ROI dict {"x","y","width","height"}."""
        detections = self.detect(img)
        if not detections:
            return None
        iw, ih = img.size
        x, y, w, h = detections[0]["bbox"]
        x = max(0, x);  y = max(0, y)
        w = min(w, iw - x);  h = min(h, ih - y)
        return {"x": x, "y": y, "width": w, "height": h} if w >= 32 and h >= 32 else None

    def composite_roi(self, img: Image.Image) -> Optional[dict]:
        """Return union of all detections as ROI dict (tightest bounding box)."""
        detections = self.detect(img)
        if not detections:
            return None
        iw, ih = img.size
        x1 = max(0, min(d["bbox_xyxy"][0] for d in detections))
        y1 = max(0, min(d["bbox_xyxy"][1] for d in detections))
        x2 = min(iw, max(d["bbox_xyxy"][2] for d in detections))
        y2 = min(ih, max(d["bbox_xyxy"][3] for d in detections))
        w, h = x2 - x1, y2 - y1
        return {"x": x1, "y": y1, "width": w, "height": h} if w >= 32 and h >= 32 else None

    @property
    def mode(self) -> str:
        return self._mode
