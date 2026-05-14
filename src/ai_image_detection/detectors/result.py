"""Standard return schema shared by all detectors."""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class DetectorResult(TypedDict, total=False):
    """Mandatory keys every detector predict() must return.

    Optional extra keys (error_map, reasoning, backbone_used, etc.) are allowed
    but must not conflict with these names.
    """

    label: str          # "Fake" | "Real" | "Unknown"
    score: Optional[float]   # fake probability [0, 1]; None when unavailable
    confidence: float   # probability of the predicted class [0, 1]
    threshold: float    # decision threshold used
    verdict: str        # human-readable verdict string
