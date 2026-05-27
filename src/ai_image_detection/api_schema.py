"""
Pydantic v2 request / response schemas for the REST API.

These are the contracts between the API layer (api.py) and any external caller.
All numpy arrays are serialised to base64-encoded PNG strings so the response
is pure JSON — no binary blobs in the response body.

Usage:
    from ai_image_detection.api_schema import AnalyzeResponse
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────────────────

class CustomPipelineRequest(BaseModel):
    """Body for POST /analyze/custom (JSON portion of the multipart form)."""
    signals: list[str] = Field(
        default=[],
        description="Signal keys to run. Empty = all available signals.",
        examples=[["srm", "ela", "dual_branch"]],
    )
    use_damage_roi: bool = Field(
        default=False,
        description="If True and yolo_damage is selected, run signals on the 1.4× expanded YOLO ROI.",
    )
    manual_roi: Optional[dict[str, int]] = Field(
        default=None,
        description="Optional user-drawn ROI: {x, y, width, height} in pixels.",
        examples=[{"x": 10, "y": 20, "width": 300, "height": 200}],
    )


# ── Response models ───────────────────────────────────────────────────────────

class SignalResult(BaseModel):
    """Per-signal breakdown row returned in every analysis response."""
    name:             str
    label:            str
    score:            Optional[float] = None
    normalised_score: Optional[float] = None
    base_weight:      float
    weight_used:      Optional[float] = None
    available:        bool


class EnsembleResult(BaseModel):
    """Ensemble scoring summary."""
    ensemble_score:   float = Field(description="Fake probability [0, 1]")
    verdict:          str   = Field(description="Authentic | Suspicious | Likely Fraudulent")
    confidence:       str   = Field(description="HIGH | MEDIUM | LOW")
    signal_count:     int
    mode:             str   = Field(description="fixed | trained | fixed_fallback")
    weights_used:     dict[str, float]
    signal_breakdown: list[SignalResult]
    fallback_reason:  Optional[str] = None


class AnalyzeResponse(BaseModel):
    """
    Unified response for all /analyze/{pipeline} endpoints.

    Heatmaps and residual maps are base64-encoded PNG data URIs so the
    response is plain JSON — no binary blobs. Render in a browser with:
        <img src="{{ heatmap_b64 }}" />
    """
    pipeline:   str
    verdict:    str   = Field(description="Authentic | Suspicious | Likely Fraudulent")
    score:      float = Field(description="Ensemble fake probability [0, 1]")
    confidence: str   = Field(description="HIGH | MEDIUM | LOW")

    signals:    dict[str, float]    = Field(description="Raw per-signal scores")
    ensemble:   EnsembleResult
    skipped:    list[str]           = Field(default_factory=list)

    # Optional extras — present when the corresponding signal ran successfully
    heatmaps:         dict[str, str]  = Field(
        default_factory=dict,
        description="Signal name → base64-encoded PNG heatmap data URI",
    )
    roi_crop_applied:   Optional[bool] = None
    manual_roi_applied: Optional[bool] = None
    manual_roi:         Optional[dict[str, int]] = None


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str


class PipelineInfo(BaseModel):
    name:        str
    description: str
    signals:     list[str]


class PipelinesResponse(BaseModel):
    pipelines: list[PipelineInfo]
    all_signals: list[str]
