"""
AI Image Detection — FastAPI REST API.

Wraps the detection framework as a deployable HTTP service.

─────────────────────────────────────────────────────────────────────────
LAUNCH (development)
    pip install "ai-image-detection[api]"
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

LAUNCH (production / Docker)
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

    Note: use workers=1 when running on GPU — the detector cache is
    process-local; multiple workers each load their own model copies.
    Scale horizontally (multiple containers) rather than vertically.

INTERACTIVE DOCS
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)

─────────────────────────────────────────────────────────────────────────
ENDPOINTS

    GET  /health              Load-balancer / k8s liveness probe
    GET  /pipelines           Discover available pipelines and signals
    POST /analyze/{pipeline}  Run detection

    pipeline ∈ { id_card | document_fraud | vehicle_damage | custom }

─────────────────────────────────────────────────────────────────────────
IMAGE UPLOAD

    All /analyze endpoints accept multipart/form-data with a single
    field called "file".  The image may be JPEG, PNG, WebP, TIFF, BMP.

    For the "custom" pipeline, pass additional JSON form fields:
        signals        (JSON array)   e.g. '["srm","ela","dual_branch"]'
        use_damage_roi (bool)         default false
        manual_roi     (JSON object)  e.g. '{"x":10,"y":20,"width":300,"height":200}'

─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from ai_image_detection import __version__
from ai_image_detection.api_schema import (
    AnalyzeResponse,
    EnsembleResult,
    HealthResponse,
    PipelineInfo,
    PipelinesResponse,
    SignalResult,
)
from ai_image_detection.pipelines import (
    ALL_SIGNAL_KEYS,
    PRESET_SIGNALS,
    SIGNAL_LABELS,
    run_custom_pipeline,
    run_document_fraud_pipeline,
    run_id_card_pipeline,
    run_vehicle_damage_pipeline,
)

logger = logging.getLogger(__name__)

# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Image Detection API",
    description=(
        "Forensic detection of AI-generated, AI-edited and fraudulent document images. "
        "Supports ID cards, general documents, and vehicle damage reports."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins in dev; restrict in production via environment config.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB hard limit


def _load_image(upload: UploadFile) -> Image.Image:
    """Read an uploaded file and return a PIL Image, with size guard."""
    data = upload.file.read()
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {_MAX_IMAGE_BYTES // (1024*1024)} MB limit.",
        )
    try:
        img = Image.open(io.BytesIO(data))
        img.load()   # force decode so we catch corrupt files here, not mid-pipeline
        return img
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image: {exc}",
        ) from exc


def _ndarray_to_b64_png(arr: np.ndarray) -> str:
    """
    Convert a numpy array (heatmap / residual map) to a base64-encoded PNG
    data URI string.  Accepts float32 [0,1] or uint8 arrays.
    Returns an empty string if conversion fails (non-fatal).
    """
    try:
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as exc:
        logger.warning("heatmap serialisation failed: %s", exc)
        return ""


def _extract_heatmaps(details: dict) -> dict[str, str]:
    """
    Walk the details dict from a pipeline result and convert any numpy arrays
    (heatmap, residual_map, error_map, localization_map) to base64 PNG strings.
    """
    ARRAY_KEYS = ("heatmap", "residual_map", "error_map", "localization_map", "noiseprint_map")
    heatmaps: dict[str, str] = {}
    for signal_name, signal_detail in details.items():
        if not isinstance(signal_detail, dict):
            continue
        for key in ARRAY_KEYS:
            arr = signal_detail.get(key)
            if isinstance(arr, np.ndarray) and arr.size > 0:
                b64 = _ndarray_to_b64_png(arr)
                if b64:
                    heatmaps[f"{signal_name}_{key}"] = b64
    return heatmaps


def _build_response(raw: dict) -> AnalyzeResponse:
    """
    Convert a raw pipeline result dict into a typed AnalyzeResponse.
    Serialises numpy arrays to base64 and normalises optional fields.
    """
    ensemble_raw  = raw.get("ensemble", {})
    breakdown_raw = ensemble_raw.get("signal_breakdown", [])
    details       = raw.get("details", {})

    ensemble = EnsembleResult(
        ensemble_score   = ensemble_raw.get("ensemble_score", raw.get("score", 0.5)),
        verdict          = ensemble_raw.get("verdict", raw.get("verdict", "Unknown")),
        confidence       = ensemble_raw.get("confidence", raw.get("confidence", "LOW")),
        signal_count     = ensemble_raw.get("signal_count", 0),
        mode             = ensemble_raw.get("mode", "fixed"),
        weights_used     = ensemble_raw.get("weights_used", {}),
        signal_breakdown = [
            SignalResult(
                name             = row.get("name", ""),
                label            = row.get("label", ""),
                score            = row.get("score"),
                normalised_score = row.get("normalised_score"),
                base_weight      = row.get("base_weight", 0.0),
                weight_used      = row.get("weight_used"),
                available        = row.get("available", False),
            )
            for row in breakdown_raw
        ],
        fallback_reason = ensemble_raw.get("_fallback_reason"),
    )

    return AnalyzeResponse(
        pipeline           = raw.get("pipeline", "unknown"),
        verdict            = raw.get("verdict", "Unknown"),
        score              = raw.get("score", 0.5),
        confidence         = raw.get("confidence", "LOW"),
        signals            = raw.get("signals", {}),
        ensemble           = ensemble,
        skipped            = raw.get("skipped", []),
        heatmaps           = _extract_heatmaps(details),
        roi_crop_applied   = raw.get("roi_crop_applied"),
        manual_roi_applied = raw.get("manual_roi_applied"),
        manual_roi         = raw.get("manual_roi"),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["meta"],
)
def health() -> HealthResponse:
    """Returns 200 OK when the service is up.  Use for k8s/load-balancer checks."""
    return HealthResponse(status="ok", version=__version__)


@app.get(
    "/pipelines",
    response_model=PipelinesResponse,
    summary="List available pipelines and signals",
    tags=["meta"],
)
def list_pipelines() -> PipelinesResponse:
    """Describe available pipelines and the signals each one runs."""
    _DESCRIPTIONS = {
        "id_card":        "ID card / passport edit detection (DualBranch, TruFor, SRM, ELA, PRNU …)",
        "document_fraud": "General document fraud (SigLIP-2, TruFor, SRM, ELA, PRNU …)",
        "vehicle_damage": "Vehicle damage fraud (YOLO ROI, DualBranch, TruFor, SRM, Metadata …)",
        "custom":         "User-selected signal combination — pass 'signals' in the request body.",
    }
    return PipelinesResponse(
        pipelines=[
            PipelineInfo(
                name=name,
                description=_DESCRIPTIONS.get(name, ""),
                signals=sorted(sigs),
            )
            for name, sigs in PRESET_SIGNALS.items()
        ],
        all_signals=ALL_SIGNAL_KEYS,
    )


@app.post(
    "/analyze/id_card",
    response_model=AnalyzeResponse,
    summary="Run ID card / passport pipeline",
    tags=["analyze"],
)
async def analyze_id_card(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, TIFF)"),
) -> AnalyzeResponse:
    """
    Run the ID card edit-detection pipeline on the uploaded image.

    Signals: DualBranch, TruFor, SRM (+ JPEG grid), ELA, PRNU, DCT/Benford, CFA.
    """
    img = _load_image(file)
    try:
        raw = run_id_card_pipeline(img)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("id_card pipeline error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return _build_response(raw)


@app.post(
    "/analyze/document_fraud",
    response_model=AnalyzeResponse,
    summary="Run general document fraud pipeline",
    tags=["analyze"],
)
async def analyze_document_fraud(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, TIFF)"),
) -> AnalyzeResponse:
    """
    Run the document fraud detection pipeline on the uploaded image.

    Signals: SigLIP-2, TruFor, SRM (+ JPEG grid), ELA, PRNU.
    """
    img = _load_image(file)
    try:
        raw = run_document_fraud_pipeline(img)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("document_fraud pipeline error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return _build_response(raw)


@app.post(
    "/analyze/vehicle_damage",
    response_model=AnalyzeResponse,
    summary="Run vehicle damage fraud pipeline",
    tags=["analyze"],
)
async def analyze_vehicle_damage(
    file:           UploadFile = File(..., description="Image file (JPEG, PNG, WebP, TIFF)"),
    use_damage_roi: bool       = Form(default=False, description="Run signals on YOLO damage ROI crop"),
) -> AnalyzeResponse:
    """
    Run the vehicle damage fraud detection pipeline on the uploaded image.

    Signals: YOLO damage detector (ROI), DualBranch, TruFor, SRM, EXIF Metadata.
    Pass use_damage_roi=true to focus all signals on the detected damage region.
    """
    img = _load_image(file)
    try:
        raw = run_vehicle_damage_pipeline(img, use_damage_roi=use_damage_roi)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("vehicle_damage pipeline error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return _build_response(raw)


@app.post(
    "/analyze/custom",
    response_model=AnalyzeResponse,
    summary="Run a custom signal combination",
    tags=["analyze"],
)
async def analyze_custom(
    file:           UploadFile    = File(...,         description="Image file"),
    signals:        Optional[str] = Form(default=None, description="JSON array of signal keys"),
    use_damage_roi: bool          = Form(default=False),
    manual_roi:     Optional[str] = Form(default=None, description="JSON object {x,y,width,height}"),
) -> AnalyzeResponse:
    """
    Run any combination of signals on the uploaded image.

    Pass `signals` as a JSON array, e.g.:
        signals=["srm","ela","dual_branch","trufor"]

    If signals is omitted, all available signals are attempted.

    Pass `manual_roi` as a JSON object to run signals on a specific crop:
        manual_roi={"x":50,"y":80,"width":400,"height":300}
    """
    img = _load_image(file)

    # Parse signals list from JSON string form field
    selected: list[str]
    if signals:
        try:
            selected = json.loads(signals)
            if not isinstance(selected, list):
                raise ValueError("signals must be a JSON array")
            # Validate against known keys
            unknown = [s for s in selected if s not in ALL_SIGNAL_KEYS]
            if unknown:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Unknown signal keys: {unknown}. Valid: {ALL_SIGNAL_KEYS}",
                )
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"'signals' is not valid JSON: {exc}",
            ) from exc
    else:
        selected = list(ALL_SIGNAL_KEYS)

    # Parse optional manual_roi
    roi: Optional[dict] = None
    if manual_roi:
        try:
            roi = json.loads(manual_roi)
            required = {"x", "y", "width", "height"}
            if not required.issubset(roi.keys()):
                raise ValueError(f"manual_roi must have keys: {required}")
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid manual_roi: {exc}",
            ) from exc

    try:
        raw = run_custom_pipeline(
            img,
            selected=selected,
            use_damage_roi=use_damage_roi,
            manual_roi=roi,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("custom pipeline error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return _build_response(raw)
