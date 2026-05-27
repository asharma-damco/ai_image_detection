"""
EXIF / Metadata Forensics Signal.

AI-generated images (diffusion models, GANs) typically lack authentic camera
metadata. Scores images on a 0–1 "authenticity" scale based on the presence
and consistency of EXIF fields that real cameras always produce.

Note: metadata can be injected. Use as supporting evidence alongside model signals.

Interpretation:
    authenticity_score < 0.30  → Strong AI signal
    0.30–0.55                  → Moderate suspicion
    0.55–0.75                  → Likely real, some fields missing
    > 0.75                     → Consistent with real camera output

Source: UAIC uaic-fraud-detection/src/backend/app/ml/metadata_analyzer.py
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.ExifTags import TAGS

CHECKS: dict[str, float] = {
    "camera_make_model":  0.20,
    "exposure_triangle":  0.20,
    "focal_length":       0.10,
    "datetime_original":  0.10,
    "gps_data":           0.05,
    "software_signature": 0.10,
    "thumbnail_present":  0.10,
    "pixel_count":        0.05,
    "exif_consistency":   0.05,
    "image_format":       0.05,
}

_AI_SOFTWARE_FRAGMENTS = [
    "stable diffusion", "midjourney", "dall-e", "dalle",
    "firefly", "imagen", "generative fill", "ai generated",
    "adobe firefly", "canva",
]
_CAMERA_MAKE_FRAGMENTS = [
    "canon", "nikon", "sony", "fujifilm", "olympus", "panasonic",
    "leica", "hasselblad", "pentax", "apple", "samsung", "google",
    "lg", "huawei", "xiaomi", "oneplus",
]


def _decode_exif(img: Image.Image) -> dict:
    try:
        raw = img._getexif()  # type: ignore[attr-defined]
        return {TAGS.get(k, k): v for k, v in raw.items()} if raw else {}
    except Exception:
        return {}


def _check_camera_make_model(exif: dict) -> tuple[float, str]:
    make  = str(exif.get("Make",  "")).strip()
    model = str(exif.get("Model", "")).strip()
    if make and model:
        is_known = any(f in make.lower() for f in _CAMERA_MAKE_FRAGMENTS)
        return (1.0, f"Make={make!r}, Model={model!r}") if is_known else (0.7, f"Make={make!r}, Model={model!r} (unknown brand)")
    if make or model:
        return (0.4, f"Partial: Make={make!r} Model={model!r}")
    return (0.0, "No camera make/model")


def _check_exposure_triangle(exif: dict) -> tuple[float, str]:
    has_f, has_exp = "FNumber" in exif, "ExposureTime" in exif
    has_iso = "ISOSpeedRatings" in exif or "PhotographicSensitivity" in exif
    count = sum([has_f, has_exp, has_iso])
    notes = []
    if has_f:   notes.append(f"f/{exif['FNumber']}")
    if has_exp: notes.append(f"t={exif['ExposureTime']}")
    if has_iso: notes.append(f"ISO={exif.get('ISOSpeedRatings') or exif.get('PhotographicSensitivity')}")
    return (count / 3.0, ", ".join(notes) if notes else "None of f/t/ISO found")


def _check_focal_length(exif: dict) -> tuple[float, str]:
    fl = exif.get("FocalLength")
    if fl is None:
        return (0.0, "FocalLength absent")
    try:
        # EXIF stores rationals as (numerator, denominator) tuples — e.g. (35, 1)
        # PIL's IFDRational also behaves like a fraction; float() works on it directly.
        if isinstance(fl, tuple) and len(fl) == 2:
            fl_mm = float(fl[0]) / float(fl[1]) if float(fl[1]) != 0 else 0.0
        else:
            fl_mm = float(fl)
    except (TypeError, ValueError):
        return (0.3, f"FocalLength unparseable: {fl!r}")
    return (1.0, f"FocalLength={fl_mm:.1f}mm") if 4.0 <= fl_mm <= 2000.0 else (0.2, f"FocalLength={fl_mm:.1f}mm (implausible)")


def _check_datetime_original(exif: dict) -> tuple[float, str]:
    dto = exif.get("DateTimeOriginal")
    if dto:
        return (1.0, f"DateTimeOriginal={dto!r}")
    dt = exif.get("DateTime")
    return (0.4, f"Only DateTime={dt!r} (may be edit timestamp)") if dt else (0.0, "No date/time tags")


def _check_gps(exif: dict) -> tuple[float, str]:
    gps = exif.get("GPSInfo")
    if isinstance(gps, dict) and len(gps) > 2:
        return (1.0, f"GPS block present ({len(gps)} sub-tags)")
    return (0.5, "GPSInfo exists but sparse") if gps else (0.0, "No GPS data")


def _check_software_signature(exif: dict) -> tuple[float, str]:
    sw = str(exif.get("Software", "")).strip()
    if not sw:
        return (0.8, "No Software tag")
    sw_lower = sw.lower()
    if any(f in sw_lower for f in _AI_SOFTWARE_FRAGMENTS):
        return (0.0, f"AI/generative software: {sw!r}")
    if "photoshop" in sw_lower or "gimp" in sw_lower or "lightroom" in sw_lower:
        return (0.4, f"Edit software: {sw!r}")
    return (1.0, f"Software={sw!r} (camera/benign)")


def _check_thumbnail(exif: dict, img: Image.Image) -> tuple[float, str]:
    try:
        raw = img._getexif()  # type: ignore[attr-defined]
        if raw is None:
            return (0.0, "No EXIF - no thumbnail")
        return (1.0, "Thumbnail present") if (513 in raw or 514 in raw) else (0.2, "No thumbnail in EXIF")
    except Exception:
        return (0.0, "Could not check thumbnail")


def _check_pixel_count(img: Image.Image) -> tuple[float, str]:
    W, H = img.size
    mp   = (W * H) / 1_000_000
    return (1.0, f"{W}×{H} = {mp:.1f}MP") if mp >= 1.0 else (0.3, f"{W}×{H} = {mp:.2f}MP (sub-megapixel)")


def _check_exif_consistency(exif: dict) -> tuple[float, str]:
    dt_str, dto_str = exif.get("DateTime"), exif.get("DateTimeOriginal")
    if not dt_str or not dto_str:
        return (0.5, "Cannot check — timestamps missing")
    fmt = "%Y:%m:%d %H:%M:%S"
    try:
        diff_h = abs((datetime.strptime(dt_str, fmt) - datetime.strptime(dto_str, fmt)).total_seconds()) / 3600
        if diff_h <= 1:   return (1.0, f"DateTime consistent (Δ={diff_h:.2f}h)")
        if diff_h <= 72:  return (0.7, f"DateTime differs {diff_h:.1f}h (editing likely)")
        return (0.3, f"DateTime differs {diff_h:.0f}h (unusual)")
    except ValueError:
        return (0.5, "Could not parse timestamp format")


def _check_image_format(img: Image.Image, file_path: Optional[str] = None) -> tuple[float, str]:
    fmt = (img.format or "").upper()
    if not fmt and file_path:
        ext = Path(file_path).suffix.lower()
        fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".tiff": "TIFF", ".tif": "TIFF", ".webp": "WEBP"}.get(ext, "UNKNOWN")
    if fmt == "JPEG":          return (1.0, "JPEG format (cameras default)")
    if fmt in ("TIFF", "DNG"): return (1.0, f"{fmt} format (professional camera)")
    if fmt == "PNG":           return (0.2, "PNG format — cameras never produce PNG; common for AI outputs")
    if fmt == "WEBP":          return (0.5, "WebP — modern phone format, ambiguous")
    return (0.5, f"Format={fmt!r}")


def analyze_metadata(img: Image.Image, file_path: Optional[str] = None) -> dict:
    """Run all metadata forensic checks on a PIL Image.

    Returns:
        authenticity_score float   0–1 (higher = more like real camera)
        authenticity_label str
        checks             dict    per-check results
        has_exif           bool
        summary_flags      list[str]
    """
    exif     = _decode_exif(img)
    has_exif = bool(exif)

    raw_scores = {
        "camera_make_model": _check_camera_make_model(exif),
        "exposure_triangle": _check_exposure_triangle(exif),
        "focal_length":      _check_focal_length(exif),
        "datetime_original": _check_datetime_original(exif),
        "gps_data":          _check_gps(exif),
        "software_signature": _check_software_signature(exif),
        "thumbnail_present": _check_thumbnail(exif, img),
        "pixel_count":       _check_pixel_count(img),
        "exif_consistency":  _check_exif_consistency(exif),
        "image_format":      _check_image_format(img, file_path),
    }

    results: dict        = {}
    summary_flags: list  = []
    weighted_sum = total_weight = 0.0

    for check, (s, note) in raw_scores.items():
        w = CHECKS[check]
        results[check] = {"score": round(s, 3), "weight": w, "note": note, "passed": s >= 0.5}
        weighted_sum  += s * w;  total_weight += w
        if s < 0.3:
            summary_flags.append(f"{check}: {note}")

    auth_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    if auth_score < 0.30:    label = "Strong AI signal"
    elif auth_score < 0.50:  label = "Suspicious — metadata inconsistent"
    elif auth_score < 0.75:  label = "Likely Real — minor gaps"
    else:                    label = "Consistent with camera output"

    return {
        "authenticity_score": round(auth_score, 3),
        "authenticity_label": label,
        "checks":             results,
        "has_exif":           has_exif,
        "summary_flags":      summary_flags,
    }
