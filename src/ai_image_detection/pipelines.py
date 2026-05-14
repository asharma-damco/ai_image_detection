"""
Pipeline runner functions — presets + custom.

Each function accepts a PIL.Image, runs forensic signals, fuses via
EnsembleScorer, and returns:
    pipeline, verdict, score, confidence, signals, ensemble, skipped

Detector imports are lazy (inside try/except) so a missing dependency
(e.g. torch not installed) skips that signal rather than crashing the pipeline.
"""

from __future__ import annotations

from PIL import Image

from .config import IMAGE_MIN_SIZE

# ── Lazy-import pattern note ───────────────────────────────────────────────────
# Detector imports live inside try/except blocks to allow partial operation when
# optional dependencies (torch, diffusers, transformers) are not installed.
# This is intentional — do not hoist them to module level.

# ── Module-level detector cache ───────────────────────────────────────────────
# Prevents reloading model weights on each pipeline call within a session.
# Key: detector class name. Value: instantiated detector.
_DETECTOR_CACHE: dict[str, object] = {}


def _cached(key: str, factory):
    """Return a cached detector instance, creating it on first call."""
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = factory()
    return _DETECTOR_CACHE[key]


def _validate_image(img: Image.Image) -> Image.Image:
    """Validate minimum image dimensions and normalise to RGB."""
    w, h = img.size
    if w < IMAGE_MIN_SIZE or h < IMAGE_MIN_SIZE:
        raise ValueError(
            f"Image too small ({w}×{h} px). "
            f"Minimum accepted size: {IMAGE_MIN_SIZE}×{IMAGE_MIN_SIZE} px."
        )
    return img.convert("RGB") if img.mode != "RGB" else img


# ── Preset pipelines ──────────────────────────────────────────────────────────

def run_id_card_pipeline(img: Image.Image) -> dict:
    """ID card edit detection pipeline."""
    import numpy as np
    from .config import ID_CARD_WEIGHTS
    from .ensemble.scorer import EnsembleScorer

    img = _validate_image(img)
    img_arr = np.array(img)
    signals: dict = {}
    details: dict = {}
    skipped: list = []

    try:
        from .detectors.dual_branch import DualBranchDetector
        r = _cached("dual_branch", DualBranchDetector).predict(img)
        signals["dual_branch"] = r["score"]
        details["dual_branch"] = r
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"dual_branch ({type(e).__name__}: {e})")

    try:
        from .detectors.srm import SRMAnalyzer
        srm    = _cached("srm", SRMAnalyzer)
        srm_r  = srm.compute_anomaly_score(img)
        jpeg_r = srm.detect_jpeg_grid_inconsistency(img)
        combined = round(0.70 * srm_r["score"] + 0.30 * jpeg_r["score"], 4)
        signals["srm"] = combined
        details["srm"] = {"score": combined, "srm": srm_r, "jpeg": jpeg_r}
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"srm ({type(e).__name__}: {e})")

    try:
        from .detectors.trufor import TruForAnalyzer
        tf_r = _cached("trufor", TruForAnalyzer).analyze(img)
        signals["trufor"] = round(1.0 - tf_r["integrity_score"], 4)
        details["trufor"] = tf_r
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"trufor ({type(e).__name__}: {e})")

    try:
        from .signals.prnu import prnu_anomaly_score
        r = prnu_anomaly_score(img_arr)
        if r["score"] is not None:
            signals["prnu"] = round(float(r["score"]), 4)
            details["prnu"] = r
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"prnu ({type(e).__name__}: {e})")

    try:
        from .signals.dct_benford import dct_benford_score
        r = dct_benford_score(img_arr)
        if r["score"] is not None:
            signals["dct_benford"] = round(float(r["score"]), 4)
            details["dct_benford"] = r
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"dct_benford ({type(e).__name__}: {e})")

    try:
        from .signals.cfa import cfa_correlation_score
        r = cfa_correlation_score(img_arr)
        if r["score"] is not None:
            signals["cfa"] = round(float(r["score"]), 4)
            details["cfa"] = r
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"cfa ({type(e).__name__}: {e})")

    try:
        from .signals.ela import ela_anomaly_score
        ela_r = ela_anomaly_score(img)
        signals["ela"] = round(float(ela_r["score"]), 4)
        details["ela"] = ela_r
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"ela ({type(e).__name__}: {e})")

    scorer   = EnsembleScorer(custom_weights=ID_CARD_WEIGHTS)
    ensemble = scorer.score(signals, mode="fixed")

    return {
        "pipeline":   "id_card",
        "verdict":    ensemble["verdict"],
        "score":      ensemble["ensemble_score"],
        "confidence": ensemble["confidence"],
        "signals":    signals,
        "ensemble":   ensemble,
        "skipped":    skipped,
    }


def run_document_fraud_pipeline(img: Image.Image) -> dict:
    """General document fraud detection pipeline."""
    import numpy as np
    from .config import DOCUMENT_FRAUD_WEIGHTS
    from .ensemble.scorer import EnsembleScorer

    img = _validate_image(img)
    img_arr = np.array(img)
    signals: dict = {}
    details: dict = {}
    skipped: list = []

    try:
        from .detectors.siglip2 import SigLIP2Detector
        score = _cached("siglip2", SigLIP2Detector).score(img)
        signals["siglip2"] = round(score, 4)
        details["siglip2"] = {"score": score}
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"siglip2 ({type(e).__name__}: {e})")

    try:
        from .detectors.trufor import TruForAnalyzer
        tf_r = _cached("trufor", TruForAnalyzer).analyze(img)
        signals["trufor"]                     = round(1.0 - tf_r["integrity_score"], 4)
        signals["noiseprint_localizer"]       = round(float(tf_r.get("mask_mean", 0.0)), 4)
        signals["mask_spatial_concentration"] = round(1.0 - tf_r.get("mask_spatial_concentration", 0.3), 4)
        details["trufor"] = tf_r
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"trufor ({type(e).__name__}: {e})")

    try:
        from .detectors.srm import SRMAnalyzer
        srm_r = _cached("srm", SRMAnalyzer).compute_anomaly_score(img)
        signals["srm"] = srm_r["score"]
        details["srm"] = srm_r
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"srm ({type(e).__name__}: {e})")

    try:
        from .signals.prnu import prnu_anomaly_score
        r = prnu_anomaly_score(img_arr)
        if r["score"] is not None:
            signals["prnu"] = round(float(r["score"]), 4)
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"prnu ({type(e).__name__}: {e})")

    try:
        from .signals.ela import ela_anomaly_score
        signals["ela"] = round(float(ela_anomaly_score(img)["score"]), 4)
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"ela ({type(e).__name__}: {e})")

    scorer   = EnsembleScorer(custom_weights=DOCUMENT_FRAUD_WEIGHTS)
    ensemble = scorer.score(signals, mode="fixed")

    return {
        "pipeline":   "document_fraud",
        "verdict":    ensemble["verdict"],
        "score":      ensemble["ensemble_score"],
        "confidence": ensemble["confidence"],
        "signals":    signals,
        "ensemble":   ensemble,
        "skipped":    skipped,
    }


def run_vehicle_damage_pipeline(
    img: Image.Image,
    use_damage_roi: bool = False,
) -> dict:
    """Vehicle damage fraud detection pipeline.

    Parameters
    ----------
    use_damage_roi : if True, all non-YOLO signals run on the 1.4× expanded
                     damage ROI crop instead of the full image.
    """
    from .config import VEHICLE_DAMAGE_WEIGHTS
    from .ensemble.scorer import EnsembleScorer

    signals: dict = {}
    details: dict = {}
    skipped: list = []

    img = _validate_image(img)
    iw, ih = img.size

    # ── YOLO damage detection (run first; ROI may crop subsequent detectors) ──
    _roi_crop_applied = False
    try:
        from .detectors.yolo_damage import DamageDetector
        _detector   = _cached("yolo_damage", DamageDetector)
        _detections = _detector.detect(img)
        # YOLO score intentionally NOT added to signals{} — ROI use only.
        _comp = _detector.composite_roi(img)
        _exp  = None
        if _comp:
            cx = _comp["x"] + _comp["width"]  / 2
            cy = _comp["y"] + _comp["height"] / 2
            nw = _comp["width"]  * 1.4
            nh = _comp["height"] * 1.4
            x1 = max(0,  int(cx - nw / 2)); y1 = max(0,  int(cy - nh / 2))
            x2 = min(iw, int(cx + nw / 2)); y2 = min(ih, int(cy + nh / 2))
            _exp = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
        details["yolo_damage"] = {
            "detections":    _detections,
            "composite_roi": _comp,
            "expanded_roi":  _exp,
        }
        if use_damage_roi and _exp:
            img = img.crop((_exp["x"], _exp["y"],
                            _exp["x"] + _exp["width"],
                            _exp["y"] + _exp["height"]))
            _roi_crop_applied = True
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"yolo_damage ({type(e).__name__}: {e})")

    try:
        from .detectors.dual_branch import DualBranchDetector
        r = _cached("dual_branch", DualBranchDetector).predict(img)
        signals["dual_branch"] = r["score"]
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"dual_branch ({type(e).__name__}: {e})")

    try:
        from .detectors.trufor import TruForAnalyzer
        tf_r = _cached("trufor", TruForAnalyzer).analyze(img)
        signals["trufor"] = round(1.0 - tf_r["integrity_score"], 4)
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"trufor ({type(e).__name__}: {e})")

    try:
        from .detectors.srm import SRMAnalyzer
        signals["srm"] = _cached("srm", SRMAnalyzer).compute_anomaly_score(img)["score"]
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"srm ({type(e).__name__}: {e})")

    try:
        from .signals.metadata import analyze_metadata
        signals["metadata"] = round(1.0 - analyze_metadata(img)["authenticity_score"], 4)
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        skipped.append(f"metadata ({type(e).__name__}: {e})")

    scorer   = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS)
    ensemble = scorer.score(signals, mode="fixed")

    return {
        "pipeline":         "vehicle_damage",
        "verdict":          ensemble["verdict"],
        "score":            ensemble["ensemble_score"],
        "confidence":       ensemble["confidence"],
        "signals":          signals,
        "details":          details,
        "ensemble":         ensemble,
        "skipped":          skipped,
        "roi_crop_applied": _roi_crop_applied,
    }


# ── Custom pipeline ───────────────────────────────────────────────────────────

def run_custom_pipeline(
    img: Image.Image,
    selected: list[str],
    use_damage_roi: bool = False,
    manual_roi: dict | None = None,
    preset_weights: dict | None = None,
) -> dict:
    """Run any user-selected combination of signals.

    Parameters
    ----------
    selected        : list of signal keys from ALL_SIGNAL_KEYS
    use_damage_roi  : if True and yolo_damage is selected, all non-YOLO signals
                      run on the 1.4× expanded damage ROI crop instead of the
                      full image.
    manual_roi      : {"x","y","width","height"} — user-drawn ROI; when provided
                      the image is cropped to this region before any signal runs
                      (takes precedence over use_damage_roi).
    preset_weights  : optional calibrated weight dict (e.g. ID_CARD_WEIGHTS);
                      when supplied, used instead of equal weights. Missing
                      signals are auto-renormalised by EnsembleScorer.
    """
    import numpy as np
    from .ensemble.scorer import EnsembleScorer

    signals: dict = {}
    details: dict = {}   # visual data for UI heatmaps (heatmap/residual_map/error_map)
    skipped: list = []

    img = _validate_image(img)
    img_arr = np.array(img)

    # ── Manual ROI crop (applied first, before any signal including YOLO) ────
    _manual_roi_applied = False
    if manual_roi:
        _x, _y = manual_roi["x"], manual_roi["y"]
        _w, _h = manual_roi["width"], manual_roi["height"]
        img     = img.crop((_x, _y, _x + _w, _y + _h))
        img_arr = np.array(img)
        _manual_roi_applied = True

    # ── YOLO runs first so its ROI can crop subsequent signals ───────────────
    # Score intentionally NOT added to signals{} — YOLO is ROI-only, not fraud scored.
    if "yolo_damage" in selected:
        try:
            from .detectors.yolo_damage import DamageDetector
            _detector   = DamageDetector()
            _detections = _detector.detect(img)
            iw, ih      = img.size
            _comp = _detector.composite_roi(img)
            _exp  = None
            if _comp:
                cx = _comp["x"] + _comp["width"]  / 2
                cy = _comp["y"] + _comp["height"] / 2
                nw = _comp["width"]  * 1.4
                nh = _comp["height"] * 1.4
                x1 = max(0,  int(cx - nw / 2)); y1 = max(0,  int(cy - nh / 2))
                x2 = min(iw, int(cx + nw / 2)); y2 = min(ih, int(cy + nh / 2))
                _exp = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
            details["yolo_damage"] = {
                "detections":    _detections,
                "composite_roi": _comp,
                "expanded_roi":  _exp,
            }
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"yolo_damage ({type(e).__name__}: {e})")

    # ── Optionally crop to damage ROI before running remaining signals ───────
    _roi_crop_applied = False
    if use_damage_roi and "yolo_damage" in details:
        _exp = details["yolo_damage"].get("expanded_roi")
        if _exp:
            img     = img.crop((_exp["x"], _exp["y"],
                                _exp["x"] + _exp["width"],
                                _exp["y"] + _exp["height"]))
            img_arr = np.array(img)
            _roi_crop_applied = True

    if "dual_branch" in selected:
        try:
            from .detectors.dual_branch import DualBranchDetector
            signals["dual_branch"] = DualBranchDetector().predict(img)["score"]
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"dual_branch ({type(e).__name__}: {e})")

    if "trufor" in selected:
        try:
            from .detectors.trufor import TruForAnalyzer
            tf_r = TruForAnalyzer().analyze(img)
            signals["trufor"] = round(1.0 - tf_r["integrity_score"], 4)
            if "noiseprint_localizer" in selected:
                signals["noiseprint_localizer"] = round(float(tf_r.get("mask_mean", 0.0)), 4)
            if "mask_spatial_concentration" in selected:
                signals["mask_spatial_concentration"] = round(1.0 - tf_r.get("mask_spatial_concentration", 0.3), 4)
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"trufor ({type(e).__name__}: {e})")

    if "srm" in selected:
        try:
            from .detectors.srm import SRMAnalyzer
            srm    = SRMAnalyzer()
            srm_r  = srm.compute_anomaly_score(img)
            jpeg_r = srm.detect_jpeg_grid_inconsistency(img)
            signals["srm"] = round(0.70 * srm_r["score"] + 0.30 * jpeg_r["score"], 4)
            if srm_r.get("residual_map") is not None:
                details["srm"] = {"residual_map": srm_r["residual_map"]}
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"srm ({type(e).__name__}: {e})")

    if "siglip2" in selected:
        try:
            from .detectors.siglip2 import SigLIP2Detector
            signals["siglip2"] = round(SigLIP2Detector().score(img), 4)
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"siglip2 ({type(e).__name__}: {e})")

    if "clip_ufd" in selected:
        try:
            from .detectors.clip_ufd import UFDAdapter
            signals["clip_ufd"] = UFDAdapter().predict(img)["score"]
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"clip_ufd ({type(e).__name__}: {e})")

    if "ela" in selected:
        try:
            from .signals.ela import ela_anomaly_score
            ela_r = ela_anomaly_score(img)
            signals["ela"] = round(float(ela_r["score"]), 4)
            if ela_r.get("heatmap") is not None:
                details["ela"] = {"heatmap": ela_r["heatmap"]}
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"ela ({type(e).__name__}: {e})")

    if "prnu" in selected:
        try:
            from .signals.prnu import prnu_anomaly_score
            r = prnu_anomaly_score(img_arr)
            if r["score"] is not None:
                signals["prnu"] = round(float(r["score"]), 4)
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"prnu ({type(e).__name__}: {e})")

    if "dct_benford" in selected:
        try:
            from .signals.dct_benford import dct_benford_score
            r = dct_benford_score(img_arr)
            if r["score"] is not None:
                signals["dct_benford"] = round(float(r["score"]), 4)
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"dct_benford ({type(e).__name__}: {e})")

    if "cfa" in selected:
        try:
            from .signals.cfa import cfa_correlation_score
            r = cfa_correlation_score(img_arr)
            if r["score"] is not None:
                signals["cfa"] = round(float(r["score"]), 4)
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"cfa ({type(e).__name__}: {e})")

    if "metadata" in selected:
        try:
            from .signals.metadata import analyze_metadata
            signals["metadata"] = round(1.0 - analyze_metadata(img)["authenticity_score"], 4)
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"metadata ({type(e).__name__}: {e})")

    if "dire" in selected:
        try:
            from .detectors.dire import DIREDetector
            r = DIREDetector().predict(img)
            if r["score"] is not None:
                # DIRE: low error = AI-generated; map confidence to fake probability [0,1]
                fake_prob = r["confidence"] if r["label"] == "Fake" else 1.0 - r["confidence"]
                signals["dire"] = round(fake_prob, 4)
                if r.get("error_map") is not None:
                    details["dire"] = {"error_map": r["error_map"]}
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"dire ({type(e).__name__}: {e})")

    if "rigid" in selected:
        try:
            import numpy as _np
            from .detectors.rigid import RIGIDDetector
            r = RIGIDDetector().predict(img)
            if r["score"] is not None:
                # RIGID sensitivity is high=real → convert to fake probability
                fake_prob = float(1.0 / (1.0 + _np.exp((r["score"] - r["threshold"]) * 50)))
                signals["rigid"] = round(fake_prob, 4)
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"rigid ({type(e).__name__}: {e})")

    if "textshield" in selected:
        try:
            from .detectors.textshield import TextShieldDetector
            r = TextShieldDetector().predict(img)
            if r["score"] is not None:
                signals["textshield"] = round(float(r["score"]), 4)
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            skipped.append(f"textshield ({type(e).__name__}: {e})")

    # Use preset calibrated weights if supplied, otherwise equal weights.
    # yolo_damage excluded from scoring — ROI only.
    if preset_weights:
        weights = {k: v for k, v in preset_weights.items() if k in signals}
    else:
        weights = {k: 1.0 for k in selected if k != "yolo_damage"}
    scorer = EnsembleScorer(custom_weights=weights)
    ensemble = scorer.score(signals, mode="fixed")

    return {
        "pipeline":           "custom",
        "verdict":            ensemble["verdict"],
        "score":              ensemble["ensemble_score"],
        "confidence":         ensemble["confidence"],
        "signals":            signals,
        "ensemble":           ensemble,
        "skipped":            skipped,
        "details":            details,
        "roi_crop_applied":   _roi_crop_applied,
        "manual_roi_applied": _manual_roi_applied,
        "manual_roi":         manual_roi,
    }


# ── Signal registry (used by UI) ──────────────────────────────────────────────

ALL_SIGNAL_KEYS: list[str] = [
    "dual_branch",
    "trufor",
    "srm",
    "siglip2",
    "clip_ufd",
    "ela",
    "prnu",
    "dct_benford",
    "cfa",
    "metadata",
    "dire",
    "rigid",
    "textshield",
    "yolo_damage",
]

SIGNAL_LABELS: dict[str, str] = {
    "dual_branch":  "DualBranchModel (RGB+DCT)",
    "trufor":       "TruFor — Pixel Forgery",
    "srm":          "SRM Filter Bank",
    "siglip2":      "SigLIP-2 So400m",
    "clip_ufd":     "CLIP / UniversalFakeDetect",
    "ela":          "Error Level Analysis (ELA)",
    "prnu":         "PRNU Noise Floor",
    "dct_benford":  "DCT / Benford's Law",
    "cfa":          "CFA Demosaicing Correlation",
    "metadata":     "EXIF Metadata Forensics",
    "dire":         "DIRE — Diffusion Recon. Error",
    "rigid":        "RIGID — DINOv2 Perturbation Sensitivity",
    "textshield":   "TextShield — VLM Document Forensics",
    "yolo_damage":  "YOLO Damage Detector (ROI only — not scored)",
}

PRESET_SIGNALS: dict[str, set] = {
    "id_card":        {"trufor", "srm", "dual_branch", "dire", "rigid", "ela", "prnu", "dct_benford", "cfa", "textshield"},
    "document_fraud": {"trufor", "srm", "siglip2", "dire", "rigid", "ela", "prnu", "textshield"},
    "vehicle_damage": {"dual_branch", "trufor", "srm", "metadata", "clip_ufd", "yolo_damage"},
    "custom":         set(ALL_SIGNAL_KEYS),
}

# Signals that require torch — shown with a note in the UI
TORCH_SIGNALS: set[str] = {"dual_branch", "trufor", "siglip2", "clip_ufd", "dire", "rigid", "textshield"}
