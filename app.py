"""
AI Image Detection — Streamlit web app.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import streamlit as st
from PIL import Image, ImageDraw

from ai_image_detection.pipelines import (
    ALL_SIGNAL_KEYS,
    PRESET_SIGNALS,
    SIGNAL_LABELS,
    TORCH_SIGNALS,
    run_custom_pipeline,
    run_vehicle_damage_pipeline,
)
from ai_image_detection.config import (
    ID_CARD_WEIGHTS,
    DOCUMENT_FRAUD_WEIGHTS,
    VEHICLE_DAMAGE_WEIGHTS,
)
from ai_image_detection.ui.components import (
    signal_bar_chart,
    signal_breakdown_table,
    skipped_signals_warning,
    verdict_banner,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Image Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Drawing helpers ───────────────────────────────────────────────────────────

_MAX_DISP_W = 680
_MAX_DISP_H = 460


def _cap_display(img: Image.Image) -> Image.Image:
    """Scale img down to fit _MAX_DISP_W × _MAX_DISP_H; never upscales."""
    scale = min(_MAX_DISP_W / img.width, _MAX_DISP_H / img.height, 1.0)
    if scale >= 1.0:
        return img
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)


def _draw_manual_roi(img: Image.Image, roi: dict) -> Image.Image:
    """Blue semi-transparent ROI rectangle (ported from UAIC framework)."""
    x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    out     = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    od      = ImageDraw.Draw(overlay)
    od.rectangle([x, y, x + w, y + h], fill=(59, 130, 246, 35))
    result = Image.alpha_composite(out, overlay)
    draw   = ImageDraw.Draw(result)
    draw.rectangle([x, y, x + w, y + h], outline=(59, 130, 246, 255), width=3)
    draw.text((x + 4, y + 4), f"{w}×{h} px", fill=(59, 130, 246, 230))
    return result.convert("RGB")


def _draw_damage_boxes(img: Image.Image, detections: list[dict]) -> Image.Image:
    """Orange bounding boxes with confidence badge (ported from UAIC framework)."""
    overlay   = img.copy().convert("RGBA")
    draw      = ImageDraw.Draw(overlay)
    highlight = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    hd        = ImageDraw.Draw(highlight)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        conf  = det["confidence"]
        label = det.get("label", "damage")
        lw    = 4 if i == 0 else 2
        draw.rectangle([x1, y1, x2, y2], outline=(249, 115, 22, 255), width=lw)
        hd.rectangle([x1, y1, x2, y2], fill=(249, 115, 22, 25))
        badge = f"{label} {conf:.0%}"
        bw    = len(badge) * 7 + 6
        draw.rectangle([x1, y1 - 20, x1 + bw, y1], fill=(249, 115, 22, 220))
        draw.text((x1 + 3, y1 - 17), badge, fill=(255, 255, 255, 255))
    return Image.alpha_composite(overlay, highlight).convert("RGB")


def _draw_roi_overlays(
    img: Image.Image,
    composite: dict | None,
    expanded: dict | None,
) -> Image.Image:
    """Orange composite border + cyan expanded ROI border."""
    out  = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    if composite:
        x, y, w, h = composite["x"], composite["y"], composite["width"], composite["height"]
        draw.rectangle([x, y, x + w, y + h], outline=(249, 115, 22, 255), width=3)
        draw.text((x + 4, y + 4), f"damage region  {w}×{h}px", fill=(249, 115, 22, 230))
    if expanded:
        x, y, w, h = expanded["x"], expanded["y"], expanded["width"], expanded["height"]
        draw.rectangle([x, y, x + w, y + h], outline=(34, 211, 238, 255), width=2)
        draw.text((x + 4, y + h - 16), f"analysis ROI  {w}×{h}px", fill=(34, 211, 238, 230))
    return out.convert("RGB")


def _cluster_detections(
    detections: list[dict],
    img_w: int,
    img_h: int,
    proximity_ratio: float = 0.25,
) -> list[list[dict]]:
    """Union-find 2D clustering by centroid proximity (ported from UAIC framework)."""
    n = len(detections)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    threshold = proximity_ratio * max(img_w, img_h)
    centers   = [
        ((d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) / 2,
         (d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2)
        for d in detections
    ]
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if (dx * dx + dy * dy) ** 0.5 <= threshold:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj

    groups: dict[int, list] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(detections[i])

    return sorted(groups.values(), key=lambda g: min(d["bbox_xyxy"][0] for d in g))


# ── Sidebar ───────────────────────────────────────────────────────────────────

_PIPELINE_LABELS = {
    "id_card":        "🪪  ID Card",
    "document_fraud": "📄  Document Fraud",
    "vehicle_damage": "🚗  Vehicle Damage",
    "custom":         "⚙️  Custom",
}

_PIPELINE_WEIGHTS = {
    "id_card":        ID_CARD_WEIGHTS,
    "document_fraud": DOCUMENT_FRAUD_WEIGHTS,
    "vehicle_damage": VEHICLE_DAMAGE_WEIGHTS,
    "custom":         None,
}

with st.sidebar:
    st.title("🔍 AI Image Detection")
    st.caption("Forensic image analysis framework")
    st.divider()

    pipeline = st.selectbox(
        "Pipeline",
        options=list(_PIPELINE_LABELS),
        format_func=_PIPELINE_LABELS.get,
        key="pipeline",
    )

    # Reset signal selection when pipeline changes
    if st.session_state.get("_last_pipeline") != pipeline:
        st.session_state["_last_pipeline"] = pipeline
        st.session_state["active_signals"] = list(PRESET_SIGNALS.get(pipeline, set(ALL_SIGNAL_KEYS)))

    st.markdown("**Active models**")
    _all_options    = ALL_SIGNAL_KEYS
    _default        = st.session_state.get("active_signals", list(PRESET_SIGNALS.get(pipeline, [])))
    _signal_labels  = {k: SIGNAL_LABELS.get(k, k) for k in _all_options}

    active_signals: list[str] = st.multiselect(
        "Select models to run",
        options=_all_options,
        default=_default,
        format_func=lambda k: (
            f"⚡ {_signal_labels[k]}" if k in TORCH_SIGNALS else _signal_labels[k]
        ),
        key="active_signals",
        help="⚡ = requires PyTorch / GPU weights. Preset signals are pre-selected; add or remove freely.",
    )

    use_damage_roi = False
    if pipeline == "vehicle_damage" and "yolo_damage" in active_signals:
        use_damage_roi = st.toggle(
            "Focus on damage ROI",
            value=False,
            help="All non-YOLO signals run on the 1.4× expanded damage crop.",
        )

    st.divider()
    st.markdown(
        """
        **Verdict thresholds**
        - 🟢 < 0.35 → Authentic
        - 🟡 0.35–0.60 → Suspicious
        - 🔴 ≥ 0.60 → Likely Fraudulent
        """
    )
    st.caption("⚡ = requires PyTorch / GPU weights")

# ── Main panel ────────────────────────────────────────────────────────────────

st.header("Image Forensics Analyser")

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("Image")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        help="Supported: JPEG, PNG, WebP, BMP, TIFF",
    )

    if uploaded is None:
        st.info("Upload an image to begin.")
        with col_right:
            st.subheader("Result")
            st.info("Run analysis to see results here.")
        st.stop()

    img_bytes = uploaded.read()
    img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    _W, _H    = img.width, img.height

    st.image(_cap_display(img), caption=f"{uploaded.name} — {_W}×{_H} px", use_container_width=True)

    with st.expander("Image details", expanded=False):
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("Width",     f"{_W} px")
        _m2.metric("Height",    f"{_H} px")
        _m3.metric("Mode",      img.mode)
        _m4.metric("File size", f"{len(img_bytes)/1024:.1f} KB")

    st.divider()

    # ── Region of Interest ────────────────────────────────────────────────────

    _manual_roi: dict | None = None

    with st.expander("📐 Region of Interest", expanded=False):
        _roi_enabled = st.checkbox(
            "Enable ROI mode",
            key="roi_enabled",
            value=False,
            help="Draw a region then click 'Run on ROI' to focus analysis on that area only.",
        )

        if _roi_enabled:
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates as _sic
                st.caption("Click and drag on the image below to select a region.")
                _disp   = _cap_display(img)
                _coords = _sic(_disp, click_and_drag=True, cursor="crosshair", key="roi_drag")

                if _coords and "x1" in _coords:
                    _dw = _coords.get("width",  _disp.width)
                    _dh = _coords.get("height", _disp.height)
                    _sx, _sy = _W / _dw, _H / _dh
                    _rx1 = max(0,  int(min(_coords["x1"], _coords["x2"]) * _sx))
                    _ry1 = max(0,  int(min(_coords["y1"], _coords["y2"]) * _sy))
                    _rx2 = min(_W, int(max(_coords["x1"], _coords["x2"]) * _sx))
                    _ry2 = min(_H, int(max(_coords["y1"], _coords["y2"]) * _sy))
                    if _rx2 - _rx1 >= 32 and _ry2 - _ry1 >= 32:
                        st.session_state.update(
                            roi_x1=_rx1, roi_y1=_ry1, roi_x2=_rx2, roi_y2=_ry2
                        )
            except ImportError:
                st.warning(
                    "Install `streamlit-image-coordinates` for canvas drawing:  \n"
                    "`pip install streamlit-image-coordinates`  \n"
                    "Use the sliders below in the meantime."
                )

            _sx1 = st.session_state.get("roi_x1", _W // 4)
            _sy1 = st.session_state.get("roi_y1", _H // 4)
            _sx2 = st.session_state.get("roi_x2", 3 * _W // 4)
            _sy2 = st.session_state.get("roi_y2", 3 * _H // 4)

            with st.expander("Fine-tune with sliders", expanded=False):
                _ft1, _ft2 = st.columns(2)
                with _ft1:
                    _sx1 = st.slider("Left (x)",   0,          _W - 1, _sx1,                 key="roi_x1")
                    _sy1 = st.slider("Top (y)",    0,          _H - 1, _sy1,                 key="roi_y1")
                with _ft2:
                    _sx2 = st.slider("Right (x)",  _sx1 + 32,  _W,     max(_sx2, _sx1 + 32), key="roi_x2")
                    _sy2 = st.slider("Bottom (y)", _sy1 + 32,  _H,     max(_sy2, _sy1 + 32), key="roi_y2")

            _manual_roi = {"x": _sx1, "y": _sy1, "width": _sx2 - _sx1, "height": _sy2 - _sy1}

            st.image(
                _draw_manual_roi(img, _manual_roi),
                caption=f"Selected ROI — {_manual_roi['width']}×{_manual_roi['height']} px "
                        f"at ({_sx1}, {_sy1})",
                use_container_width=True,
            )
        else:
            st.caption("Enable ROI mode to draw a region for focused analysis.")

    st.divider()

    # ── Run buttons ───────────────────────────────────────────────────────────

    _run_disabled = len(active_signals) == 0
    _btn1, _btn2, _ = st.columns([2, 2, 3])
    with _btn1:
        run_btn = st.button(
            "▶  Run Analysis",
            type="primary",
            disabled=_run_disabled,
            use_container_width=True,
            help="Analyse the full image",
        )
    with _btn2:
        roi_run_btn = st.button(
            "▶  Run on ROI",
            type="secondary",
            disabled=(_manual_roi is None),
            use_container_width=True,
            help="Analyse only the selected region of interest",
        )

    if not run_btn and not roi_run_btn:
        with col_right:
            st.subheader("Result")
            st.info("Click 'Run Analysis' to start.")
        st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────

_roi_for_run = _manual_roi if roi_run_btn else None

with st.spinner("Running forensic analysis… (first run loads ML models, may take 30–60 s)"):
    try:
        result = run_custom_pipeline(
            img,
            selected=active_signals,
            use_damage_roi=use_damage_roi,
            manual_roi=_roi_for_run,
            preset_weights=_PIPELINE_WEIGHTS.get(pipeline),
        )
    except Exception as exc:
        # M-8: log full traceback to console/server logs; show clean message in UI
        logger.exception("Pipeline failed with unhandled exception")
        st.error(f"Pipeline error — {type(exc).__name__}: {exc}")
        st.stop()

# ── Results ───────────────────────────────────────────────────────────────────

with col_right:
    st.subheader("Result")

    verdict_banner(result["verdict"], result["score"], result["confidence"])

    if result.get("manual_roi_applied"):
        _mr = result.get("manual_roi", {})
        st.info(
            f"Scored on **manual ROI** — "
            f"{_mr.get('width','?')}×{_mr.get('height','?')} px "
            f"at ({_mr.get('x','?')}, {_mr.get('y','?')})"
        )
    elif result.get("roi_crop_applied"):
        _exp_r = result.get("details", {}).get("yolo_damage", {}).get("expanded_roi", {})
        st.info(
            f"Scored on **YOLO damage ROI** — "
            f"{_exp_r.get('width','?')}×{_exp_r.get('height','?')} px"
        )

    with st.expander("📊 Signal Scores", expanded=True):
        signal_bar_chart(result["ensemble"].get("signal_breakdown", []))

    with st.expander("📋 Signal Breakdown", expanded=False):
        signal_breakdown_table(result["ensemble"].get("signal_breakdown", []))
        skipped_signals_warning(result.get("skipped", []))

    with st.expander("🗂️ Raw JSON", expanded=False):
        import json
        import numpy as np

        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return "<ndarray>"
            return obj

        st.json(_clean(result))

# ── YOLO Vehicle Damage Visualisation ─────────────────────────────────────────

_yolo_d = result.get("details", {}).get("yolo_damage")
if _yolo_d is not None:
    detections    = _yolo_d.get("detections", [])
    composite_roi = _yolo_d.get("composite_roi")
    expanded_roi  = _yolo_d.get("expanded_roi")

    with st.expander("🚗 Vehicle Damage Detection", expanded=bool(detections)):
        if not detections:
            st.info("YOLO detected no damage regions in this image.")
        else:
            st.markdown(f"**{len(detections)} damage region(s) detected**")

            st.table([
                {
                    "Label":             d.get("label", "—"),
                    "Confidence":        f"{d['confidence']:.1%}",
                    "Bbox (x1,y1,x2,y2)": str(d["bbox_xyxy"]),
                }
                for d in detections
            ])

            _img_boxes = _draw_damage_boxes(img, detections)
            _img_roi   = _draw_roi_overlays(_img_boxes, composite_roi, expanded_roi)

            _vc1, _vc2 = st.columns(2)
            with _vc1:
                st.markdown("**Damage Bounding Boxes**")
                st.image(_img_boxes, use_container_width=True)
            with _vc2:
                st.markdown("**Region of Interest Overlay**")
                st.image(_img_roi, use_container_width=True)
                if composite_roi:
                    st.caption(
                        f"Orange = damage region "
                        f"({composite_roi['width']}×{composite_roi['height']} px) | "
                        "Cyan = 1.4× analysis ROI"
                    )

            _clusters = _cluster_detections(detections, img.width, img.height)
            if len(_clusters) > 1:
                st.markdown(f"**{len(_clusters)} damage cluster(s)**")
                _cl_cols = st.columns(min(len(_clusters), 3))
                for _col, _cluster in zip(_cl_cols, _clusters):
                    with _col:
                        _bx1 = min(d["bbox_xyxy"][0] for d in _cluster)
                        _by1 = min(d["bbox_xyxy"][1] for d in _cluster)
                        _bx2 = max(d["bbox_xyxy"][2] for d in _cluster)
                        _by2 = max(d["bbox_xyxy"][3] for d in _cluster)
                        _pad = 20
                        _crop_b = (
                            max(0, _bx1 - _pad), max(0, _by1 - _pad),
                            min(img.width,  _bx2 + _pad),
                            min(img.height, _by2 + _pad),
                        )
                        _cl_img = _draw_damage_boxes(img, _cluster).crop(_crop_b)
                        _lbl    = ", ".join(d.get("label", "damage") for d in _cluster)
                        st.image(_cl_img, caption=_lbl, use_container_width=True)

# ── Explainability ────────────────────────────────────────────────────────────

if "dual_branch" in result.get("signals", {}):
    with st.expander("🔬 Explainability — DualBranch", expanded=False):
        try:
            import torch
            from ai_image_detection.detectors.dual_branch import DualBranchDetector
            from ai_image_detection.explainability.gradcam import generate_gradcam, overlay_heatmap
            from ai_image_detection.explainability.heatmap import build_evidence_panel

            _det    = DualBranchDetector()
            _model  = _det._model
            _device = torch.device("cpu")
            _label  = result["verdict"]

            with st.spinner("Generating explainability panels…"):
                _hm_bgr, _tcls, _tcls_conf          = generate_gradcam(_model, img, _device)
                _gradcam_img                          = overlay_heatmap(img, _hm_bgr, alpha=0.5)
                _, _patch_grid, _dct_color, _stats   = build_evidence_panel(_model, img, _device, _label)

            _ec1, _ec2, _ec3 = st.columns(3)
            with _ec1:
                st.markdown("**Grad-CAM**")
                st.image(_gradcam_img, use_container_width=True)
                st.caption(f"Class: {'Fake' if _tcls == 0 else 'Real'} ({_tcls_conf:.2%})")
            with _ec2:
                st.markdown("**Patch Verdict Grid**")
                st.image(_patch_grid, use_container_width=True)
                _flagged = _stats.get("flagged_patches", 0)
                _total   = _stats.get("total_patches", 1)
                st.caption(f"{_flagged}/{_total} patches flagged")
            with _ec3:
                st.markdown("**DCT Artifact Map**")
                st.image(_dct_color, use_container_width=True)
                st.caption("Smooth = AI-generated · Noisy = real photo")

        except Exception as _e:
            st.info(f"DualBranch weights required for explainability panels. ({_e})")
