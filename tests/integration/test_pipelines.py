"""Integration tests — all three pipelines + custom pipeline + CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synth_image() -> Image.Image:
    """224×224 synthetic image — enough for all detectors."""
    return Image.new("RGB", (224, 224), color=(120, 80, 200))


@pytest.fixture
def tiny_image() -> Image.Image:
    """32×32 — below IMAGE_MIN_SIZE=64, should raise ValueError."""
    return Image.new("RGB", (32, 32), color=(0, 0, 0))


# ── Pipeline result schema validator ─────────────────────────────────────────

def _assert_pipeline_schema(result: dict, pipeline_name: str) -> None:
    """Verify that a pipeline result has the required top-level keys."""
    required = {"pipeline", "verdict", "score", "confidence", "signals", "ensemble", "skipped"}
    missing  = required - set(result.keys())
    assert not missing, f"{pipeline_name} result missing keys: {missing}"
    assert result["pipeline"] == pipeline_name
    assert result["verdict"] in ("Authentic", "Suspicious", "Likely Fraudulent")
    assert 0.0 <= result["score"] <= 1.0
    assert result["confidence"] in ("HIGH", "MEDIUM", "LOW")
    assert isinstance(result["skipped"], list)
    assert isinstance(result["signals"], dict)
    for k, v in result["signals"].items():
        assert 0.0 <= v <= 1.0, f"Signal '{k}' score {v} out of [0, 1]"


# ── ID card pipeline ──────────────────────────────────────────────────────────

def test_id_card_pipeline_schema(synth_image):
    from ai_image_detection.pipelines import run_id_card_pipeline
    result = run_id_card_pipeline(synth_image)
    _assert_pipeline_schema(result, "id_card")


def test_id_card_pipeline_deterministic(synth_image):
    from ai_image_detection.pipelines import run_id_card_pipeline
    r1 = run_id_card_pipeline(synth_image)
    r2 = run_id_card_pipeline(synth_image)
    assert r1["score"]   == r2["score"],   "id_card pipeline is not deterministic"
    assert r1["verdict"] == r2["verdict"], "id_card pipeline verdict is not deterministic"


# ── Document fraud pipeline ───────────────────────────────────────────────────

def test_document_fraud_pipeline_schema(synth_image):
    from ai_image_detection.pipelines import run_document_fraud_pipeline
    result = run_document_fraud_pipeline(synth_image)
    _assert_pipeline_schema(result, "document_fraud")


# ── Vehicle damage pipeline ───────────────────────────────────────────────────

def test_vehicle_damage_pipeline_schema(synth_image):
    from ai_image_detection.pipelines import run_vehicle_damage_pipeline
    result = run_vehicle_damage_pipeline(synth_image)
    _assert_pipeline_schema(result, "vehicle_damage")


# ── Custom pipeline ───────────────────────────────────────────────────────────

def test_custom_pipeline_schema(synth_image):
    from ai_image_detection.pipelines import run_custom_pipeline
    result = run_custom_pipeline(synth_image, selected=["srm", "ela", "prnu"])
    assert result["pipeline"] == "custom"
    assert result["verdict"] in ("Authentic", "Suspicious", "Likely Fraudulent")
    assert 0.0 <= result["score"] <= 1.0


def test_custom_pipeline_empty_selection(synth_image):
    from ai_image_detection.pipelines import run_custom_pipeline
    result = run_custom_pipeline(synth_image, selected=[])
    assert result["score"] == 0.5   # no signals → neutral


# ── Image size validation ─────────────────────────────────────────────────────

def test_id_card_rejects_tiny_image(tiny_image):
    from ai_image_detection.pipelines import run_id_card_pipeline
    with pytest.raises(ValueError, match="too small"):
        run_id_card_pipeline(tiny_image)


def test_document_fraud_rejects_tiny_image(tiny_image):
    from ai_image_detection.pipelines import run_document_fraud_pipeline
    with pytest.raises(ValueError, match="too small"):
        run_document_fraud_pipeline(tiny_image)


def test_vehicle_damage_rejects_tiny_image(tiny_image):
    from ai_image_detection.pipelines import run_vehicle_damage_pipeline
    with pytest.raises(ValueError, match="too small"):
        run_vehicle_damage_pipeline(tiny_image)


# ── Detector cache ────────────────────────────────────────────────────────────

def test_detector_cache_returns_same_instance(synth_image):
    """SRMAnalyzer should be reused across consecutive pipeline calls."""
    from ai_image_detection.pipelines import _DETECTOR_CACHE, run_id_card_pipeline
    run_id_card_pipeline(synth_image)
    srm_a = _DETECTOR_CACHE.get("srm")
    run_id_card_pipeline(synth_image)
    srm_b = _DETECTOR_CACHE.get("srm")
    assert srm_a is srm_b, "SRMAnalyzer should be cached across calls"


# ── CLI JSON output ───────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (Path(__file__).parent.parent.parent / "main.py").exists(),
    reason="main.py not found",
)
def test_cli_json_output_valid(tmp_path):
    """CLI --json flag must produce parseable JSON."""
    img = Image.new("RGB", (224, 224), color=(128, 64, 192))
    img_path = tmp_path / "test.png"
    img.save(img_path)

    proc = subprocess.run(
        [sys.executable, "main.py", "--image", str(img_path), "--pipeline", "id_card", "--json"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert proc.returncode == 0, f"CLI exited with {proc.returncode}: {proc.stderr}"
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"CLI --json output is not valid JSON: {exc}\nOutput: {proc.stdout[:500]}")

    assert "verdict" in parsed
    assert "score"   in parsed
