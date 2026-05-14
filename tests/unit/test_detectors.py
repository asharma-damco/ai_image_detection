"""Unit tests for individual detector modules."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ai_image_detection.detectors.srm import SRMAnalyzer
from ai_image_detection.detectors.base import BaseDetector


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_image() -> Image.Image:
    """8×8 image — smaller than SRM block_size=16."""
    return Image.new("RGB", (8, 8), color=(200, 100, 50))


@pytest.fixture
def normal_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def large_image() -> Image.Image:
    return Image.new("RGB", (512, 512), color=(80, 160, 240))


# ── SRMAnalyzer ───────────────────────────────────────────────────────────────

class TestSRMAnalyzer:
    def test_predict_normal_image(self, normal_image):
        srm = SRMAnalyzer()
        result = srm.predict(normal_image)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert result["label"] in ("Fake", "Real")
        assert "confidence" in result

    def test_compute_anomaly_score_normal(self, normal_image):
        srm = SRMAnalyzer()
        result = srm.compute_anomaly_score(normal_image)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert "residual_map" in result
        assert isinstance(result["residual_map"], np.ndarray)

    def test_small_image_guard(self, small_image):
        """Images smaller than block_size must return neutral score, not crash."""
        srm = SRMAnalyzer(block_size=16)
        result = srm.compute_anomaly_score(small_image)
        assert result["score"] == 0.5
        assert result["residual_map"].shape == (8, 8)

    def test_extract_residuals_shape(self, normal_image):
        srm = SRMAnalyzer()
        residuals = srm.extract_residuals(normal_image)
        H, W = normal_image.size[1], normal_image.size[0]
        assert residuals.shape == (H, W, len(srm.kernels))

    def test_detect_jpeg_grid(self, large_image):
        srm = SRMAnalyzer()
        result = srm.detect_jpeg_grid_inconsistency(large_image)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_large_image(self, large_image):
        srm = SRMAnalyzer()
        result = srm.predict(large_image)
        assert 0.0 <= result["score"] <= 1.0


# ── BaseDetector ABC ──────────────────────────────────────────────────────────

class TestBaseDetectorContract:
    def test_cannot_instantiate_abstract(self):
        """BaseDetector must be abstract — direct instantiation raises TypeError."""
        with pytest.raises(TypeError):
            BaseDetector()

    def test_concrete_must_implement_predict(self):
        """Subclasses that don't implement predict() must raise TypeError."""
        class _Incomplete(BaseDetector):
            pass

        with pytest.raises(TypeError):
            _Incomplete()


# ── RIGIDDetector (model-free paths) ─────────────────────────────────────────

class TestRIGIDDetectorSafePaths:
    def test_unavailable_returns_none_score(self, normal_image):
        """When DINOv2 is unavailable, RIGID returns score=None and verdict=unavailable."""
        from ai_image_detection.detectors.rigid import RIGIDDetector
        det = RIGIDDetector.__new__(RIGIDDetector)
        det.n_perturbations = 3
        det.noise_std       = 0.05
        det._device_override = None
        det._model  = None
        det._device = None
        det._get_model = lambda: (None, None)

        result = det.predict(normal_image)
        assert result["score"] is None
        assert result["verdict"] == "unavailable"
        assert result["label"]   == "Unknown"

    def test_n_perturbations_default(self):
        from ai_image_detection.detectors.rigid import RIGIDDetector, _N_PERTURB
        assert _N_PERTURB == 3, "Default perturbations should be 3 (reduced from 10 for CPU)"


# ── DIREDetector (model-free paths) ──────────────────────────────────────────

class TestDIREDetectorSafePaths:
    def test_cpu_returns_unavailable(self, normal_image):
        """On CPU, DIRE should skip gracefully without attempting to load pipeline."""
        from ai_image_detection.detectors.dire import DIREDetector
        import torch
        det = DIREDetector.__new__(DIREDetector)
        det._loaded      = False
        det._flux_loaded = False
        det._flux_pipe   = None
        det.pipe         = None
        det._model_id    = DIREDetector.DEFAULT_MODEL
        det.num_inversion_steps = 10
        det.device       = torch.device("cpu")
        det.dtype        = torch.float32

        result = det.predict(normal_image)
        assert result["score"] is None
        assert result["verdict"] == "unavailable"
        assert result["backbone_used"] == "none"

    def test_flux_heuristic_1024(self):
        """1024×1024 no-EXIF image should be suspected as Flux-generated."""
        from ai_image_detection.detectors.dire import DIREDetector
        det = object.__new__(DIREDetector)
        img = Image.new("RGB", (1024, 1024))
        assert DIREDetector._is_flux_suspected(det, img) is True

    def test_flux_heuristic_800x600(self):
        """800×600 image should NOT be suspected as Flux-generated."""
        from ai_image_detection.detectors.dire import DIREDetector
        det = object.__new__(DIREDetector)
        img = Image.new("RGB", (800, 600))
        assert DIREDetector._is_flux_suspected(det, img) is False

    def test_singleton_returns_same_instance(self):
        from ai_image_detection.detectors import dire
        a = dire.get_dire_detector()
        b = dire.get_dire_detector()
        assert a is b


# ── TextShieldDetector (model-free paths) ─────────────────────────────────────

class TestTextShieldSafePaths:
    def test_unavailable_returns_none(self, normal_image):
        from ai_image_detection.detectors.textshield import TextShieldDetector
        det = TextShieldDetector.__new__(TextShieldDetector)
        det.available  = False
        det._model     = None
        det._processor = None
        det._device    = None

        result = det.detect(normal_image, domain="invoice")
        assert result["score"] is None
        assert result["verdict"] == "unavailable"

    def test_parse_score_explicit_label(self):
        from ai_image_detection.detectors.textshield import TextShieldDetector
        det = TextShieldDetector.__new__(TextShieldDetector)
        assert det._parse_score("tampering likelihood 0.82") == 0.82

    def test_parse_score_no_number(self):
        from ai_image_detection.detectors.textshield import TextShieldDetector
        det = TextShieldDetector.__new__(TextShieldDetector)
        assert det._parse_score("no numbers here at all") == 0.5

    def test_parse_score_one(self):
        from ai_image_detection.detectors.textshield import TextShieldDetector
        det = TextShieldDetector.__new__(TextShieldDetector)
        assert det._parse_score("score: 1.0") == 1.0
