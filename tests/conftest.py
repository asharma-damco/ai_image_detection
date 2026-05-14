"""Shared pytest fixtures for the ai_image_detection test suite."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


SAMPLES_DIR = Path(__file__).parent.parent / "samples"


@pytest.fixture
def sample_image() -> Image.Image:
    """Return a small RGB PIL image for fast unit tests."""
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def sample_image_path() -> Path:
    """Return path to the first available sample image, or None."""
    candidates = list(SAMPLES_DIR.glob("*.jpg")) + list(SAMPLES_DIR.glob("*.png"))
    if not candidates:
        pytest.skip("No sample images found in samples/")
    return candidates[0]
