"""TextShield-R1 — Qwen2.5-VL document tamper detection.

Reference: github.com/qcf-568/TextShield
Backbone: Qwen/Qwen2.5-VL-7B-Instruct (zero-shot, no fine-tuning required).

Designed for document domains (invoice, ID) where language-vision reasoning
about field consistency, font uniformity, and layout is more informative than
purely pixel-level forensics.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Optional

import torch
from PIL import Image

from .base import BaseDetector

logger = logging.getLogger(__name__)

_MODEL_ID       = "Qwen/Qwen2.5-VL-7B-Instruct"
_THRESHOLD      = 0.5
_MAX_NEW_TOKENS = 300

# ── Module-level singleton ────────────────────────────────────────────────────
_TS_MODEL: Optional[object]    = None
_TS_PROCESSOR: Optional[object] = None
_TS_DEVICE: Optional[str]      = None
_TS_LOAD_FAILED: bool          = False

_PROMPTS: dict[str, str] = {
    "invoice": (
        "Examine this invoice for signs of digital tampering. "
        "Check numeric field consistency, font uniformity, and text alignment. "
        "List any anomalies found and rate tampering likelihood 0-1."
    ),
    "id_document": (
        "Examine this ID document for signs of forgery. "
        "Check text sharpness, font consistency, and field alignment. "
        "Rate forgery likelihood 0-1."
    ),
    "default": (
        "Examine this document image for signs of digital manipulation or forgery. "
        "Assess text consistency, layout integrity, and visual artifacts. "
        "Rate tampering likelihood 0-1."
    ),
}


def _load_textshield_model():
    global _TS_MODEL, _TS_PROCESSOR, _TS_DEVICE, _TS_LOAD_FAILED

    if _TS_MODEL is not None:
        return _TS_MODEL, _TS_PROCESSOR, _TS_DEVICE
    if _TS_LOAD_FAILED:
        return None, None, None

    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning(
                "TextShield: Qwen2.5-VL-7B requires a CUDA GPU. "
                "Refusing to load on CPU (would take ~10 min/image). "
                "Set CUDA_VISIBLE_DEVICES or run on a GPU host to enable TextShield."
            )
            _TS_LOAD_FAILED = True
            return None, None, None

        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        try:
            model = AutoModelForVision2Seq.from_pretrained(
                _MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
            )
        except Exception:
            # Explicit class fallback for older transformers builds
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                _MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
            )

        processor = AutoProcessor.from_pretrained(_MODEL_ID)
        model.eval()

        _TS_MODEL     = model
        _TS_PROCESSOR = processor
        _TS_DEVICE    = device
        logger.info("TextShield: Qwen2.5-VL-7B loaded on %s", device)
        return _TS_MODEL, _TS_PROCESSOR, _TS_DEVICE

    except Exception as exc:
        _TS_LOAD_FAILED = True
        logger.warning("TextShield: model load failed — detector unavailable: %s", exc)
        return None, None, None


class TextShieldDetector(BaseDetector):
    """Document tamper detector using Qwen2.5-VL language-vision reasoning.

    Suited for invoice and ID document domains where semantic understanding
    of field layout and font consistency aids forgery detection.
    """

    def __init__(self) -> None:
        model, processor, device = _load_textshield_model()
        self.available  = model is not None
        # References to singleton — no per-instance storage of weights
        self._model     = model
        self._processor = processor
        self._device    = device

    def _build_prompt(self, domain: str) -> str:
        """Return the domain-specific forensic prompt."""
        return _PROMPTS.get(domain, _PROMPTS["default"])

    def _parse_score(self, text: str) -> float:
        """Extract a [0, 1] float from model output via progressive regex fallback."""
        # Pattern 1: explicit label + value, e.g. "likelihood: 0.75"
        m = re.search(
            r'(?:likelihood|score|probability|rating|tamper)[^\d]{0,10}([01]\.?\d*)',
            text, re.IGNORECASE,
        )
        if m:
            val = float(m.group(1))
            if 0.0 <= val <= 1.0:
                return round(val, 4)

        # Pattern 2: standalone decimal, e.g. "0.82" or "1.0"
        for tok in re.findall(r'\b(0\.\d+|1\.0|1)\b', text):
            val = float(tok)
            if 0.0 <= val <= 1.0:
                return round(val, 4)

        # Pattern 3: any number that could be a decimal score
        for tok in re.findall(r'\b\d+(?:\.\d+)?\b', text):
            val = float(tok)
            if 0.0 <= val <= 1.0:
                return round(val, 4)

        logger.warning("TextShield: could not parse score from output — defaulting to 0.5")
        return 0.5

    @torch.no_grad()
    def detect(
        self,
        img: Image.Image,
        domain: str = "invoice",
        threshold: float = _THRESHOLD,
        **kwargs,
    ) -> dict:
        """Run TextShield forensic analysis.

        Args:
            img:       Input PIL image.
            domain:    "invoice" | "id_document" | any (falls to default prompt).
            threshold: Score >= threshold → "tampered".

        Returns:
            score      float — 0-1, high = tampered / forged
            verdict    str   — "tampered" | "authentic" | "unavailable"
            reasoning  str   — model's text explanation
            domain     str   — domain used
            threshold  float — threshold used
            label      str   — "Fake" | "Real" | "Unknown"
            confidence float — probability of predicted class
        """
        if not self.available:
            return {
                "score":      None,
                "verdict":    "unavailable",
                "reasoning":  "Model not loaded.",
                "domain":     domain,
                "threshold":  threshold,
                "label":      "Unknown",
                "confidence": 0.0,
            }

        prompt  = self._build_prompt(domain)
        img_rgb = img.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            text_input = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text_input],
                images=[img_rgb],
                return_tensors="pt",
            ).to(self._device)

            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
            )
            # Strip prompt tokens — only decode the newly generated portion
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            reasoning = self._processor.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()

        except Exception as exc:
            logger.warning("TextShield: inference failed: %s", exc)
            return {
                "score":      None,
                "verdict":    "unavailable",
                "reasoning":  str(exc),
                "domain":     domain,
                "threshold":  threshold,
                "label":      "Unknown",
                "confidence": 0.0,
            }

        score      = self._parse_score(reasoning)
        verdict    = "tampered" if score >= threshold else "authentic"
        label      = "Fake" if verdict == "tampered" else "Real"
        confidence = round(score if label == "Fake" else 1.0 - score, 4)

        return {
            "score":      score,
            "verdict":    verdict,
            "reasoning":  reasoning,
            "domain":     domain,
            "threshold":  threshold,
            "label":      label,
            "confidence": confidence,
        }

    def predict(self, img: Image.Image, **kwargs) -> dict:
        """BaseDetector-required entry point. Delegates to detect()."""
        return self.detect(img, **kwargs)


if __name__ == "__main__":
    import sys

    # Smoke test 1: unavailable model → graceful None result
    det = TextShieldDetector.__new__(TextShieldDetector)
    det.available  = False
    det._model     = None
    det._processor = None
    det._device    = None

    r = det.detect(Image.new("RGB", (64, 64)), domain="invoice")
    assert r["score"] is None
    assert r["verdict"] == "unavailable"

    # Smoke test 2: prompt builder
    det2 = TextShieldDetector.__new__(TextShieldDetector)
    assert "0-1" in det2._build_prompt("invoice")
    assert "0-1" in det2._build_prompt("id_document")
    assert "0-1" in det2._build_prompt("unknown_domain")

    # Smoke test 3: score parser
    det3 = TextShieldDetector.__new__(TextShieldDetector)
    assert det3._parse_score("tampering likelihood 0.82") == 0.82
    assert det3._parse_score("no numbers here at all") == 0.5
    assert det3._parse_score("score: 1.0") == 1.0

    print("TextShield scaffold OK")
    sys.exit(0)
