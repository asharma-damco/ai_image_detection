"""
SigLIP-2 So400m Document AI-Detection Detector.

Uses image-text contrastive scoring with 8 prompts (4 authentic + 4 fake)
to estimate fake probability. Uses sigmoid activation (NOT softmax) as
required by SigLIP-2's training objective.

Falls back to CLIP ViT-L/14 (UniversalFakeDetect) if SigLIP-2 is unavailable.

Source: UAIC uaic-fraud-detection/poc/siglip2_detector.py
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import torch
from PIL import Image

from ..config import UFD_REPO

logger = logging.getLogger(__name__)

_FULL_MODEL_ID = "google/siglip2-so400m-patch14-384"
_ZONE_MODEL_ID = "google/siglip2-so400m-patch16-naflex"

_AUTHENTIC_PROMPTS = [
    "this is a photo of a real photograph of an official government identity document",
    "this is a photo of a genuine authentic passport scan or id card",
    "this is a photo of a legitimate official photograph of an identity document",
    "this is a photo of a real scanned government issued certificate",
]
_FAKE_PROMPTS = [
    "this is a photo of a computer generated fake artificial identity document",
    "this is a photo of an ai synthesized fake passport or identity card",
    "this is a photo of a digitally manipulated fraudulent government document",
    "this is a photo of a synthetically created artificial fake id card",
]


class SigLIP2Detector:
    """
    SigLIP-2 So400m wrapper for document / ID card fraud detection.

    Loads two models:
      - patch14-384  : full-image / full-document mode
      - patch16-naflex : variable-resolution zone crop mode (lazy-loaded)

    Falls back to CLIP ViT-L/14 when SigLIP-2 weights are absent.
    """

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._full_model     = None
        self._full_processor = None
        self._zone_model     = None
        self._zone_processor = None
        self.fallback_to_clip = False
        self._fallback_error  = ""
        self._zone_loaded     = False
        self._last_probs:  list = []
        self._last_n_auth: int  = 4

        self._load_full_model()

    def _load_full_model(self) -> None:
        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError:
            self.fallback_to_clip = True
            self._fallback_error  = "transformers>=4.50.0 required. pip install 'transformers>=4.50.0'"
            logger.warning(self._fallback_error)
            return

        try:
            logger.info("Loading SigLIP-2 full model: %s", _FULL_MODEL_ID)
            self._full_processor = AutoProcessor.from_pretrained(_FULL_MODEL_ID, padding="max_length")
            self._full_model     = AutoModel.from_pretrained(_FULL_MODEL_ID).to(self.device)
            self._full_model.eval()
        except Exception as exc:
            self.fallback_to_clip = True
            self._fallback_error  = f"SigLIP-2 load failed: {exc}"
            logger.warning(self._fallback_error)

    def _load_zone_model(self) -> None:
        if self._zone_loaded:
            return
        self._zone_loaded = True
        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore
            logger.info("Lazy-loading SigLIP-2 NaFlex model: %s", _ZONE_MODEL_ID)
            self._zone_processor = AutoProcessor.from_pretrained(_ZONE_MODEL_ID, padding="max_length")
            self._zone_model     = AutoModel.from_pretrained(_ZONE_MODEL_ID).to(self.device)
            self._zone_model.eval()
        except Exception as exc:
            logger.warning("SigLIP-2 NaFlex load failed (%s); zone mode uses full model.", exc)
            self._zone_model     = self._full_model
            self._zone_processor = self._full_processor

    def score(self, image: Image.Image, mode: str = "full") -> float:
        """Return fake_probability [0, 1] for the given image.

        Args:
            image: PIL Image (full document or zone crop).
            mode:  "full" (patch14-384) or "zone" (patch16-naflex).

        Returns 0.5 (neutral) on any error.
        """
        if self.fallback_to_clip:
            return self._clip_fallback(image, mode)
        try:
            return self._siglip2_score(image, mode)
        except Exception as exc:
            logger.warning("SigLIP-2 scoring failed (%s); activating CLIP fallback.", exc)
            self.fallback_to_clip = True
            self._fallback_error  = str(exc)
            return self._clip_fallback(image, mode)

    def score_detailed(self, image: Image.Image, mode: str = "full") -> dict:
        """Return full result dict including per-category breakdown."""
        self._last_probs  = []
        self._last_n_auth = 4
        fake_prob = self.score(image, mode)
        method    = "clip-fallback" if self.fallback_to_clip else f"siglip2-{mode}"
        result    = {
            "fake_prob":  round(fake_prob, 4),
            "real_prob":  round(1.0 - fake_prob, 4),
            "label":      "Fake" if fake_prob >= 0.5 else "Real",
            "confidence": round(max(fake_prob, 1.0 - fake_prob), 4),
            "score":      round(fake_prob, 4),
            "method":     method,
        }
        if not self.fallback_to_clip and len(self._last_probs) >= 8:
            p       = self._last_probs
            n       = self._last_n_auth
            fake_p  = p[n:]
            result["per_category"] = {
                "authentic":        round(sum(p[:n]) / n, 4),
                "ai_generated":     round((fake_p[0] + fake_p[1]) / 2, 4),
                "digitally_edited": round(fake_p[2], 4),
                "synthetic":        round(fake_p[3], 4),
            }
        return result

    def _siglip2_score(self, image: Image.Image, mode: str) -> float:
        if mode == "zone" and not self._zone_loaded:
            self._load_zone_model()
        use_zone  = (mode == "zone") and (self._zone_model is not None)
        model     = self._zone_model     if use_zone else self._full_model
        processor = self._zone_processor if use_zone else self._full_processor

        all_prompts = _AUTHENTIC_PROMPTS + _FAKE_PROMPTS
        n_auth      = len(_AUTHENTIC_PROMPTS)

        proc_kwargs: dict = {
            "text":           all_prompts,
            "images":         [image.convert("RGB")],
            "return_tensors": "pt",
            "padding":        "max_length",
        }
        if use_zone:
            proc_kwargs["max_num_patches"] = 512

        inputs = processor(**proc_kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits  = outputs.logits_per_image   # [1, N]
            probs   = torch.sigmoid(logits[0])   # sigmoid — NOT softmax

        self._last_probs  = probs.detach().cpu().tolist()
        self._last_n_auth = n_auth

        auth_prob   = probs[:n_auth].mean().item()
        ai_gen_prob = probs[n_auth:n_auth + 2].mean().item()  # first 2 fake prompts

        return float(max(0.0, min(1.0, (ai_gen_prob - auth_prob + 1.0) / 2.0)))

    def _clip_fallback(self, image: Image.Image, mode: str) -> float:
        """Fall back to CLIP ViT-L/14 zero-shot path via UniversalFakeDetect."""
        import math
        import torchvision.transforms as _T

        _clip_preprocess = _T.Compose([_T.Resize(256), _T.CenterCrop(224), _T.ToTensor()])

        try:
            # Validate UFD_REPO is the real UniversalFakeDetect clone before insertion.
            if not (UFD_REPO / "models").is_dir():
                logger.warning("CLIP fallback: UFD_REPO missing 'models' dir — skipping")
                return 0.5
            if str(UFD_REPO) not in sys.path:
                sys.path.insert(0, str(UFD_REPO))

            from universal_fake_detect_adapter import UFDAdapter, is_available  # type: ignore
            if not is_available():
                return 0.5

            ufd      = UFDAdapter()
            raw_clip = ufd._model.model
            device   = ufd.device

            from models.clip import clip as _clip_mod  # type: ignore

            authentic_prompts = [
                "a real photograph of an official government identity document",
                "a genuine authentic passport scan or ID card",
                "a legitimate official photograph of an identity document",
                "a real scanned government issued certificate",
            ]
            fake_prompts = [
                "a computer generated fake artificial identity document",
                "an AI synthesized fake passport or identity card",
                "a digitally manipulated fraudulent government document",
                "a synthetically created artificial fake ID card",
            ]
            all_texts  = authentic_prompts + fake_prompts
            tokens     = _clip_mod.tokenize(all_texts).to(device)
            img_tensor = _clip_preprocess(image.convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feat  = raw_clip.encode_image(img_tensor).float()
                text_feat = raw_clip.encode_text(tokens).float()
                img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                sims      = (img_feat @ text_feat.T).squeeze(0)

            n_auth     = len(authentic_prompts)
            auth_score = sims[:n_auth].mean().item()
            fake_score = sims[n_auth:].mean().item()
            temp  = 0.05
            return math.exp(fake_score / temp) / (math.exp(auth_score / temp) + math.exp(fake_score / temp))

        except Exception as exc:
            logger.warning("CLIP fallback also failed: %s", exc)
            return 0.5
