"""
DINOv2 ViT-L/14 Frozen-Feature Anomaly Signal.

Three zero-training distribution-shift indicators:
  1. Patch token entropy   — real: high diversity; AI: smooth/uniform (low entropy)
  2. CLS token magnitude   — AI images are OOD for DINOv2 → higher CLS L2 norm
  3. Attention head disagr — real: heads specialise; AI: heads converge

All three combined via fixed weights into a single score [0, 1].
Gracefully degrades to score=0.5 if torch/DINOv2 is unavailable.

Source: UAIC uaic-fraud-detection/poc/signals/dino_probe.py
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Module-level singleton ────────────────────────────────────────────────────
_DINO_MODEL = None
_DINO_DEVICE: Optional[str] = None
_DINO_LOAD_FAILED: bool = False

_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD  = (0.229, 0.224, 0.225)

# CLS norm prior recalibrated for ID document images
_CLS_PRIOR_MEAN: float = 44.0
_CLS_PRIOR_STD:  float = 3.0

_W_PATCH: float = 0.40
_W_CLS:   float = 0.35
_W_HEAD:  float = 0.25


def _load_dino_model():
    global _DINO_MODEL, _DINO_DEVICE, _DINO_LOAD_FAILED

    if _DINO_MODEL is not None:
        return _DINO_MODEL, _DINO_DEVICE
    if _DINO_LOAD_FAILED:
        return None, None

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14",
            pretrained=True, verbose=False,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model = model.to(device)
        _DINO_MODEL  = model
        _DINO_DEVICE = device
        logger.info("DINOv2 ViT-L/14 loaded on %s", device)
        return _DINO_MODEL, _DINO_DEVICE
    except Exception as exc:
        _DINO_LOAD_FAILED = True
        logger.warning("DINOv2 unavailable — graceful degradation: %s", exc)
        return None, None


def _preprocess(img_pil, device: str):
    import torch
    from PIL import Image
    img = img_pil.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(_DINO_MEAN, dtype=np.float32)) / np.array(_DINO_STD, dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)


def _patch_entropy_score(patch_tokens) -> float:
    """Coefficient of variation of per-patch L2 norms.

    Real docs: high spatial diversity (std ≈ 0.8–2.0) → score → 0
    AI docs:   homogeneous synthesis  (std < 0.5)      → score → 1
    """
    import torch
    patch_norms = torch.norm(patch_tokens.float(), dim=-1)
    diversity   = float(patch_norms.std().item())
    return float(np.clip((0.8 - diversity) / 0.6, 0.0, 1.0))


def _cls_magnitude_score(cls_token) -> tuple[float, float]:
    """Sigmoid z-score of CLS token L2 norm."""
    norm  = float(cls_token.float().norm().item())
    z     = (norm - _CLS_PRIOR_MEAN) / _CLS_PRIOR_STD
    score = 1.0 / (1.0 + math.exp(-z))
    return float(np.clip(score, 0.0, 1.0)), norm


def _head_disagreement_score(attn_map) -> tuple[float, float]:
    """Attention range across heads.

    Higher range = heads specialise (real) → score → 0
    Lower range  = heads converge    (AI)  → score → 1
    """
    a          = attn_map[0].float()                                   # [n_heads, N, N]
    head_range = (a.max(dim=0).values - a.min(dim=0).values).mean()
    disagree   = float(head_range.item())
    score      = float(np.clip((0.10 - disagree) / 0.08, 0.0, 1.0))
    return score, disagree


def dino_feature_score(img_pil) -> dict:
    """Compute DINOv2 feature anomaly score.

    Args:
        img_pil: PIL Image (any mode — converted to RGB internally).

    Returns:
        score             float [0, 1]  0=authentic, 1=AI-generated
        patch_entropy     float  patch token diversity anomaly
        cls_norm          float  raw CLS token L2 norm
        head_disagreement float  inter-head attention variance
        model_loaded      bool
    """
    _FALLBACK = {
        "score": 0.5, "patch_entropy": 0.5,
        "cls_norm": float(_CLS_PRIOR_MEAN), "head_disagreement": 0.5,
        "model_loaded": False,
    }

    model, device = _load_dino_model()
    if model is None:
        return _FALLBACK

    try:
        import torch

        tensor = _preprocess(img_pil, device)

        _attn_capture: list = []

        last_attn  = model.blocks[-1].attn
        _orig_fwd  = last_attn.forward

        def _patched_forward(x):
            import math as _math
            B, N, C = x.shape
            qkv = last_attn.qkv(x)
            qkv = qkv.reshape(B, N, 3, last_attn.num_heads, C // last_attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scale = _math.sqrt(q.shape[-1])
            attn  = torch.softmax((q @ k.transpose(-2, -1)) / scale, dim=-1)
            _attn_capture.append(attn.detach())
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return last_attn.proj(out)

        last_attn.forward = _patched_forward
        try:
            with torch.no_grad():
                out = model.forward_features(tensor)
        finally:
            last_attn.forward = _orig_fwd

        if isinstance(out, dict):
            patch_tokens = out.get("x_norm_patchtokens")
            cls_token    = out.get("x_norm_clstoken")
        else:
            cls_token    = out[:, 0, :]
            patch_tokens = out[:, 1:, :]

        if patch_tokens is None or cls_token is None:
            return _FALLBACK

        patch_tokens = patch_tokens.squeeze(0)
        cls_token    = cls_token.squeeze(0)

        score_patch              = _patch_entropy_score(patch_tokens)
        score_cls, cls_norm_val  = _cls_magnitude_score(cls_token)

        if _attn_capture:
            score_heads, head_disagree_val = _head_disagreement_score(_attn_capture[-1])
        else:
            score_heads, head_disagree_val = 0.5, 0.5

        combined = float(np.clip(
            _W_PATCH * score_patch + _W_CLS * score_cls + _W_HEAD * score_heads,
            0.0, 1.0,
        ))

        return {
            "score":             combined,
            "patch_entropy":     score_patch,
            "cls_norm":          cls_norm_val,
            "head_disagreement": score_heads,
            "model_loaded":      True,
        }

    except Exception as exc:
        logger.warning("dino_feature_score error — returning 0.5: %s", exc)
        return _FALLBACK
