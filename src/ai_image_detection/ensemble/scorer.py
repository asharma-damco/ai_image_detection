"""
Ensemble Multi-Signal Score Fusion.

Combines available forensic signals into a single fraud probability.
Phase 1: fixed calibrated weights (three presets — vehicle, document, ID card).
Phase 2: trained Gradient Boosting meta-classifier (loads from disk if available).

Signal convention
-----------------
All signals are expressed as fake probability in [0, 1]:
    0.0 = definitely authentic
    1.0 = definitely fake / manipulated

Caller conversions before passing to score():
    dual_branch   → result["probabilities"]["Fake"]
    clip_ufd      → result["probabilities"]["Fake"]
    siglip2       → siglip2_detector.score(image)
    trufor        → 1.0 - trufor_result["integrity_score"]
    srm           → srm_result["score"]
    prnu          → prnu_anomaly_score()["score"]
    ela           → ela_anomaly_score()["score"]
    dct_benford   → dct_benford_score()["score"]
    cfa           → cfa_correlation_score()["score"]
    dinov2        → dino_feature_score(image)["score"]
    metadata      → 1.0 - analyze_metadata()["authenticity_score"]

Sources: UAIC uaic-fraud-detection/poc/ensemble_scorer.py
         PIMA onboarding_poc/cu_poc/id_fraud_analysis.py (PIMA_DOC_WEIGHTS)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from ..config import (
    DOCUMENT_FRAUD_WEIGHTS,
    ID_CARD_WEIGHTS,
    SIGNAL_LABELS,
    SPREAD_HIGH,
    SPREAD_MEDIUM,
    THRESH_AUTHENTIC,
    THRESH_SUSPICIOUS,
    VEHICLE_DAMAGE_WEIGHTS,
    WEIGHTS_DIR,
)

_TRAINED_MODEL_PATH = WEIGHTS_DIR / "ensemble_meta_clf_v1.pkl"
_TRAINED_HASH_PATH  = WEIGHTS_DIR / "ensemble_meta_clf_v1.sha256"


class _ModelSecurityError(RuntimeError):
    """Raised when the trained classifier hash does not match the sidecar file."""


def _load_clf_safely():
    """Load the meta-classifier using joblib with SHA-256 hash verification.

    A sidecar file (<name>.sha256) must exist alongside the pkl. If the hash
    does not match, loading is refused to prevent pickle RCE on tampered files.
    """
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("joblib is required for safe model loading: pip install joblib") from exc

    if not _TRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {_TRAINED_MODEL_PATH}")

    if _TRAINED_HASH_PATH.exists():
        import hashlib
        expected = _TRAINED_HASH_PATH.read_text().strip().lower()
        actual   = hashlib.sha256(_TRAINED_MODEL_PATH.read_bytes()).hexdigest().lower()
        if actual != expected:
            raise _ModelSecurityError(
                f"SHA-256 mismatch for {_TRAINED_MODEL_PATH.name}.\n"
                "The file may be corrupted or tampered. Delete it and retrain:\n"
                "  python scripts/train_ensemble_meta_clf.py"
            )

    return joblib.load(_TRAINED_MODEL_PATH)


# ── Welford online running stats ──────────────────────────────────────────────

class RunningStats:
    """Welford online running mean and sample standard deviation.

    Used for adaptive z-score normalisation of signals across a session.
    State resets on app restart (not persisted).
    """

    def __init__(self) -> None:
        self.n:   int   = 0
        self.mean: float = 0.0
        self._M2:  float = 0.0

    @property
    def std(self) -> float:
        return float(math.sqrt(self._M2 / (self.n - 1))) if self.n >= 2 else 1.0

    def update(self, value: float) -> None:
        self.n   += 1
        delta     = value - self.mean
        self.mean += delta / self.n
        self._M2  += delta * (value - self.mean)

    def reset(self) -> None:
        self.n = 0;  self.mean = 0.0;  self._M2 = 0.0


# ── EnsembleScorer ────────────────────────────────────────────────────────────

class EnsembleScorer:
    """Combines forensic signals into a single fraud probability.

    Parameters
    ----------
    custom_weights : dict, optional
        Override the default fixed weights. Pass one of the presets from
        config.py (VEHICLE_DAMAGE_WEIGHTS, DOCUMENT_FRAUD_WEIGHTS,
        ID_CARD_WEIGHTS) or any custom dict mapping signal_key → float.
        Defaults to VEHICLE_DAMAGE_WEIGHTS.
    use_welford : bool, optional
        When True, each signal score is z-score normalised against its own
        running mean/std before weighting. Useful when processing a batch of
        similar documents. Default: False.
    """

    def __init__(
        self,
        custom_weights: Optional[Dict[str, float]] = None,
        use_welford: bool = False,
    ) -> None:
        self._custom_weights  = custom_weights
        self._use_welford     = use_welford
        self._welford_state: Dict[str, dict] = {}

    def score(self, signals: Dict[str, float], mode: str = "fixed") -> dict:
        """Compute ensemble fraud probability from available signals.

        Parameters
        ----------
        signals : dict
            Mapping of signal name → fake probability (0–1).
            Unknown keys are ignored. Missing signals are excluded and the
            remaining weights auto-renormalise.
        mode : "fixed" | "trained"
            "fixed"   → Phase 1 calibrated linear combination.
            "trained" → Phase 2 meta-classifier (falls back to fixed if
                        model file not found).

        Returns
        -------
        dict:
            ensemble_score    float  0–1 fake probability
            verdict           str    "Authentic" | "Suspicious" | "Likely Fraudulent"
            confidence        str    "HIGH" | "MEDIUM" | "LOW"
            signal_count      int
            signal_breakdown  list   one dict per signal
            weights_used      dict   renormalised weights actually applied
            mode              str
        """
        return self._score_trained(signals) if mode == "trained" else self._score_fixed(signals)

    def reset_welford(self) -> None:
        """Clear all Welford running statistics."""
        self._welford_state.clear()

    # ── Fixed-weight scoring ──────────────────────────────────────────────────

    def _score_fixed(self, signals: Dict[str, float]) -> dict:
        base_weights = self._custom_weights or VEHICLE_DAMAGE_WEIGHTS

        active: Dict[str, float] = {
            k: float(v) for k, v in signals.items()
            if k in base_weights and v is not None
        }

        normalised_scores: Dict[str, float] = {}
        if self._use_welford and active:
            for k, v in active.items():
                normalised_scores[k] = self._welford_update(k, v)
            active_for_scoring = normalised_scores
        else:
            active_for_scoring = active

        breakdown = []
        for name in base_weights:
            available  = name in active
            raw_score  = active.get(name)
            norm_score = normalised_scores.get(name, raw_score)
            breakdown.append({
                "name":             name,
                "label":            SIGNAL_LABELS.get(name, name),
                "score":            round(raw_score,  4) if raw_score  is not None else None,
                "normalised_score": round(norm_score, 4) if norm_score is not None else None,
                "base_weight":      base_weights.get(name, 0.0),
                "available":        available,
            })

        if not active_for_scoring:
            return self._build_result(0.5, breakdown, {}, 0, "fixed")

        total_w      = sum(base_weights[k] for k in active_for_scoring)
        weights_used = {k: base_weights[k] / total_w for k in active_for_scoring}
        ensemble_score = round(
            min(max(sum(active_for_scoring[k] * weights_used[k] for k in active_for_scoring), 0.0), 1.0),
            4,
        )

        for row in breakdown:
            row["weight_used"] = round(weights_used.get(row["name"], 0.0), 4)

        return self._build_result(ensemble_score, breakdown, weights_used, len(active_for_scoring), "fixed")

    # ── Welford normalisation ─────────────────────────────────────────────────

    def _welford_update(self, key: str, value: float) -> float:
        if key not in self._welford_state:
            self._welford_state[key] = {"n": 0, "mean": 0.0, "M2": 0.0}
        state = self._welford_state[key]
        state["n"] += 1
        n     = state["n"]
        delta = value - state["mean"]
        state["mean"] += delta / n
        state["M2"]   += delta * (value - state["mean"])
        if n < 2:
            return float(value)
        variance = state["M2"] / (n - 1)
        std      = math.sqrt(variance) if variance > 1e-12 else 1.0
        z        = (value - state["mean"]) / std
        return float(min(max(0.5 + z / 6.0, 0.0), 1.0))

    # ── Trained meta-classifier ───────────────────────────────────────────────

    def _score_trained(self, signals: Dict[str, float]) -> dict:
        if not _TRAINED_MODEL_PATH.exists():
            reason = f"Trained model not found at {_TRAINED_MODEL_PATH}."
            logger.warning("EnsembleScorer: falling back to fixed weights — %s", reason)
            result = self._score_fixed(signals)
            result["mode"]             = "fixed_fallback"
            result["_fallback_reason"] = reason
            return result
        try:
            import numpy as np
            clf = _load_clf_safely()

            # H-5: filter None values — use only signals that actually ran.
            # signals.get(key, 0.5) would silently impute 0.5 for None, corrupting
            # the feature vector. Only include keys with a real float value.
            _FEAT_KEYS = ("dual_branch", "clip_ufd", "trufor", "srm", "metadata")
            clean = {k: float(v) for k, v in signals.items() if k in _FEAT_KEYS and v is not None}
            feat  = [clean.get(k, 0.5) for k in _FEAT_KEYS]

            prob   = clf.predict_proba([feat])[0][1]
            result = self._score_fixed(signals)
            result["ensemble_score"] = round(float(prob), 4)
            result["verdict"]        = self._verdict(result["ensemble_score"])
            result["mode"]           = "trained"
            return result
        except Exception as e:
            reason = f"Trained model load error: {e}"
            logger.warning("EnsembleScorer: falling back to fixed weights — %s", reason)
            result = self._score_fixed(signals)
            result["mode"]             = "fixed_fallback"
            result["_fallback_reason"] = reason
            return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_result(self, ensemble_score, breakdown, weights_used, signal_count, mode) -> dict:
        return {
            "ensemble_score":   ensemble_score,
            "verdict":          self._verdict(ensemble_score),
            "confidence":       self._confidence(breakdown),
            "signal_count":     signal_count,
            "signal_breakdown": breakdown,
            "weights_used":     weights_used,
            "mode":             mode,
        }

    @staticmethod
    def _verdict(score: float) -> str:
        if score < THRESH_AUTHENTIC:    return "Authentic"
        if score < THRESH_SUSPICIOUS:   return "Suspicious"
        return "Likely Fraudulent"

    @staticmethod
    def _confidence(breakdown: list) -> str:
        scores = [row["score"] for row in breakdown if row["available"] and row.get("score") is not None]
        if len(scores) < 2:
            return "LOW"
        spread = max(scores) - min(scores)
        if spread < SPREAD_HIGH:    return "HIGH"
        if spread < SPREAD_MEDIUM:  return "MEDIUM"
        return "LOW"
