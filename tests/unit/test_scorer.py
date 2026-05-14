"""Unit tests for EnsembleScorer and RunningStats."""
from __future__ import annotations

import math

import pytest

from ai_image_detection.ensemble.scorer import EnsembleScorer, RunningStats
from ai_image_detection.config import (
    VEHICLE_DAMAGE_WEIGHTS,
    ID_CARD_WEIGHTS,
    THRESH_AUTHENTIC,
    THRESH_SUSPICIOUS,
)


# ── RunningStats (Welford) ────────────────────────────────────────────────────

class TestRunningStats:
    def test_single_value_std_is_one(self):
        rs = RunningStats()
        rs.update(0.7)
        assert rs.std == 1.0

    def test_mean_updates_correctly(self):
        rs = RunningStats()
        for v in [0.2, 0.4, 0.6]:
            rs.update(v)
        assert abs(rs.mean - 0.4) < 1e-6

    def test_std_positive_with_two_values(self):
        rs = RunningStats()
        rs.update(0.0)
        rs.update(1.0)
        assert rs.std > 0

    def test_reset_clears_state(self):
        rs = RunningStats()
        rs.update(0.9)
        rs.reset()
        assert rs.n == 0
        assert rs.mean == 0.0


# ── EnsembleScorer — fixed mode ───────────────────────────────────────────────

class TestEnsembleScorerFixed:
    def test_all_zero_signals_returns_authentic(self):
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS)
        signals = {"trufor": 0.0, "dual_branch": 0.0, "clip_ufd": 0.0, "srm": 0.0, "rigid": 0.0}
        result = scorer.score(signals, mode="fixed")
        assert result["ensemble_score"] < THRESH_AUTHENTIC
        assert result["verdict"] == "Authentic"

    def test_all_one_signals_returns_fraudulent(self):
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS)
        signals = {"trufor": 1.0, "dual_branch": 1.0, "clip_ufd": 1.0, "srm": 1.0, "rigid": 1.0}
        result = scorer.score(signals, mode="fixed")
        assert result["ensemble_score"] >= THRESH_SUSPICIOUS
        assert result["verdict"] == "Likely Fraudulent"

    def test_score_in_valid_range(self):
        scorer = EnsembleScorer(custom_weights=ID_CARD_WEIGHTS)
        signals = {"trufor": 0.55, "srm": 0.30, "dual_branch": 0.45, "rigid": 0.20}
        result = scorer.score(signals, mode="fixed")
        assert 0.0 <= result["ensemble_score"] <= 1.0

    def test_missing_signals_renormalise(self):
        """Subset of signals should renormalise weights and still return valid result."""
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS)
        result = scorer.score({"trufor": 0.8}, mode="fixed")
        assert result["ensemble_score"] == 0.8
        assert result["signal_count"] == 1

    def test_empty_signals_returns_neutral(self):
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS)
        result = scorer.score({}, mode="fixed")
        assert result["ensemble_score"] == 0.5
        assert result["signal_count"] == 0

    def test_confidence_field_present(self):
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS)
        result = scorer.score({"trufor": 0.6, "srm": 0.7}, mode="fixed")
        assert result["confidence"] in ("HIGH", "MEDIUM", "LOW")

    def test_weights_used_sum_to_one(self):
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS)
        signals = {"trufor": 0.5, "dual_branch": 0.5, "srm": 0.5}
        result = scorer.score(signals, mode="fixed")
        total = sum(result["weights_used"].values())
        assert abs(total - 1.0) < 1e-6

    def test_mode_field_is_fixed(self):
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS)
        result = scorer.score({"trufor": 0.5}, mode="fixed")
        assert result["mode"] == "fixed"


# ── EnsembleScorer — Welford mode ─────────────────────────────────────────────

class TestEnsembleScorerWelford:
    def test_welford_first_call_returns_input_unchanged(self):
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS, use_welford=True)
        signals = {"trufor": 0.7}
        result = scorer.score(signals, mode="fixed")
        # First call: n=1, normalised score should equal raw score
        assert result["ensemble_score"] == 0.7

    def test_welford_second_call_shifts_normalised(self):
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS, use_welford=True)
        scorer.score({"trufor": 0.3}, mode="fixed")
        scorer.score({"trufor": 0.7}, mode="fixed")
        result = scorer.score({"trufor": 0.5}, mode="fixed")
        # Third call should produce a normalised score (may differ from 0.5)
        assert 0.0 <= result["ensemble_score"] <= 1.0

    def test_reset_welford_clears_state(self):
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS, use_welford=True)
        scorer.score({"trufor": 0.9}, mode="fixed")
        scorer.reset_welford()
        assert len(scorer._welford_state) == 0


# ── EnsembleScorer — trained mode fallback ────────────────────────────────────

class TestEnsembleScorerTrainedFallback:
    def test_trained_mode_falls_back_when_model_absent(self, tmp_path):
        """When no pkl file exists, trained mode silently falls back to fixed."""
        import os
        from ai_image_detection.config import WEIGHTS_DIR
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS)
        signals = {"trufor": 0.6, "dual_branch": 0.4}
        result = scorer.score(signals, mode="trained")
        # Must fall back gracefully
        assert result["mode"] in ("fixed_fallback", "trained")
        assert 0.0 <= result["ensemble_score"] <= 1.0
        assert "_fallback_reason" in result or result["mode"] == "trained"
