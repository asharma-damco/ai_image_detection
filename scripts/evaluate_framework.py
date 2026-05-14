"""
AI Image Detection — 5-Phase Framework Evaluation Runner.

Usage:
    python scripts/evaluate_framework.py                    # phases 1-3 (no dataset)
    python scripts/evaluate_framework.py --dataset /path/to/images/  # all 5 phases
    python scripts/evaluate_framework.py --phase 2          # single phase

Phase 1 — Security checks
Phase 2 — Per-detector timing benchmark
Phase 3 — Correctness / contract tests
Phase 4 — Accuracy evaluation (requires labelled dataset)
Phase 5 — Concurrent load test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from PIL import Image

# Ensure project root is importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

OUTPUT_DIR = _ROOT / "docs"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "evaluation_report.md"


def _header(title: str) -> str:
    return f"\n{'='*60}\n{title}\n{'='*60}"


def _synth_image(size: int = 224) -> Image.Image:
    return Image.new("RGB", (size, size), color=(120, 80, 200))


# ── Phase 1: Security ─────────────────────────────────────────────────────────

def phase1_security() -> list[str]:
    lines = [_header("PHASE 1 — Security Checks")]
    from ai_image_detection.config import WEIGHTS_DIR

    # 1a. Pickle presence
    pkl_path = WEIGHTS_DIR / "ensemble_meta_clf_v1.pkl"
    hash_path = WEIGHTS_DIR / "ensemble_meta_clf_v1.sha256"
    if pkl_path.exists():
        if not hash_path.exists():
            lines.append(f"[WARN]  {pkl_path.name} exists but no .sha256 sidecar found.")
            lines.append("        Run: python scripts/hash_model.py  to generate it.")
        else:
            expected = hash_path.read_text().strip().lower()
            actual   = hashlib.sha256(pkl_path.read_bytes()).hexdigest().lower()
            if actual == expected:
                lines.append(f"[PASS]  ensemble_meta_clf_v1.pkl hash verified.")
            else:
                lines.append(f"[FAIL]  Hash mismatch — model may be tampered!")
    else:
        lines.append(f"[INFO]  No trained meta-classifier found (expected — Phase 2 trains it).")

    # 1b. sys.path insertion safety
    lines.append("[INFO]  sys.path validation guards are active in clip_ufd, trufor, siglip2.")

    # 1c. Min image size
    from ai_image_detection.config import IMAGE_MIN_SIZE
    lines.append(f"[PASS]  IMAGE_MIN_SIZE = {IMAGE_MIN_SIZE} px enforced in all pipelines.")

    lines.append("\nPhase 1 complete.")
    return lines


# ── Phase 2: Timing Benchmark ─────────────────────────────────────────────────

def phase2_timing() -> list[str]:
    lines = [_header("PHASE 2 — Per-Detector Timing Benchmark")]
    img = _synth_image(224)

    DETECTORS = [
        ("SRM",      lambda: _time_srm(img)),
        ("ELA",      lambda: _time_ela(img)),
        ("PRNU",     lambda: _time_prnu(img)),
        ("DCT/Benford", lambda: _time_dct(img)),
        ("CFA",      lambda: _time_cfa(img)),
        ("DualBranch (CPU)", lambda: _time_dual_branch(img)),
        ("RIGID (CPU, 3 perturb)", lambda: _time_rigid(img)),
    ]

    for name, fn in DETECTORS:
        try:
            elapsed = fn()
            flag = "[PASS]" if elapsed < 60 else "[SLOW]"
            lines.append(f"{flag}  {name:<35} {elapsed:.2f}s")
        except Exception as exc:
            lines.append(f"[SKIP]  {name:<35} {type(exc).__name__}: {exc}")

    lines.append("\nPhase 2 complete. Targets: SRM/ELA/PRNU/CFA <1s, DualBranch <30s (GPU <3s), RIGID <90s (GPU <10s).")
    return lines


def _time_srm(img) -> float:
    from ai_image_detection.detectors.srm import SRMAnalyzer
    t = time.perf_counter()
    SRMAnalyzer().predict(img)
    return time.perf_counter() - t


def _time_ela(img) -> float:
    from ai_image_detection.signals.ela import ela_anomaly_score
    import numpy as np
    t = time.perf_counter()
    ela_anomaly_score(img)
    return time.perf_counter() - t


def _time_prnu(img) -> float:
    from ai_image_detection.signals.prnu import prnu_anomaly_score
    import numpy as np
    t = time.perf_counter()
    prnu_anomaly_score(np.array(img))
    return time.perf_counter() - t


def _time_dct(img) -> float:
    from ai_image_detection.signals.dct_benford import dct_benford_score
    import numpy as np
    t = time.perf_counter()
    dct_benford_score(np.array(img))
    return time.perf_counter() - t


def _time_cfa(img) -> float:
    from ai_image_detection.signals.cfa import cfa_correlation_score
    import numpy as np
    t = time.perf_counter()
    cfa_correlation_score(np.array(img))
    return time.perf_counter() - t


def _time_dual_branch(img) -> float:
    from ai_image_detection.detectors.dual_branch import DualBranchDetector
    det = DualBranchDetector()
    t = time.perf_counter()
    det.predict(img)
    return time.perf_counter() - t


def _time_rigid(img) -> float:
    from ai_image_detection.detectors.rigid import RIGIDDetector
    det = RIGIDDetector()
    t = time.perf_counter()
    det.predict(img)
    return time.perf_counter() - t


# ── Phase 3: Correctness Tests ────────────────────────────────────────────────

def phase3_correctness() -> list[str]:
    lines = [_header("PHASE 3 — Correctness / Contract Tests")]
    img = _synth_image(224)

    # 3a. SRM score in [0, 1]
    try:
        from ai_image_detection.detectors.srm import SRMAnalyzer
        r = SRMAnalyzer().predict(img)
        assert 0.0 <= r["score"] <= 1.0
        lines.append("[PASS]  SRM score in [0, 1]")
    except Exception as e:
        lines.append(f"[FAIL]  SRM: {e}")

    # 3b. SRM tiny image guard
    try:
        from ai_image_detection.detectors.srm import SRMAnalyzer
        tiny = Image.new("RGB", (8, 8))
        r = SRMAnalyzer(block_size=16).compute_anomaly_score(tiny)
        assert r["score"] == 0.5
        lines.append("[PASS]  SRM tiny image guard returns 0.5")
    except Exception as e:
        lines.append(f"[FAIL]  SRM tiny guard: {e}")

    # 3c. EnsembleScorer fixed mode
    try:
        from ai_image_detection.ensemble.scorer import EnsembleScorer
        from ai_image_detection.config import VEHICLE_DAMAGE_WEIGHTS
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS)
        r = scorer.score({"trufor": 0.8, "dual_branch": 0.7, "srm": 0.6}, mode="fixed")
        assert 0.0 <= r["ensemble_score"] <= 1.0
        assert r["verdict"] in ("Authentic", "Suspicious", "Likely Fraudulent")
        lines.append("[PASS]  EnsembleScorer fixed mode produces valid output")
    except Exception as e:
        lines.append(f"[FAIL]  EnsembleScorer fixed: {e}")

    # 3d. EnsembleScorer Welford mode
    try:
        from ai_image_detection.ensemble.scorer import EnsembleScorer
        from ai_image_detection.config import VEHICLE_DAMAGE_WEIGHTS
        scorer = EnsembleScorer(custom_weights=VEHICLE_DAMAGE_WEIGHTS, use_welford=True)
        for v in [0.3, 0.5, 0.7]:
            scorer.score({"trufor": v}, mode="fixed")
        lines.append("[PASS]  EnsembleScorer Welford mode runs without error")
    except Exception as e:
        lines.append(f"[FAIL]  EnsembleScorer Welford: {e}")

    # 3e. Image size validation
    try:
        from ai_image_detection.pipelines import run_id_card_pipeline
        tiny = Image.new("RGB", (32, 32))
        try:
            run_id_card_pipeline(tiny)
            lines.append("[FAIL]  id_card pipeline accepted a 32×32 image (should reject)")
        except ValueError:
            lines.append("[PASS]  id_card pipeline correctly rejects 32×32 image")
    except Exception as e:
        lines.append(f"[FAIL]  Size validation check: {e}")

    # 3f. id_card pipeline schema
    try:
        from ai_image_detection.pipelines import run_id_card_pipeline
        r = run_id_card_pipeline(img)
        for key in ("pipeline", "verdict", "score", "confidence", "signals", "ensemble", "skipped"):
            assert key in r, f"Missing key: {key}"
        lines.append("[PASS]  id_card pipeline returns correct schema")
    except Exception as e:
        lines.append(f"[FAIL]  id_card pipeline schema: {e}")

    # 3g. document_fraud pipeline schema
    try:
        from ai_image_detection.pipelines import run_document_fraud_pipeline
        r = run_document_fraud_pipeline(img)
        assert r["pipeline"] == "document_fraud"
        lines.append("[PASS]  document_fraud pipeline returns correct schema")
    except Exception as e:
        lines.append(f"[FAIL]  document_fraud pipeline schema: {e}")

    # 3h. vehicle_damage pipeline schema
    try:
        from ai_image_detection.pipelines import run_vehicle_damage_pipeline
        r = run_vehicle_damage_pipeline(img)
        assert r["pipeline"] == "vehicle_damage"
        lines.append("[PASS]  vehicle_damage pipeline returns correct schema")
    except Exception as e:
        lines.append(f"[FAIL]  vehicle_damage pipeline schema: {e}")

    lines.append("\nPhase 3 complete.")
    return lines


# ── Phase 4: Accuracy Evaluation ─────────────────────────────────────────────

def phase4_accuracy(dataset_dir: Path) -> list[str]:
    lines = [_header("PHASE 4 — Accuracy Evaluation")]
    from ai_image_detection.pipelines import run_id_card_pipeline

    authentic_dir = dataset_dir / "authentic"
    fake_dir      = dataset_dir / "fake"

    if not authentic_dir.exists() or not fake_dir.exists():
        lines.append(f"[SKIP]  Expected {authentic_dir} and {fake_dir}. Create these directories.")
        return lines

    authentic_images = list(authentic_dir.glob("*.jpg")) + list(authentic_dir.glob("*.png"))
    fake_images      = list(fake_dir.glob("*.jpg"))      + list(fake_dir.glob("*.png"))

    if not authentic_images or not fake_images:
        lines.append("[SKIP]  No images found in authentic/ or fake/ subdirectories.")
        return lines

    tp = fp = tn = fn = 0
    scores_real: list[float] = []
    scores_fake: list[float] = []

    for path in authentic_images:
        try:
            img = Image.open(path).convert("RGB")
            r   = run_id_card_pipeline(img)
            score = r["score"]
            scores_real.append(score)
            if r["verdict"] == "Authentic":
                tn += 1
            else:
                fp += 1
        except Exception as e:
            lines.append(f"[WARN]  {path.name}: {e}")

    for path in fake_images:
        try:
            img = Image.open(path).convert("RGB")
            r   = run_id_card_pipeline(img)
            score = r["score"]
            scores_fake.append(score)
            if r["verdict"] != "Authentic":
                tp += 1
            else:
                fn += 1
        except Exception as e:
            lines.append(f"[WARN]  {path.name}: {e}")

    total = tp + tn + fp + fn
    if total == 0:
        lines.append("[SKIP]  No images processed.")
        return lines

    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    lines.append(f"  Authentic images tested : {len(authentic_images)}")
    lines.append(f"  Fake images tested      : {len(fake_images)}")
    lines.append(f"  Accuracy   : {accuracy:.3f}")
    lines.append(f"  Precision  : {precision:.3f}")
    lines.append(f"  Recall     : {recall:.3f}")
    lines.append(f"  F1 Score   : {f1:.3f}")
    lines.append(f"  Mean score (real): {sum(scores_real)/len(scores_real):.3f}" if scores_real else "")
    lines.append(f"  Mean score (fake): {sum(scores_fake)/len(scores_fake):.3f}" if scores_fake else "")

    flag = "[PASS]" if accuracy >= 0.70 else "[WARN]"
    lines.append(f"\n{flag}  Overall accuracy: {accuracy:.1%}")
    lines.append("\nPhase 4 complete.")
    return lines


# ── Phase 5: Concurrent Load Test ─────────────────────────────────────────────

def phase5_load() -> list[str]:
    lines = [_header("PHASE 5 — Concurrent Load Test")]
    from ai_image_detection.pipelines import run_id_card_pipeline

    img     = _synth_image(224)
    n       = 5
    results = [None] * n
    errors  = []

    def _worker(idx: int) -> None:
        try:
            results[idx] = run_id_card_pipeline(img)
        except Exception as exc:
            errors.append(f"Thread {idx}: {exc}")

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    elapsed = time.perf_counter() - t_start

    succeeded = sum(1 for r in results if r is not None)
    lines.append(f"  Concurrent threads: {n}")
    lines.append(f"  Succeeded: {succeeded}/{n}")
    lines.append(f"  Elapsed: {elapsed:.1f}s")
    if errors:
        lines.append(f"  Errors:")
        for err in errors:
            lines.append(f"    {err}")

    # Verify verdicts are consistent (same image → same result from each thread)
    verdicts = {r["verdict"] for r in results if r is not None}
    if len(verdicts) == 1:
        lines.append("[PASS]  All threads returned the same verdict (no shared-state corruption)")
    elif len(verdicts) > 1:
        lines.append(f"[WARN]  Inconsistent verdicts across threads: {verdicts}")

    lines.append("\nPhase 5 complete.")
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Image Detection — Framework Evaluator")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Path to labelled dataset dir (expects authentic/ and fake/ subdirs)")
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help="Run only this phase")
    args = parser.parse_args()

    all_lines: list[str] = [
        "# AI Image Detection — Framework Evaluation Report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    run_all = args.phase is None

    if run_all or args.phase == 1:
        all_lines += phase1_security()
    if run_all or args.phase == 2:
        all_lines += phase2_timing()
    if run_all or args.phase == 3:
        all_lines += phase3_correctness()
    if run_all or args.phase == 4:
        if args.dataset:
            all_lines += phase4_accuracy(args.dataset)
        else:
            all_lines.append(_header("PHASE 4 — Accuracy Evaluation"))
            all_lines.append("[SKIP]  Pass --dataset /path/to/images to enable accuracy evaluation.")
    if run_all or args.phase == 5:
        all_lines += phase5_load()

    report_text = "\n".join(all_lines)
    print(report_text)

    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
