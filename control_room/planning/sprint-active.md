# Sprint 1 — AI Image Detection Framework
# Project: Internal POC
# Dates: 2026-05-14 → TBD | Capacity: 17 stories

---

## US-001 | Central Config System
Status: DONE
Description: Central configuration for the framework — thresholds, 3 weight presets, path management, env-var overrides.
Acceptance: config.py exports THRESH_AUTHENTIC, THRESH_SUSPICIOUS, ID_CARD_WEIGHTS, DOCUMENT_FRAUD_WEIGHTS, VEHICLE_DAMAGE_WEIGHTS, WEIGHTS_DIR. All paths overridable via env vars.
Output: src/ai_image_detection/config.py

## US-002 | Pluggable Detector Interface
Status: DONE
Description: Abstract base class that all ML-model detectors implement. Enforces a consistent predict() contract across the framework.
Acceptance: BaseDetector ABC exposes predict(img) → dict and predict_roi(img, roi) → dict. Every detector inherits from it.
Output: src/ai_image_detection/detectors/base.py

## US-003 | Deep-Learning Detector Adapters
Status: DONE
Description: Adapters for 9 deep-learning detectors. Each wraps an external model behind the BaseDetector interface.
Acceptance: Detectors implemented — DualBranch (RGB+DCT EfficientNet), TruFor (pixel-level forgery), SRM (filter bank), SigLIP-2 (So400m), CLIP/UFD (UniversalFakeDetect), DIRE (diffusion reconstruction error), RIGID (DINOv2 perturbation sensitivity), TextShield (VLM document forensics), YOLO (damage ROI locator). All lazy-load weights; missing weights skip gracefully.
Output: src/ai_image_detection/detectors/ (dual_branch, trufor, srm, siglip2, clip_ufd, dire, rigid, textshield, yolo_damage, dino_probe)

## US-004 | Forensic Signal Library
Status: DONE
Description: Five classical (non-ML) image forensic signals as standalone scoring functions.
Acceptance: ELA, PRNU, DCT/Benford's Law, CFA demosaicing correlation, EXIF metadata forensics. Each returns {"score": float [0,1], ...}. All handle edge cases without raising.
Output: src/ai_image_detection/signals/ (ela, prnu, dct_benford, cfa, metadata)

## US-005 | Preprocessing Utilities
Status: DONE
Description: Shared image preprocessing helpers used by DualBranch and explainability modules.
Acceptance: preprocess_image() returns (rgb_tensor, dct_tensor) ready for model inference. extract_dct_high_freq() returns (1, 224, 224) array for visualisation.
Output: src/ai_image_detection/preprocessing/dct.py, image.py

## US-006 | Ensemble Scoring Engine
Status: DONE
Description: Multi-signal score fusion with three modes — fixed calibrated weights, Welford online normalisation, and Phase 2 trained meta-classifier fallback.
Acceptance: EnsembleScorer.score(signals, mode) returns ensemble_score, verdict (Authentic / Suspicious / Likely Fraudulent), confidence (HIGH/MEDIUM/LOW), signal_breakdown, weights_used. Missing signals auto-renormalise. RunningStats implements Welford online mean/std.
Output: src/ai_image_detection/ensemble/scorer.py

## US-007 | Preset Pipeline Runner + Custom Pipeline
Status: DONE
Description: Three calibrated preset pipelines (id_card, document_fraud, vehicle_damage) and a free-form custom pipeline that accepts any signal combination.
Acceptance: id_card runs TruFor+SRM+DualBranch+DIRE+RIGID+ELA+PRNU+DCT/Benford+CFA. document_fraud runs TruFor+SRM+SigLIP2+DIRE+RIGID+ELA+PRNU. vehicle_damage runs DualBranch+TruFor+SRM+Metadata+CLIP/UFD+YOLO(ROI only). custom accepts any list from ALL_SIGNAL_KEYS. All return {pipeline, verdict, score, confidence, signals, ensemble, skipped}.
Output: src/ai_image_detection/pipelines.py

## US-008 | CLI Entry Point
Status: DONE
Description: Command-line interface for single-image analysis.
Acceptance: main.py accepts --image, --pipeline, --json flags. Prints formatted verdict+signal table or clean JSON. Exits 1 on missing image. All 3 pipelines selectable.
Output: main.py

## US-009 | YAML Pipeline Configs
Status: DONE
Description: Declarative YAML configuration files for each preset pipeline, enabling future config-driven pipeline loading.
Acceptance: Configs exist for id_card.yaml, document_fraud.yaml, vehicle_damage.yaml.
Output: configs/id_card.yaml, document_fraud.yaml, vehicle_damage.yaml

## US-010 | Grad-CAM Explainability
Status: DONE
Description: Class-discriminative localization map for DualBranchModel showing where the model suspects manipulation.
Acceptance: generate_gradcam() hooks EfficientNet-B0 features[8], computes weighted activation map, resizes to source image. overlay_heatmap() blends result. gradcam_for_roi() handles ROI crops and pastes back into the full image.
Output: src/ai_image_detection/explainability/gradcam.py

## US-011 | Patch-Verdict Grid + DCT Artifact Map
Status: DONE
Description: Spatial explainability visualisations — per-patch inference overlay and DCT frequency artifact map.
Acceptance: patch_verdict_grid() tiles the image with overlapping patches, runs inference on each, paints RED/GREEN verdict cells. dct_artifact_map() renders grayscale + INFERNO colourmap of high-frequency DCT content. build_evidence_panel() is a convenience wrapper returning (original, grid, dct_color, stats).
Output: src/ai_image_detection/explainability/heatmap.py

## US-012 | Streamlit UI Components
Status: DONE
Description: Reusable Streamlit display components for the web app layer.
Acceptance: verdict_banner() renders colour-coded verdict card with score + confidence. signal_bar_chart() renders Plotly horizontal bar chart with threshold lines. signal_breakdown_table() renders signal/score/weight/status table. skipped_signals_warning() shows collapsible list of skipped signals.
Output: src/ai_image_detection/ui/components.py

## US-013 | Document Type Classifier
Status: DONE
Description: Keyword-weighted text classifier for routing documents to the correct TextShield forensic domain.
Acceptance: classify_document(text) returns (doc_type, confidence) for US_PASSPORT, US_DRIVERS_LICENSE, INVOICE, UNKNOWN. textshield_domain() maps doc type to TextShield domain string for pipeline routing.
Output: src/ai_image_detection/document/classifier.py

## US-014 | Model Weights Download Script
Status: IN PROGRESS
Description: Utility script to acquire all model weights required by the pipelines.
Acceptance: YOLO weights auto-download from GitHub. TruFor, UniversalFakeDetect, DualBranch weights print clear manual-step instructions. Env-var override paths documented. Needs: verify all paths match config.py defaults; test with clean weights/ dir.
Output: scripts/download_weights.py

## US-015 | End-to-End Test Suite
Status: DONE
Description: Unit and integration tests covering all pipelines, signals, detectors, and the ensemble scorer.
Acceptance: Tests cover: all 3 pipelines run without error on a synthetic 224×224 image; EnsembleScorer fixed and Welford modes produce valid output; BaseDetector contract enforced; CLI --json output is valid JSON. conftest.py fixtures already in place.
Output: tests/unit/, tests/integration/

## US-016 | Streamlit App Main Entrypoint
Status: IN PROGRESS
Description: Wire all UI components into a runnable Streamlit app with pipeline selector, image upload, and results display.
Acceptance: app.py launches with `streamlit run app.py`. User can upload image, select pipeline, run analysis, view verdict banner + signal chart + breakdown table + explainability panels.
Output: app.py (project root)

## US-017 | Phase 2 Meta-Classifier Training
Status: PLANNED
Description: Train the Gradient Boosting meta-classifier that EnsembleScorer falls back to in "trained" mode.
Acceptance: Training script produces ensemble_meta_clf_v1.pkl in weights/. Model trained on labelled signal vectors. EnsembleScorer mode="trained" uses it without fallback. Accuracy baseline documented.
Output: scripts/train_ensemble_meta_clf.py, weights/ensemble_meta_clf_v1.pkl

## US-018 | Code Quality, Security & Evaluation Framework
Status: IN_PROGRESS
Description: Full engineering audit and hardening of the existing codebase. 16 issues addressed across security, performance, correctness, and code standards. 5-phase evaluation framework created.
Acceptance: (1) pickle.load replaced with joblib+SHA-256 hash sidecar. (2) sys.path validated before insertion in clip_ufd, trufor, siglip2. (3) DIRE dead code removed + module singleton added. (4) RIGID n_perturbations reduced 10→3 + pre-batched noise. (5) SRM tiny-image guard added. (6) DualBranch patches batched into single forward pass. (7) TextShield blocks on CPU. (8) All pipelines validate IMAGE_MIN_SIZE + use detector cache + narrow except clauses. (9) DetectorResult TypedDict defined. (10) 8 inference constants centralised in config.py. (11) 3 test files + evaluate_framework.py created.
Output: src/ai_image_detection/detectors/result.py, tests/unit/test_detectors.py, tests/unit/test_scorer.py, tests/integration/test_pipelines.py, scripts/evaluate_framework.py + 10 modified source files
