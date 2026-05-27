# Project Plan
# Client: Damco Solutions (Internal) | SOW: INTERNAL-POC-2026

## Phases
| Phase | Name | Dates | Status |
|-------|------|-------|--------|
| 1 | Core Framework + Pipelines + UI | 2026-05-14 → TBD | IN PROGRESS |
| 2 | Meta-Classifier Training + Evaluation | TBD | NOT STARTED |

## Streams
| Stream | Owner | Deliverable | Status |
|--------|-------|-------------|--------|
| Detection Engine | Abhishek Sharma | 9 detector adapters + 5 forensic signals | DONE |
| Pipeline Runner | Abhishek Sharma | 3 preset pipelines + custom pipeline | DONE |
| Explainability | Abhishek Sharma | Grad-CAM + patch-verdict + DCT map | DONE |
| UI | Abhishek Sharma | Streamlit app (app.py) | IN PROGRESS |
| Evaluation | Abhishek Sharma | evaluate_framework.py + test suite | DONE |
| Phase 2 ML | Abhishek Sharma | Meta-classifier training script | PLANNED |

## Active Sprint — Sprint 1
Sprint: 1 | Dates: 2026-05-14 → TBD | Capacity: 17 stories | Used: 18

### US-001 | Central Config System | DONE
Description: Central configuration for the framework — thresholds, 3 weight presets, path management, env-var overrides.
Output: src/ai_image_detection/config.py

### US-002 | Pluggable Detector Interface | DONE
Description: Abstract base class that all ML-model detectors implement.
Output: src/ai_image_detection/detectors/base.py

### US-003 | Deep-Learning Detector Adapters | DONE
Description: Adapters for 9 deep-learning detectors behind the BaseDetector interface.
Output: src/ai_image_detection/detectors/ (dual_branch, trufor, srm, siglip2, clip_ufd, dire, rigid, textshield, yolo_damage)

### US-004 | Forensic Signal Library | DONE
Description: Five classical (non-ML) image forensic signals as standalone scoring functions.
Output: src/ai_image_detection/signals/ (ela, prnu, dct_benford, cfa, metadata)

### US-005 | Preprocessing Utilities | DONE
Description: Shared image preprocessing helpers used by DualBranch and explainability modules.
Output: src/ai_image_detection/preprocessing/dct.py, image.py

### US-006 | Ensemble Scoring Engine | DONE
Description: Multi-signal score fusion with fixed weights, Welford normalisation, and trained meta-classifier fallback.
Output: src/ai_image_detection/ensemble/scorer.py

### US-007 | Preset Pipeline Runner + Custom Pipeline | DONE
Description: Three calibrated preset pipelines (id_card, document_fraud, vehicle_damage) and a free-form custom pipeline.
Output: src/ai_image_detection/pipelines.py

### US-008 | CLI Entry Point | DONE
Description: Command-line interface for single-image analysis.
Output: main.py

### US-009 | YAML Pipeline Configs | DONE
Description: Declarative YAML configuration files for each preset pipeline.
Output: configs/id_card.yaml, document_fraud.yaml, vehicle_damage.yaml

### US-010 | Grad-CAM Explainability | DONE
Description: Class-discriminative localization map for DualBranchModel.
Output: src/ai_image_detection/explainability/gradcam.py

### US-011 | Patch-Verdict Grid + DCT Artifact Map | DONE
Description: Spatial explainability visualisations.
Output: src/ai_image_detection/explainability/heatmap.py

### US-012 | Streamlit UI Components | DONE
Description: Reusable Streamlit display components (verdict banner, signal bar chart, breakdown table).
Output: src/ai_image_detection/ui/components.py

### US-013 | Document Type Classifier | DONE
Description: Keyword-weighted text classifier for routing documents to the correct TextShield forensic domain.
Output: src/ai_image_detection/document/classifier.py

### US-014 | Model Weights Download Script | IN PROGRESS
Description: Utility script to acquire all model weights. YOLO auto-downloads; TruFor/UFD/DualBranch are manual. Needs path verification against config.py.
Output: scripts/download_weights.py

### US-015 | End-to-End Test Suite | DONE
Description: Unit and integration tests covering all pipelines, signals, detectors, and ensemble scorer.
Output: tests/unit/, tests/integration/

### US-016 | Streamlit App Main Entrypoint | IN PROGRESS
Description: Wire all UI components into a runnable Streamlit app with pipeline selector, image upload, and results display.
Output: app.py

### US-017 | Phase 2 Meta-Classifier Training | PLANNED
Description: Train the Gradient Boosting meta-classifier for EnsembleScorer "trained" mode.
Output: scripts/train_ensemble_meta_clf.py, weights/ensemble_meta_clf_v1.pkl

### US-018 | Code Quality, Security & Evaluation Framework | DONE
Description: Full engineering audit — 16 issues addressed across security, performance, correctness, code standards.
Output: src/ai_image_detection/detectors/result.py, tests/unit/test_detectors.py, tests/unit/test_scorer.py, tests/integration/test_pipelines.py, scripts/evaluate_framework.py

## Backlog (unprioritised)
| US-ID | Title | Points | Priority | Source | Notes |
|-------|-------|--------|----------|--------|-------|
| | | | | | |

Priority values: HIGH / MED / LOW
Run /plan to pull items into the active sprint.
