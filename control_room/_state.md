# Project state — AI Image Detection
# Last updated: 2026-05-14 | Updated by: CHG-029

## IDs
Next CHG-ID: CHG-032
Next DEC-ID: DEC-001
Next US-ID:  US-019
Next R-ID:   R-01

---

## Sprint
Sprint: 1
Status: IN PROGRESS
Dates: 2026-05-14 → TBD
Capacity: 17 stories
Used: 18 stories
Progress: 83% (15/18 stories done)

---

## Active stories
US-014 | Model weights download script        | IN PROGRESS — YOLO auto-downloads; TruFor/UFD/DualBranch manual; needs path verification against config.py
US-016 | Streamlit app main entrypoint        | PLANNED
US-017 | Phase 2 meta-classifier training     | PLANNED

---

## Completed stories
US-001 | Central config system
US-002 | Pluggable detector interface (BaseDetector ABC)
US-003 | Deep-learning detector adapters (9 detectors)
US-004 | Forensic signal library (ELA, PRNU, DCT/Benford, CFA, Metadata)
US-005 | Preprocessing utilities (dct.py, image.py)
US-006 | Ensemble scoring engine (fixed + Welford + trained fallback)
US-007 | 3 preset pipelines + custom pipeline runner
US-008 | CLI entry point (main.py)
US-009 | YAML pipeline configs (id_card, document_fraud, vehicle_damage)
US-010 | Grad-CAM explainability
US-011 | Patch-verdict grid + DCT artifact map
US-012 | Streamlit UI components
US-013 | Document type classifier
US-015 | End-to-end test suite
US-018 | Code quality, security & evaluation framework

---

## Blocked stories
None.

---

## Open action items
• Validate end-to-end run: python main.py --image samples/test.png --pipeline id_card
• Confirm model weights available in weights/ or run scripts/download_weights.py
• Decide target accuracy / false-positive tolerance for POC sign-off

---

## Top risks
None logged. Run /risk to add.

---

## Pending approvals
None.

---

## Last 5 changes
CHG-031 | DONE | /code framework audit fixes — 9 commits on fix/framework-audit-critical-high: all Critical+High bugs fixed, FastAPI api.py created | 2026-05-27
CHG-030 | DONE | Control Room v2 upgrade — README.md (Brain), CLAUDE.md rewritten, .damco-project.yml, project-context/ (10 memory types), 6 new commands, 3 new skills, 3 workflows, all paths updated | 2026-05-22
CHG-029 | DONE | US-015 marked DONE — end-to-end test suite delivered via CHG-028 (test_detectors.py, test_scorer.py, test_pipelines.py, evaluate_framework.py) | 2026-05-14
CHG-028 | DONE | US-018 code quality + security hardening — pickle→joblib+hash, sys.path validation, DIRE singleton, RIGID n_perturb 10→3, SRM guard, DualBranch batched patches, TextShield CPU block, pipelines bare-except narrowed + size validation + detector cache, result.py TypedDict, config constants, 5 new test/eval files | 2026-05-14
CHG-027 | DONE | US-016 branding removed + model selector added — PIMA FCU/UAIC stripped from all UI; multiselect for all pipeline types; preset_weights param keeps calibration when signals toggled | 2026-05-14
CHG-026 | DONE | US-016 app.py full rewrite — fixed width="stretch" TypeError, YOLO key access, added drawing helpers, ROI canvas section, damage bbox overlay, cluster view, [3,2] layout matching old framework | 2026-05-14
CHG-024 | DONE | US-016 Streamlit app created — app.py wires verdict_banner, signal_bar_chart, breakdown_table, Grad-CAM + patch-verdict explainability, raw JSON tab | 2026-05-14
CHG-023 | DONE | Control room reset — sprint-active.md and _state.md fully rewritten to reflect AI Image Detection framework; Dagriya FinTech content removed | 2026-05-14
CHG-022 | DONE | Full repo restructure — src/ layout, control_room/, .gitignore, pyproject.toml, requirements.txt, tests/, notebooks/, scripts/, CLAUDE.md to root | 2026-05-14
CHG-021 | DONE | Control Room initialised for AI Image Detection project — Sprint 1 opened | 2026-05-14

---

## Last synced
Never. Run /sync to push to external systems.

---

## Today's recommendation
15 of 18 stories done. Priority: run `python scripts/evaluate_framework.py` to validate the full framework, then close US-018 and move to US-016 Streamlit app or US-017 meta-classifier training.
