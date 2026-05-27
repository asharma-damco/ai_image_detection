# Pending Changes — AI Image Detection

## Format for each entry
CHG-XXX | [PENDING / DONE / UNDONE] | /command | [date]
Description: [what this change does]
Files: [which files are affected]
---

## Pending (awaiting APPROVE)
None.

## Completed (last 10)
CHG-031 | DONE | /code fix framework-audit-critical-high | 2026-05-27
Description: 9 commits on branch fix/framework-audit-critical-high fixing all Critical and High audit findings + new FastAPI deployment layer.
  C-2  siglip2.py: broken fallback import fixed (wrong module → .clip_ufd)
  C-3  trufor.py: H/W unpacking clarified to W,H = pil_image.size
  C-4  requirements.txt + pyproject.toml: added diffusers, ultralytics, joblib, streamlit; pyproject now has full [project].dependencies; version aligned to 1.0.0
  C-1  pipelines.py: all 8 bare constructors in run_custom_pipeline → _cached(); detector cache now works for custom pipeline
  H-1  pipelines.py: PRNU details (mid_ratio, high_ratio, fft_score) now stored + document_fraud returns 'details' key
  H-10 pipelines.py: SRM formula standardised (0.70×srm + 0.30×jpeg) across all 3 pipelines
  H-6  pipelines.py: manual_roi bounds validation added
  H-3  metadata.py: FocalLength tuple (35,1) → float parsing fixed
  H-5  scorer.py: logger.warning() on both fallback paths + None-signal filter in feature vector
  M-4  ela.py: BytesIO + recompressed Image closed with context manager
  M-8  app.py: logger.exception() on pipeline error + exc type shown in UI
  API  api.py + api_schema.py: FastAPI REST layer with 5 endpoints, Pydantic v2 schemas, base64 heatmaps, CORS, 20MB guard
Files: src/ai_image_detection/detectors/siglip2.py, src/ai_image_detection/detectors/trufor.py, requirements.txt, pyproject.toml, src/ai_image_detection/pipelines.py, src/ai_image_detection/signals/metadata.py, src/ai_image_detection/ensemble/scorer.py, src/ai_image_detection/signals/ela.py, app.py, api.py (new), src/ai_image_detection/api_schema.py (new)
---

## Completed (last 10)
CHG-030 | DONE | Control Room v2 upgrade | 2026-05-22
Description: Full structural upgrade to match control_room_v2 format. Created control_room/README.md (the Brain), rewrote CLAUDE.md as 5-line entry point, added .damco-project.yml, restructured context/ + planning/ into project-context/ (10 memory types: contacts, team, decisions, risks, project-plan, kpis, notes, team-health, communications/, status/, updates/). Added 6 new commands (branch, code, pr, ingest, report, standup). Updated all 14 existing commands + 13 skill files to use control_room/ paths. Added 3 new skill files (skill-ingest, skill-standup, skill-report). Added 3 workflow templates (draft-client-email, draft-steering-summary, draft-weekly-update). Updated settings.json permissions.
Files: control_room/README.md (new), CLAUDE.md, .damco-project.yml (new), .claude/commands/ (19 files), .claude/skills/ (16 files), .claude/workflows/ (3 files), .claude/settings.json, control_room/project-context/ (10 files + 3 subfolders)
---

## Completed (last 10)
CHG-029 | DONE | /dev US-015 DONE | 2026-05-14
Description: Marked US-015 End-to-end test suite as DONE. Tests delivered as part of CHG-028 (test_detectors.py, test_scorer.py, test_pipelines.py, evaluate_framework.py).
Files: control_room/planning/sprint-active.md, control_room/_state.md
---

CHG-028 | DONE | /build US-018 DEV | 2026-05-14
Description: US-018 — Code Quality, Security & Evaluation. 16 issues addressed: pickle RCE → joblib+hash, sys.path validation in clip_ufd/trufor/siglip2, DIRE dead code + singleton, RIGID n_perturbations 10→3 + batch noise, SRM empty-array guard, DualBranch batched patches, TextShield CPU block, pipelines bare-except narrowing + image size validation + detector cache, DetectorResult TypedDict, new constants in config. New files: result.py, test_detectors.py, test_scorer.py, test_pipelines.py, evaluate_framework.py.
Files: src/ai_image_detection/detectors/result.py (new), src/ai_image_detection/config.py, src/ai_image_detection/ensemble/scorer.py, src/ai_image_detection/detectors/clip_ufd.py, src/ai_image_detection/detectors/trufor.py, src/ai_image_detection/detectors/siglip2.py, src/ai_image_detection/detectors/dire.py, src/ai_image_detection/detectors/rigid.py, src/ai_image_detection/detectors/srm.py, src/ai_image_detection/detectors/dual_branch.py, src/ai_image_detection/detectors/textshield.py, src/ai_image_detection/pipelines.py, tests/unit/test_detectors.py (new), tests/unit/test_scorer.py (new), tests/integration/test_pipelines.py (new), scripts/evaluate_framework.py (new), control_room/planning/sprint-active.md, control_room/_state.md
---

## Completed (last 10)
CHG-027 | DONE | /build US-016 DEV | 2026-05-14
Description: Removed "PIMA FCU" and "UAIC" from all user-visible surfaces (app.py labels, pipelines.py docstrings, config.py comments, main.py help text, configs/*.yaml headers). Added model selection multiselect to sidebar for ALL pipeline types (not just Custom) — defaults to preset signals, user can add/remove freely. Added preset_weights param to run_custom_pipeline so calibrated weights are preserved when signals are toggled. Unified all pipeline execution through run_custom_pipeline.
Files: app.py, src/ai_image_detection/pipelines.py, src/ai_image_detection/config.py, main.py, configs/id_card.yaml, configs/document_fraud.yaml, configs/vehicle_damage.yaml
---

CHG-026 | DONE | /build US-016 DEV | 2026-05-14
Description: Full app.py rewrite to fix failing UI and match UAIC framework patterns. Fixed width="stretch" TypeError (4 locations) → use_container_width=True. Fixed YOLO detection key access (d.get("x") → d["bbox_xyxy"]). Replaced st.tabs() with st.expander() (old framework pattern). Added drawing helpers (_cap_display, _draw_manual_roi, _draw_damage_boxes, _draw_roi_overlays, _cluster_detections). Added ROI section with streamlit_image_coordinates canvas + slider fine-tune. Added "Run on ROI" button. Added YOLO damage expander with orange bbox overlay, composite+cyan ROI, cluster thumbnails. Layout: [3,2] columns with gap="large" (matches old framework).
Files: app.py
---

CHG-025 | DONE | /build US-016 POC | 2026-05-14
Description: Fix Streamlit 1.57 deprecation — replace use_container_width with width='stretch' on all st.image calls in app.py; remove stale noqa comment in components.py.
Files: app.py, src/ai_image_detection/ui/components.py
---

## Completed (last 10)
CHG-022 | DONE | /wake | 2026-05-14
Description: Full repo restructure — src/ layout, control_room/ rename, .gitignore, pyproject.toml, requirements.txt, tests/, notebooks/, scripts/, CLAUDE.md to root, config.py hardcoded path removed.
Files: src/ai_image_detection/ (moved), main.py (renamed), samples/ (renamed), control_room/ (from control_room_v2), .claude/ (to root), CLAUDE.md (to root, rewritten), .gitignore, pyproject.toml, requirements.txt, README.md, tests/conftest.py, scripts/download_weights.py
---
CHG-021 | DONE | /wake | 2026-05-14
Description: Control Room initialised for AI Image Detection project. Sprint 1 opened with 5 stories (US-001–US-005) pulled from docs/user-stories.md. Statuses inferred from existing codebase.
Files: control_room_v2/_state.md, control_room_v2/pending-changes.md
---
