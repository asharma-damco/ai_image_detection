# AI Image Detection — Framework Evaluation Report
Generated: 2026-05-14 18:08:12


============================================================
PHASE 1 — Security Checks
============================================================
[INFO]  No trained meta-classifier found (expected — Phase 2 trains it).
[INFO]  sys.path validation guards are active in clip_ufd, trufor, siglip2.
[PASS]  IMAGE_MIN_SIZE = 64 px enforced in all pipelines.

Phase 1 complete.

============================================================
PHASE 2 — Per-Detector Timing Benchmark
============================================================
[PASS]  SRM                                 0.02s
[PASS]  ELA                                 0.02s
[PASS]  PRNU                                0.01s
[PASS]  DCT/Benford                         1.33s
[PASS]  CFA                                 0.15s
[SKIP]  DualBranch (CPU)                    FileNotFoundError: [Errno 2] No such file or directory: 'D:\\ai_image_detection\\weights\\dtc_rgb_model_v1.pth'
[PASS]  RIGID (CPU, 3 perturb)              22.32s

Phase 2 complete. Targets: SRM/ELA/PRNU/CFA <1s, DualBranch <30s (GPU <3s), RIGID <90s (GPU <10s).

============================================================
PHASE 3 — Correctness / Contract Tests
============================================================
[PASS]  SRM score in [0, 1]
[PASS]  SRM tiny image guard returns 0.5
[PASS]  EnsembleScorer fixed mode produces valid output
[PASS]  EnsembleScorer Welford mode runs without error
[PASS]  id_card pipeline correctly rejects 32×32 image
[PASS]  id_card pipeline returns correct schema
[PASS]  document_fraud pipeline returns correct schema
[PASS]  vehicle_damage pipeline returns correct schema

Phase 3 complete.

============================================================
PHASE 4 — Accuracy Evaluation
============================================================
[SKIP]  Pass --dataset /path/to/images to enable accuracy evaluation.

============================================================
PHASE 5 — Concurrent Load Test
============================================================
  Concurrent threads: 5
  Succeeded: 5/5
  Elapsed: 1.2s
[PASS]  All threads returned the same verdict (no shared-state corruption)

Phase 5 complete.