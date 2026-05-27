# KPIs — AI Image Detection

## Format
KPI-XX | [name] | [INTERNAL / EXTERNAL]
Why it matters: [business reason]
Baseline: [current value or "not yet measured"]
Target: [goal]
Source: [where the data comes from]
Capture method: [manual / automated / agent hook]
Cadence: [daily / weekly / per-sprint]
Owner: [who is responsible for this KPI]
---

No KPIs formally defined yet. Run /capture or add manually.

## Suggested KPIs for this POC

### Detection accuracy
KPI-01 | Detection Accuracy (true positive rate) | INTERNAL
Why it matters: Core metric — measures whether the framework correctly flags fraudulent images
Baseline: not yet measured
Target: ≥ 85% on test set
Source: scripts/evaluate_framework.py
Capture method: manual (run evaluate_framework.py)
Cadence: per-sprint
Owner: Abhishek Sharma

### False positive rate
KPI-02 | False Positive Rate | INTERNAL
Why it matters: Determines operational viability — too many false positives = user trust erosion
Baseline: not yet measured
Target: ≤ 10%
Source: scripts/evaluate_framework.py
Capture method: manual
Cadence: per-sprint
Owner: Abhishek Sharma

### Pipeline latency
KPI-03 | End-to-End Pipeline Latency (seconds) | INTERNAL
Why it matters: Determines whether the system is usable in near-real-time workflows
Baseline: not yet measured
Target: < 30s per image on CPU, < 5s on GPU
Source: evaluate_framework.py timing output
Capture method: manual
Cadence: per-sprint
Owner: Abhishek Sharma
