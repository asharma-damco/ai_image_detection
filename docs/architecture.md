# Architecture — ai_image_detection
**Updated:** 2026-04-28

## System Overview
The system is a Python-based image analysis pipeline that accepts input images, preprocesses them, and passes them through one or more detection models to classify whether each image is AI-generated or AI-edited. Results are returned with a confidence score and supporting evidence (e.g. artefact map, metadata flags).

## Components

### AI Image Detection Module
- **Purpose:** Core detection logic — loads models, runs inference, returns classification results
- **Inputs:** Image file path or binary blob
- **Outputs:** Label (AI-generated / AI-edited / authentic), confidence score
- **Status:** Not started

## Integration Points
- No external APIs defined yet
- Model weights to be sourced (local files or HuggingFace Hub — TBD)

## Data Flow
1. Image input (file / URL / batch)
2. Preprocessing — resize, normalise, format conversion
3. Detection module — model inference and/or frequency analysis
4. Post-processing — threshold application, confidence scoring
5. Output — classification result returned to caller

## Open Architecture Questions
- AQ-01: Will inference run locally or via a hosted model endpoint?
- AQ-02: Should the framework expose a CLI, a REST API, or both?
