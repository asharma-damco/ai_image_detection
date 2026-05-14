# AI Image Detection

A multi-signal ensemble framework for detecting AI-generated and AI-edited images.

## Pipelines

| Pipeline | Use case | Key signals |
|----------|----------|-------------|
| `id_card` | PIMA FCU ID card edit detection | TruFor + SRM + DualBranch + ELA + PRNU + Benford + CFA |
| `document_fraud` | UAIC general document fraud | TruFor + SRM + SigLIP-2 + NoisePrint + ELA + PRNU |
| `vehicle_damage` | UAIC vehicle damage fraud | DualBranch + CLIP/UFD + TruFor + SRM + Metadata |

## Quick start

```bash
pip install -e .
python main.py --image samples/1.png --pipeline id_card
python main.py --image samples/1.png --pipeline id_card --json
```

## Project structure

```
src/ai_image_detection/   # production package
tests/                    # test suite
configs/                  # YAML pipeline configs
docs/                     # project documentation
scripts/                  # utilities (weight download, etc.)
notebooks/                # R&D experiments
weights/                  # model weights (not in git — see scripts/download_weights.py)
```

## Setup — model weights

```bash
python scripts/download_weights.py
```

See `docs/architecture.md` for full detector descriptions.
