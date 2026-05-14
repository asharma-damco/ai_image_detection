"""
ai_image_detection — Top-level pipeline runner (CLI entry point).

Usage
-----
    python ai_image_detection.py --image path/to/image.jpg --pipeline id_card
    python ai_image_detection.py --image path/to/image.jpg --pipeline document_fraud
    python ai_image_detection.py --image path/to/image.jpg --pipeline vehicle_damage

Pipelines
---------
    id_card         ID card edit detection
                    Signals: TruFor + SRM + DualBranch + ELA + PRNU + Benford + CFA

    document_fraud  General document fraud detection
                    Signals: TruFor + SRM + SigLIP-2 + NoisePrint + ELA + PRNU

    vehicle_damage  Vehicle damage fraud detection
                    Signals: DualBranch + CLIP/UFD + TruFor + SRM + Metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from ai_image_detection.pipelines import (
    run_document_fraud_pipeline,
    run_id_card_pipeline,
    run_vehicle_damage_pipeline,
)

_PIPELINES = {
    "id_card":        run_id_card_pipeline,
    "document_fraud": run_document_fraud_pipeline,
    "vehicle_damage": run_vehicle_damage_pipeline,
}


def main():
    parser = argparse.ArgumentParser(description="AI Image Detection Framework")
    parser.add_argument("--image",    required=True, help="Path to image file")
    parser.add_argument("--pipeline", required=True, choices=list(_PIPELINES), help="Detection pipeline to run")
    parser.add_argument("--json",     action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    img    = Image.open(img_path)
    result = _PIPELINES[args.pipeline](img)

    if args.json:
        def _clean(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return "<ndarray>"
            return obj
        print(json.dumps(_clean(result), indent=2))
    else:
        print(f"\nPipeline : {result['pipeline']}")
        print(f"Verdict  : {result['verdict']}")
        print(f"Score    : {result['score']:.4f}")
        print(f"Confidence: {result['confidence']}")
        print(f"\nSignals:")
        for k, v in result["signals"].items():
            print(f"  {k:30s} {v:.4f}")
        if result["skipped"]:
            print(f"\nSkipped: {', '.join(result['skipped'])}")


if __name__ == "__main__":
    main()
