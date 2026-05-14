"""
Download all model weights required by the ai_image_detection pipelines.

Usage
-----
    python scripts/download_weights.py

Environment overrides (optional)
---------------------------------
    DUAL_BRANCH_WEIGHTS_PATH  — path to dtc_rgb_model_v1.pth
    TRUFOR_DIR                — path to cloned TruFor repo
    UFD_REPO                  — path to cloned UniversalFakeDetect repo
    YOLO_MODEL_DIR            — path to YOLO damage model dir
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

YOLO_URL = (
    "https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector"
    "/raw/main/trained.pt"
)


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already exists: {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  saved → {dest}")


def main() -> None:
    print("=== AI Image Detection — weight downloader ===")
    print(f"Target: {WEIGHTS_DIR}\n")

    print("[YOLO damage model]")
    download(YOLO_URL, WEIGHTS_DIR / "trained.pt")

    print("\n[TruFor]")
    print("  Manual step: clone https://github.com/grip-unina/TruFor into weights/TruFor")
    print("  then download trufor.pth.tar per the TruFor README.")

    print("\n[UniversalFakeDetect / CLIP-UFD]")
    print("  Manual step: clone https://github.com/WisconsinAIVision/UniversalFakeDetect")
    print("  into weights/UniversalFakeDetect and download fc_weights.pth.")

    print("\n[DualBranch]")
    print("  Place dtc_rgb_model_v1.pth in weights/ or set DUAL_BRANCH_WEIGHTS_PATH env var.")

    print("\nDone. Set env vars in .env if weights are in non-default locations.")


if __name__ == "__main__":
    main()
