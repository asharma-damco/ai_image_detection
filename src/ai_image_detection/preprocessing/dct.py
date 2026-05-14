"""
DCT high-frequency extractor — ported from DCT_RGB_V1.ipynb (PIMA/UAIC)

Converts an RGB image into a single-channel high-frequency DCT map
consumed by the DCT branch of DualBranchModel.

Steps:
  1. Convert RGB → grayscale
  2. Resize to (size × size)
  3. Normalise pixel values to [0, 1]
  4. Apply 2-D DCT
  5. Zero out the low-frequency top-left block (h//8 × w//8)
  6. Log-scale: log(|dct| + 1)
  7. Add channel dimension → shape (1, size, size)
"""

import cv2
import numpy as np
from PIL import Image


def extract_dct_high_freq(img: Image.Image, size: int = 224) -> np.ndarray:
    """
    Extract high-frequency DCT components from a PIL Image.

    Args:
        img:  PIL Image (any mode — converted to RGB then grayscale internally).
        size: Output spatial resolution (default 224 to match DualBranchModel input).

    Returns:
        np.ndarray of shape (1, size, size), dtype float32.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (size, size))
    gray = np.float32(gray) / 255.0

    dct = cv2.dct(gray)

    h, w = dct.shape
    dct[: h // 8, : w // 8] = 0.0   # suppress DC / low-frequency block

    dct = np.log(np.abs(dct) + 1.0)  # log-scale for numerical stability

    return np.expand_dims(dct, axis=0).astype(np.float32)  # (1, H, W)
