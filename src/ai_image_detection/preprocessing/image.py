"""
Image preprocessing pipeline for DualBranchModel inference.

Produces the (rgb_tensor, dct_tensor) pair the model expects.
Normalisation matches training: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5].
"""

from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

from .dct import extract_dct_high_freq

# Must match the transform used during training (test_transform in notebook)
_RGB_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def preprocess_image(
    img: Image.Image,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a PIL Image into the two input tensors for DualBranchModel.

    Args:
        img: PIL.Image in any mode (converted to RGB internally).

    Returns:
        rgb_tensor : shape [1, 3, 224, 224], float32, normalised to [-1, 1]
        dct_tensor : shape [1, 1, 224, 224], float32
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    rgb_tensor = _RGB_TRANSFORM(img).unsqueeze(0)            # [1, 3, 224, 224]

    dct_array  = extract_dct_high_freq(img)                  # (1, 224, 224)
    dct_tensor = torch.from_numpy(dct_array).unsqueeze(0)    # [1, 1, 224, 224]

    return rgb_tensor, dct_tensor
