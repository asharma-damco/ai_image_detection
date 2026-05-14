"""
Grad-CAM for DualBranchModel — RGB branch only.

Computes a class-discriminative localization map:
  1. Forward pass — capture final conv feature maps (EfficientNet-B0 features[8])
  2. Backward pass — capture gradients for target class
  3. Global-average-pool gradients → per-channel weights
  4. Weighted sum of feature maps + ReLU → heatmap
  5. Resize to original image size + apply JET colormap

Source: UAIC uaic-fraud-detection/src/backend/app/ml/gradcam.py
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..detectors.dual_branch import DualBranchModel
from ..preprocessing.image import preprocess_image


def generate_gradcam(
    model: DualBranchModel,
    img: Image.Image,
    device: torch.device,
    target_class: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """Generate a Grad-CAM heatmap for the given image.

    Args:
        model:        Loaded DualBranchModel (eval mode).
        img:          PIL Image (RGB), any size.
        device:       torch.device.
        target_class: 0=Fake, 1=Real. If None, uses predicted class.

    Returns:
        heatmap_bgr:  np.ndarray (H, W, 3) uint8 BGR — same size as input image.
        target_class: int — class that was visualised.
        class_prob:   float — probability of that class.
    """
    model.eval()

    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    def _fwd_hook(_, __, output):
        activations.append(output.detach())

    def _bwd_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.rgb_encoder.features[8]
    fwd_handle   = target_layer.register_forward_hook(_fwd_hook)
    bwd_handle   = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        rgb_t, dct_t = preprocess_image(img)
        rgb_t = rgb_t.to(device).requires_grad_(True)
        dct_t = dct_t.to(device)

        logits = model(rgb_t, dct_t)
        probs  = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        model.zero_grad()
        logits[0, target_class].backward()

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    act  = activations[0].squeeze(0)       # (C, H', W')
    grad = gradients[0].squeeze(0)         # (C, H', W')
    weights = grad.mean(dim=(1, 2))        # global avg pool → (C,)
    cam  = F.relu((weights[:, None, None] * act).sum(dim=0))  # (H', W')

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)

    cam_np     = cam.cpu().numpy()
    W, H       = img.size
    cam_resized = cv2.resize(cam_np, (W, H), interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    return heatmap_bgr, target_class, float(probs[0, target_class].item())


def overlay_heatmap(
    img: Image.Image,
    heatmap_bgr: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Blend Grad-CAM heatmap over the original image.

    Args:
        img:         Original PIL Image (RGB).
        heatmap_bgr: BGR heatmap from generate_gradcam().
        alpha:       Heatmap opacity (0 = original only, 1 = heatmap only).
    """
    img_np    = np.array(img.convert("RGB"))
    heat_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    if heat_rgb.shape[:2] != img_np.shape[:2]:
        heat_rgb = cv2.resize(heat_rgb, (img_np.shape[1], img_np.shape[0]))
    blended = cv2.addWeighted(img_np, 1 - alpha, heat_rgb, alpha, 0)
    return Image.fromarray(blended)


def gradcam_for_roi(
    model: DualBranchModel,
    img: Image.Image,
    roi: dict,
    device: torch.device,
    target_class: int | None = None,
    alpha: float = 0.5,
) -> tuple[Image.Image, int, float]:
    """Generate Grad-CAM for a specific ROI crop, paste back into full image.

    Returns full-image PIL with heatmap only in the ROI region.
    """
    x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    crop        = img.crop((x, y, x + w, y + h))
    heatmap_bgr, cls, conf = generate_gradcam(model, crop, device, target_class)
    overlay     = overlay_heatmap(crop, heatmap_bgr, alpha)
    result      = img.copy().convert("RGB")
    result.paste(overlay, (x, y))
    return result, cls, conf
