"""DIRE — Diffusion Reconstruction Error detector.

Reference: Wang et al., "DIRE for Diffusion-Generated Image Detection", ICCV 2023.
Extended with RF-Solver style Euler inversion for Flux-generated image detection.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionXLPipeline
from PIL import Image

from ..config import DIRE_INVERSION_STEPS
from .base import BaseDetector

logger = logging.getLogger(__name__)

# ── SDXL constants ────────────────────────────────────────────────────────────
_VAE_SCALE   = 0.13025
_TARGET_SIZE = 512
_STEPS       = DIRE_INVERSION_STEPS   # from config (default 10; was 20 — halves latency)
_THRESHOLD   = 0.05

_SEQ_LEN    = 77
_HIDDEN_DIM = 2048
_POOLED_DIM = 1280
_TIME_IDS   = [_TARGET_SIZE, _TARGET_SIZE, 0, 0, _TARGET_SIZE, _TARGET_SIZE]

# ── Flux constants ────────────────────────────────────────────────────────────
_FLUX_MODEL     = "black-forest-labs/FLUX.1-dev"
_FLUX_VAE_SCALE = 0.3611
_FLUX_VAE_SHIFT = 0.1159
_FLUX_TARGET    = 1024

# Known Flux default output resolutions (w, h)
_FLUX_RESOLUTIONS: set[tuple[int, int]] = {
    (1024, 1024),
    (1280,  720), ( 720, 1280),
    ( 832, 1216), (1216,  832),
    ( 896, 1152), (1152,  896),
}
_FLUX_ASPECT_RATIOS: list[float] = [w / h for w, h in _FLUX_RESOLUTIONS]
_FLUX_AR_TOLERANCE = 0.02   # ±2% for AR fuzzy match


class DIREDetector(BaseDetector):
    """Detect AI-generated images via diffusion reconstruction error (DIRE).

    Routes between SDXL DDIM inversion (default) and RF-Solver style Euler
    inversion for images suspected to be Flux-generated.
    """

    DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        num_inversion_steps: int = _STEPS,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.num_inversion_steps = num_inversion_steps
        self._model_id = model_id
        self.pipe = None
        self._loaded = False
        self._flux_pipe = None      # lazy-loaded on first Flux-suspected call
        self._flux_loaded = False
        # Pipeline loading is deferred to first predict() call to avoid OOM on CPU startup.

    # ── SDXL pipeline ─────────────────────────────────────────────────────────

    # Keys present in SDXL DDIM config that DDIMInverseScheduler does not accept.
    _DDIM_INV_UNSUPPORTED = frozenset({"interpolation_type", "use_karras_sigmas", "skip_prk_steps"})

    def _load_pipeline(self, model_id: str) -> None:
        kwargs = dict(torch_dtype=self.dtype, use_safetensors=True)
        if self.dtype == torch.float16:
            kwargs["variant"] = "fp16"
        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **kwargs).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        inv_cfg = {k: v for k, v in self.pipe.scheduler.config.items()
                   if k not in self._DDIM_INV_UNSUPPORTED and not k.startswith("_")}
        self.inverse_scheduler = DDIMInverseScheduler(**inv_cfg)
        self.pipe.set_progress_bar_config(disable=True)
        self._loaded = True

    def _uncond_kwargs(self) -> tuple[torch.Tensor, dict]:
        enc = torch.zeros(1, _SEQ_LEN, _HIDDEN_DIM, device=self.device, dtype=self.dtype)
        pooled = torch.zeros(1, _POOLED_DIM, device=self.device, dtype=self.dtype)
        time_ids = torch.tensor([_TIME_IDS], device=self.device, dtype=self.dtype)
        return enc, {"text_embeds": pooled, "time_ids": time_ids}

    def _encode_latent(self, img: Image.Image) -> torch.Tensor:
        """Encode a 512×512 PIL image to SDXL VAE latent space."""
        img_512 = img.convert("RGB").resize((_TARGET_SIZE, _TARGET_SIZE))
        arr = np.array(img_512, dtype=np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device, self.dtype)
        t = (t / 127.5) - 1.0
        return self.pipe.vae.encode(t).latent_dist.sample() * _VAE_SCALE

    @torch.no_grad()
    def _ddim_inversion(self, latent: torch.Tensor) -> torch.Tensor:
        """Run DDIM inversion: clean latent → noisy latent."""
        enc, added = self._uncond_kwargs()
        self.inverse_scheduler.set_timesteps(self.num_inversion_steps)
        z = latent.clone()
        for t in self.inverse_scheduler.timesteps:
            noise_pred = self.pipe.unet(
                z, t, encoder_hidden_states=enc, added_cond_kwargs=added
            ).sample
            z = self.inverse_scheduler.step(noise_pred, t, z).prev_sample
        return z

    @torch.no_grad()
    def _reconstruct(self, inv_latent: torch.Tensor) -> torch.Tensor:
        """Denoise an inverted latent back to pixel space; returns (1,3,H,W) in [-1,1]."""
        enc, added = self._uncond_kwargs()
        self.pipe.scheduler.set_timesteps(self.num_inversion_steps)
        z = inv_latent.clone()
        for t in self.pipe.scheduler.timesteps:
            noise_pred = self.pipe.unet(
                z, t, encoder_hidden_states=enc, added_cond_kwargs=added
            ).sample
            z = self.pipe.scheduler.step(noise_pred, t, z).prev_sample
        return self.pipe.vae.decode(z / _VAE_SCALE).sample

    # ── Flux routing heuristic ────────────────────────────────────────────────

    def _is_flux_suspected(self, img: Image.Image) -> bool:
        """Heuristic: True if image characteristics match known Flux output patterns.

        Checks resolution exact-match, aspect-ratio fuzzy-match (±2%),
        and absence of EXIF camera model. Routing only — not a detector.
        """
        w, h = img.size

        if (w, h) in _FLUX_RESOLUTIONS:
            ar_match = True
        else:
            ar = w / h
            ar_match = any(
                abs(ar - ref) / ref < _FLUX_AR_TOLERANCE
                for ref in _FLUX_ASPECT_RATIOS
            )

        if not ar_match:
            return False

        # Tag 0x010F = Make, 0x0110 = Model; real photos almost always carry these
        try:
            exif = img.getexif()
            has_camera = bool(exif.get(0x010F) or exif.get(0x0110))
        except Exception:
            has_camera = False

        return not has_camera

    # ── Flux RF-Solver style inversion ────────────────────────────────────────

    def _load_flux_pipeline(self) -> bool:
        """Lazy-load the Flux pipeline. Returns True on success."""
        if self._flux_loaded:
            return True
        if self.device.type != "cuda":
            logger.warning("RF-Solver unavailable, using SDXL fallback (Flux requires CUDA)")
            return False
        try:
            from diffusers import FluxPipeline
            self._flux_pipe = FluxPipeline.from_pretrained(
                _FLUX_MODEL, torch_dtype=torch.bfloat16
            ).to(self.device)
            self._flux_pipe.set_progress_bar_config(disable=True)
            self._flux_loaded = True
            return True
        except Exception as exc:
            logger.warning(f"RF-Solver unavailable, using SDXL fallback: {exc}")
            return False

    @staticmethod
    def _pack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
        """Rearrange (1, 16, H, W) VAE latents into Flux transformer input (1, H/2·W/2, 64)."""
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)
        return latents

    @staticmethod
    def _unpack_flux_latents(packed: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Reverse of _pack_flux_latents: (1, H/2·W/2, 64) → (1, 16, H, W)."""
        b, _, _ = packed.shape
        packed = packed.view(b, h // 2, w // 2, 16, 2, 2)
        packed = packed.permute(0, 3, 1, 4, 2, 5).reshape(b, 16, h, w)
        return packed

    @staticmethod
    def _get_flux_img_ids(h_lat: int, w_lat: int) -> torch.Tensor:
        """Flux position IDs for image patches; shape (H/2·W/2, 3) — no batch dim."""
        ids = torch.zeros(h_lat // 2, w_lat // 2, 3)
        ids[..., 1] = torch.arange(h_lat // 2, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(w_lat // 2, dtype=torch.float32)[None, :]
        return ids.reshape(-1, 3)

    @torch.no_grad()
    def _get_flux_dire_score(self, img: Image.Image) -> tuple[float, np.ndarray, str]:
        """RF-Solver second-order midpoint inversion for Flux-suspected images.

        Encodes image to Flux latent space, inverts along the rectified-flow
        ODE (RF-Solver midpoint, t: 0→1), then reconstructs (t: 1→0)
        and measures pixel error.

        Returns:
            (mean_error, error_map, backbone_used)
        """
        if not self._load_flux_pipeline():
            score, error_map = self.compute_dire(img)
            return score, error_map, "sdxl_fallback"

        pipe  = self._flux_pipe
        vae   = pipe.vae
        dtype = torch.bfloat16

        img_proc = img.convert("RGB").resize((_FLUX_TARGET, _FLUX_TARGET))
        arr = np.array(img_proc, dtype=np.float32)
        orig_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype)
        orig_t = (orig_t / 127.5) - 1.0

        # Flux VAE encode with shift + scale
        latents = vae.encode(orig_t).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        _, _, h_lat, w_lat = latents.shape

        # Empty prompt — txt_ids has no batch dim per Flux transformer contract
        prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
            prompt="", prompt_2="", device=self.device, num_images_per_prompt=1
        )
        img_ids  = self._get_flux_img_ids(h_lat, w_lat).to(self.device, dtype)
        text_ids = text_ids.to(self.device, dtype) if text_ids is not None else \
                   torch.zeros(prompt_embeds.shape[1], 3, device=self.device, dtype=dtype)

        guidance = torch.full((1,), 3.5, device=self.device, dtype=dtype)
        steps    = self.num_inversion_steps
        sigmas   = torch.linspace(0.0, 1.0, steps + 1, device=self.device, dtype=dtype)

        def _unet(z: torch.Tensor, t: float) -> torch.Tensor:
            t_vec = torch.full((1,), t, device=self.device, dtype=dtype)
            return pipe.transformer(
                hidden_states=z,
                timestep=t_vec,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                img_ids=img_ids,
                txt_ids=text_ids,
                guidance=guidance,
                return_dict=False,
            )[0]

        packed = self._pack_flux_latents(latents)

        # ── Old RF-Euler (removed):                                              ──
        # for i in range(steps):                                                  ──
        #     t_vec = sigmas[i]                                                   ──
        #     v = unet(z, t_vec)          # one velocity eval                     ──
        #     z = z + (sigmas[i+1] - sigmas[i]) * v   # straight-line step       ──
        #                                                                          ──
        # ── New RF-Solver: second-order midpoint corrector (replaces RF-Euler) ──
        # for i in range(steps):                                                  ──
        #     dt = sigmas[i+1] - sigmas[i]                                        ──
        #     v1 = unet(z, sigmas[i])              # velocity at current point    ──
        #     x_mid = z + v1 * (dt / 2)           # half-step to midpoint        ──
        #     v2 = unet(x_mid, sigmas[i] + dt/2)  # velocity at midpoint         ──
        #     z = z + v2 * dt                     # full step using midpoint vel  ──

        # RF-Solver: second-order midpoint corrector (replaces RF-Euler)
        z = packed.clone()
        for i in range(steps):
            dt  = float(sigmas[i + 1] - sigmas[i])
            t_i = float(sigmas[i])

            v1    = _unet(z, t_i)
            x_mid = z + v1 * (dt / 2.0)
            v2    = _unet(x_mid, t_i + dt / 2.0)
            z     = z + v2 * dt

            if i == 0:
                logger.debug(
                    f"RF-Solver step 0: v1_norm={v1.norm():.4f} "
                    f"v2_norm={v2.norm():.4f} delta={abs(v1.norm() - v2.norm()):.4f}"
                )

        # RF-Solver: second-order midpoint corrector (reconstruction, t: 1→0)
        for i in range(steps - 1, -1, -1):
            dt  = float(sigmas[i] - sigmas[i + 1])   # negative
            t_i = float(sigmas[i + 1])

            v1    = _unet(z, t_i)
            x_mid = z + v1 * (dt / 2.0)
            v2    = _unet(x_mid, t_i + dt / 2.0)
            z     = z + v2 * dt

        rec_latents = self._unpack_flux_latents(z, h_lat, w_lat)
        rec_latents = rec_latents / vae.config.scaling_factor + vae.config.shift_factor
        rec_t = vae.decode(rec_latents).sample

        error_map = torch.abs(orig_t - rec_t).squeeze(0).mean(dim=0).cpu().numpy()
        return float(error_map.mean()), error_map, "flux_rf_solver"

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_dire(self, img: Image.Image) -> tuple[float, np.ndarray]:
        """SDXL DDIM path: encode → invert → reconstruct → L1 pixel error.

        Returns:
            (mean_error, error_map) where lower → more likely AI-generated.
        """
        img_512 = img.convert("RGB").resize((_TARGET_SIZE, _TARGET_SIZE))
        arr = np.array(img_512, dtype=np.float32)
        orig = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device, self.dtype)
        orig = (orig / 127.5) - 1.0

        latent = self.pipe.vae.encode(orig).latent_dist.sample() * _VAE_SCALE
        inv_latent = self._ddim_inversion(latent)
        reconst = self._reconstruct(inv_latent)

        error_map = torch.abs(orig - reconst).squeeze(0).mean(dim=0).cpu().numpy()
        return float(error_map.mean()), error_map

    def predict(self, img: Image.Image, threshold: float = _THRESHOLD, **kwargs) -> dict:
        """Run DIRE detection, routing to the Flux RF path if suspected.

        Returns:
            score         float | None  — mean DIRE error (lower = more likely AI)
            verdict       str           — "likely_ai" | "likely_real" | "unavailable"
            error_map     np.ndarray    — per-pixel L1 error heatmap
            threshold     float         — threshold used
            label         str           — "Fake" | "Real" | "Unknown"
            confidence    float         — probability of predicted class
            backbone_used str           — "sdxl_ddim" | "flux_rf_solver" | "sdxl_fallback" | "none"
        """
        if not self._loaded:
            if self.device.type != "cuda":
                warnings.warn(
                    "DIRE requires a CUDA GPU — skipping on CPU. "
                    "Install CUDA and a GPU-enabled PyTorch build to enable DIRE."
                )
                return {
                    "score": None,
                    "verdict": "unavailable",
                    "error_map": None,
                    "threshold": threshold,
                    "label": "Unknown",
                    "confidence": 0.0,
                    "backbone_used": "none",
                }
            try:
                self._load_pipeline(self._model_id)
            except Exception as exc:
                warnings.warn(
                    f"DIRE: could not load '{self._model_id}' on {self.device}: {exc}. "
                    "predict() will return None scores."
                )
                return {
                    "score": None,
                    "verdict": "unavailable",
                    "error_map": None,
                    "threshold": threshold,
                    "label": "Unknown",
                    "confidence": 0.0,
                    "backbone_used": "none",
                }
        # NOTE: the block above either returns early or sets self._loaded = True.
        # No further "if not self._loaded" check is reachable here.

        if self._is_flux_suspected(img):
            score, error_map, backbone = self._get_flux_dire_score(img)
        else:
            score, error_map = self.compute_dire(img)
            backbone = "sdxl_ddim"

        verdict = "likely_ai" if score < threshold else "likely_real"
        label = "Fake" if verdict == "likely_ai" else "Real"
        fake_prob = float(1.0 / (1.0 + np.exp((score - threshold) * 100)))
        confidence = round(fake_prob if label == "Fake" else 1.0 - fake_prob, 4)

        return {
            "score": round(score, 6),
            "verdict": verdict,
            "error_map": error_map,
            "threshold": threshold,
            "label": label,
            "confidence": confidence,
            "backbone_used": backbone,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
# Avoids reloading the SDXL pipeline (several GB) across repeated pipeline calls.

_DIRE_SINGLETON: DIREDetector | None = None


def get_dire_detector(**kwargs) -> DIREDetector:
    """Return the module-level DIREDetector singleton, creating it on first call."""
    global _DIRE_SINGLETON
    if _DIRE_SINGLETON is None:
        _DIRE_SINGLETON = DIREDetector(**kwargs)
    return _DIRE_SINGLETON


if __name__ == "__main__":
    import sys

    # Test 1: unloaded model → None score, backbone = "none"
    det = object.__new__(DIREDetector)
    det._loaded = False
    r = det.predict(Image.new("RGB", (64, 64), color=(128, 128, 128)))
    assert r["score"] is None
    assert r["backbone_used"] == "none"

    # Test 2: 1024×1024 no EXIF → Flux suspected
    det2 = object.__new__(DIREDetector)
    assert DIREDetector._is_flux_suspected(det2, Image.new("RGB", (1024, 1024)))

    # Test 3: 800×600 → not Flux suspected
    assert not DIREDetector._is_flux_suspected(det2, Image.new("RGB", (800, 600)))

    print("DIRE scaffold OK")
    sys.exit(0)
