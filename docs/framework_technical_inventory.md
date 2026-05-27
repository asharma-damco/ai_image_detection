# AI Image Detection Framework — Technical Inventory

**Generated:** 2026-05-22  
**Reviewer target:** External AI image researcher evaluating technical merit  
**Reading time:** ~15 minutes

---

## 1. Environment & Dependencies

### Python Version
`>=3.10` (constraint in `pyproject.toml`)

### Direct Dependencies (`requirements.txt`)

| Package | Pinned version | Purpose |
|---|---|---|
| Pillow | `>=10.0.0` | Image I/O, mode conversion |
| numpy | `>=1.24.0` | Numerical ops throughout |
| torch | `>=2.0.0` | Deep learning inference |
| torchvision | `>=0.15.0` | EfficientNet weights, transforms |
| transformers | `>=4.35.0` | SigLIP-2, TextShield (Qwen), AutoModel |
| timm | `>=0.9.0` | Listed but not directly imported in current code |
| scipy | `>=1.10.0` | CFA Wiener filter, Benford chi-square |
| opencv-python | `>=4.8.0` | DCT, colormap, resize, filter2D |
| PyYAML | `>=6.0` | Config loading |
| pytest | `>=7.4.0` | Test runner |
| pytest-cov | `>=4.1.0` | Coverage |

**Unlisted runtime dependencies (required but not in requirements.txt):**

| Package | Where required | Notes |
|---|---|---|
| `diffusers` | `detectors/dire.py` | DDIMScheduler, StableDiffusionXLPipeline, FluxPipeline |
| `ultralytics` | `detectors/yolo_damage.py` | YOLOv11 inference |
| `joblib` | `ensemble/scorer.py` | Meta-classifier loading |
| `jpegio` | `signals/dct_benford.py` | Raw DCT coefficient extraction |
| `hashlib` | stdlib | SHA-256 pkl verification |

**Unpinned / potentially outdated concerns:**
- All versions use `>=` lower bounds with no upper pins — no version ceiling means a future breaking release (e.g. transformers 5.x) would not be caught.
- `transformers>=4.35.0` is the stated floor but SigLIP-2 (`google/siglip2-so400m-patch14-384`) requires `>=4.50.0` (noted in `siglip2.py` fallback message).
- No CUDA version specified; `torch>=2.0.0` with CUDA 11.8 or 12.1 is implied by diffusers SDXL usage.
- `diffusers` is completely absent from `requirements.txt` despite being a hard dependency of DIRE.
- `timm` is listed but no detector currently uses it directly.

### System-level dependencies
- `opencv-python` requires `libGL.so.1` on Linux (common CI failure point).
- DIRE SDXL path requires GPU VRAM ≥ 6 GB (`fp16`, SDXL); Flux path requires ≥ 16 GB (`bfloat16`).
- TextShield refuses to load on CPU (Qwen2.5-VL-7B-Instruct, ~14 GB).

---

## 2. Detector & Signal Inventory

### 2.1 `detectors/dual_branch.py` — `DualBranchDetector`

**Architecture:** Custom dual-branch CNN trained in-house.

| Component | Detail |
|---|---|
| RGB branch | EfficientNet-B0 (ImageNet1K V1 pretrained via torchvision) → AdaptiveAvgPool → 1280-d |
| DCT branch | 3-layer CNN: Conv(1→16→32→64, 3×3, BN+ReLU+MaxPool2) → AdaptiveAvgPool → FC(64→256) |
| Fusion | Learnable soft gating: concat [rgb_proj, dct_proj] → Linear(1024→2) → Softmax → weighted sum |
| Classifier | Linear(512→128) + ReLU + Dropout(0.3) + Linear(128→2) |
| Weight file | `weights/dtc_rgb_model_v1.pth` |
| Overridable via | `DUAL_BRANCH_WEIGHTS_PATH` env var |
| Training | Fine-tuned on proprietary UAIC/PIMA dataset (not released); EfficientNet-B0 backbone pretrained on ImageNet |

**Input preprocessing** (`preprocessing/image.py`):
- Resize to 224×224
- ToTensor → Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  ← maps to [-1, 1]
- DCT branch: grayscale → 224×224 → DCT2D → zero out top-left `h//8 × w//8` (DC/low-freq) → `log(|dct|+1)` → shape (1, 224, 224)

**Inference parameters:**
| Parameter | Value | Source |
|---|---|---|
| `FULL_IMAGE_THRESHOLD` | `0.10` | `config.py` — sensitive; fake_prob > 0.10 → FAKE |
| `PATCH_THRESHOLD` | `0.50` | patch inference |
| `PATCH_SIZE` | `224` | sliding window |
| `PATCH_STRIDE` | `112` | 50% overlap |
| `PATCH_TOP_K` | `5` | top-k aggregation |

**Output:**
- `score` = `probs[0]` (Fake class probability), range [0,1]
- `label`: "Fake" if `fake_prob > FULL_IMAGE_THRESHOLD` (0.10), else "Real"
- `probabilities`: `{"Real": float, "Fake": float}` from softmax
- Note: class index 0 = Fake, index 1 = Real

**Device:** auto-selects CUDA or CPU; `torch.no_grad()` inference; fp32 on CPU, fp32/fp16 depending on device

---

### 2.2 `detectors/trufor.py` — `TruForAnalyzer`

**Underlying model:** [TruFor (Guillaro et al., 2023)](https://github.com/grip-unina/TruFor)  
Architecture: NoisePrint++ (camera fingerprint anomaly) + RGB features fused in a transformer.

| Parameter | Value |
|---|---|
| Weight file | `weights/TruFor/weights/trufor.pth.tar` (~300 MB) |
| Repo path | `weights/TruFor/` (git clone required) |
| Env override | `TRUFOR_DIR` |
| Training | Off-the-shelf pretrained (GRIP lab, University Federico II) |
| Python import | sys.path insertion of `weights/TruFor/`; imports `networks.trainer.Trainer` |

**Input preprocessing:** BGR conversion via `cv2.COLOR_RGB2BGR`; normalized to [0,1] as float32 tensor; no fixed resize (processed at native resolution).

**Output:**
- `integrity_score` ∈ [0,1] — 0 = tampered, 1 = authentic
- `localization_map` — float32 (H,W) pixel forgery probability
- `confidence_map` — float32 (H,W) prediction reliability
- `detection`: `"tampered"` if integrity < 0.40, `"uncertain"` if < 0.65, else `"authentic"`
- `mask_mean`, `mask_top10_mean`, `mask_spatial_concentration`, `text_zone_activation`, `portrait_zone_activation` — spatial statistics from `_compute_mask_stats()`

**Ensemble conversion:** `signals["trufor"] = 1.0 - integrity_score`

**Fallback:** When TruFor unavailable, `_FallbackAnalyzer` uses SRM residuals as proxy, with `confidence_map` set uniformly to 0.4.

**NoisePrint hook:** Forward hook registered on any module whose name contains `"noiseprint"`, `"noisep"`, `"np_enc"`, `"noise_enc"`, or `"np_branch"` — brittle to upstream architecture changes.

---

### 2.3 `detectors/srm.py` — `SRMAnalyzer`

**Algorithm:** 5 fixed SRM kernels (Fridrich & Kodovsky 2012, Table 1 subset). No GPU, no training.

**Kernels:**

```python
SRM_KERNELS: dict[str, np.ndarray] = {
    "hp3":   np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32) / 4.0,
    "hp5v":  np.array([[0,0,0,0,0],[0,0,0,0,0],[-1,2,-2,2,-1],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32) / 4.0,
    "hp5h":  np.array([[0,0,-1,0,0],[0,0,2,0,0],[0,0,-2,0,0],[0,0,2,0,0],[0,0,-1,0,0]], dtype=np.float32) / 4.0,
    "sq3":   np.array([[-1,0,0,0,1],[0,-1,0,1,0],[0,0,0,0,0],[0,1,0,-1,0],[1,0,0,0,-1]], dtype=np.float32) / 4.0,
    "edge5": np.array([[-1,-1,-1,-1,-1],[-1,1,1,1,-1],[-1,1,8,1,-1],[-1,1,1,1,-1],[-1,-1,-1,-1,-1]], dtype=np.float32) / 8.0,
}
```

**Anomaly score computation:**
```python
# Per-kernel: p95 block variance normalised by SRM_VAR_NORM (0.010)
raw_score = min(p95_var / 0.010, 1.0)

# Weighted aggregate
weights = {"hp3": 1.0, "hp5v": 1.0, "hp5h": 1.0, "sq3": 0.8, "edge5": 1.2}
score = sum(weights[n] * s for n, s in zip(kernels, kernel_scores)) / w_total
```

**JPEG grid inconsistency:**
```python
# Per 8×8 block: mean high-frequency DCT energy (coefficients [4:, 4:])
# p95 inconsistency (variance of HF energy in 3×3 block neighbourhood)
score = min(p95 / 15.0, 1.0)
```

**In pipelines:**
- `id_card`: `combined = 0.70 * srm_score + 0.30 * jpeg_score`
- `document_fraud`: SRM score only (no JPEG blend)
- `vehicle_damage`: SRM score only

**Normalization constant:** `SRM_VAR_NORM = 0.010` (p95 variance; from `config.py`)  
**Threshold:** `SRM_FAKE_THRESHOLD = 0.60` (used in `predict()` wrapper only)  
**Block size:** `block_size=16` default; `block_size=32` when used as TruFor fallback

---

### 2.4 `detectors/dire.py` — `DIREDetector`

**Reference:** Wang et al., "DIRE for Diffusion-Generated Image Detection", ICCV 2023.  
**Extended with:** RF-Solver second-order midpoint inversion for Flux-suspected images.

**SDXL DDIM path:**
| Parameter | Value |
|---|---|
| Backbone | `stabilityai/stable-diffusion-xl-base-1.0` (via HuggingFace) |
| VAE scale | `0.13025` |
| Target resolution | `512×512` |
| Inversion steps | `DIRE_INVERSION_STEPS = 10` (was 20; halved for latency) |
| Threshold | `_THRESHOLD = 0.05` — mean L1 error < 0.05 → `"likely_ai"` |
| Precision | fp16 on CUDA, fp32 on CPU |

**SDXL inference:**
```python
# Uncond embeddings for classifier-free guidance
enc     = torch.zeros(1, 77, 2048, ...)   # _SEQ_LEN=77, _HIDDEN_DIM=2048
pooled  = torch.zeros(1, 1280, ...)       # _POOLED_DIM=1280
time_ids = torch.tensor([[512,512,0,0,512,512]], ...)

# DDIM inversion (clean → noisy)
for t in inverse_scheduler.timesteps:
    noise_pred = unet(z, t, encoder_hidden_states=enc, added_cond_kwargs=...)
    z = inverse_scheduler.step(noise_pred, t, z).prev_sample

# DDIM reconstruction (noisy → clean)
for t in scheduler.timesteps:
    noise_pred = unet(z, t, ...)
    z = scheduler.step(noise_pred, t, z).prev_sample
rec = vae.decode(z / VAE_SCALE).sample
error_map = |orig - rec|.mean(dim=0)
```

**Flux RF-Solver path (CUDA only):**
| Parameter | Value |
|---|---|
| Backbone | `black-forest-labs/FLUX.1-dev` |
| VAE scale | `0.3611`, shift `0.1159` |
| Target resolution | `1024×1024` |
| Integrator | RF-Solver second-order midpoint corrector |

```python
# RF-Solver: second-order midpoint corrector (inversion, t: 0→1)
for i in range(steps):
    dt  = float(sigmas[i + 1] - sigmas[i])
    t_i = float(sigmas[i])
    v1    = _unet(z, t_i)
    x_mid = z + v1 * (dt / 2.0)
    v2    = _unet(x_mid, t_i + dt / 2.0)
    z     = z + v2 * dt

# Reconstruction (t: 1→0)
for i in range(steps - 1, -1, -1):
    dt  = float(sigmas[i] - sigmas[i + 1])   # negative
    t_i = float(sigmas[i + 1])
    v1    = _unet(z, t_i)
    x_mid = z + v1 * (dt / 2.0)
    v2    = _unet(x_mid, t_i + dt / 2.0)
    z     = z + v2 * dt
```

**Flux routing heuristic:** Image suspected as Flux if resolution matches known Flux output sizes `{(1024,1024), (1280,720), (720,1280), (832,1216), (1216,832), (896,1152), (1152,896)}` within ±2% AR tolerance AND no EXIF camera Make/Model tags present.

**Score semantics:** Lower error → more likely AI-generated (inverted from typical [0,1] convention).  
**Fake probability mapping:** `sigmoid((score - threshold) * 100)`, so score=0.05 → fake_prob≈0.5 at boundary.

**Failures:** Requires CUDA; returns `score=None, verdict="unavailable"` on CPU without attempting load.  
**Module singleton:** `get_dire_detector()` returns a module-level instance to avoid reloading SDXL (several GB).

---

### 2.5 `detectors/rigid.py` — `RIGIDDetector`

**Reference:** arXiv 2411.19117, "RIGID: Training-free and Model-agnostic Image Forgery Detection."

**Algorithm:** DINOv2 ViT-L/14 patch perturbation sensitivity.
- Real images: patch embeddings shift MORE under Gaussian noise (higher local Lipschitz constant)
- AI images: patch embeddings shift LESS (generated images lie closer to learned manifold)

**Parameters:**
| Parameter | Value | Source |
|---|---|---|
| Backbone | `dinov2_vitl14` via `torch.hub.load("facebookresearch/dinov2")` | `dino_probe.py` singleton |
| Input size | `518×518` (37×37 patches at patch_size=14) | `DINO_INPUT_SIZE` |
| Fallback size | `224×224` | `DINO_FALLBACK_SIZE` |
| `n_perturbations` | `3` (was 10; reduced for CPU practicality) | `RIGID_N_PERTURBATIONS` |
| `noise_std` | `0.05` (Gaussian, normalised pixel space) | `NOISE_STD` |
| Threshold | `0.12` | `_THRESHOLD` in rigid.py |

**Perturbation loop:**
```python
noise_batch = torch.randn(n_perturbations, *orig_tensor.shape[1:]) * noise_std
for i in range(n_perturbations):
    perturbed = orig_tensor + noise_batch[i:i+1]
    perturbed_patches = forward_patches(model, perturbed)
    cos_dist = (1.0 - (orig_norm * perturbed_norm).sum(dim=-1)).mean().item()
    distances.append(cos_dist)
sensitivity = mean(distances)
```

**Score semantics:** Higher sensitivity → more likely real (inverted). Fake probability:  
`fake_prob = 1.0 / (1.0 + exp((score - threshold) * 50))`

**Normalization:** ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` (from `dino_probe.py`).

**DINOv2 singleton:** Shared with `dino_probe.py`; loaded once, all parameters frozen.

---

### 2.6 `detectors/siglip2.py` — `SigLIP2Detector`

**Backbone:** `google/siglip2-so400m-patch14-384` (full image mode); `google/siglip2-so400m-patch16-naflex` (zone crop mode, lazy-loaded).

**Prompts:**
```python
_AUTHENTIC_PROMPTS = [
    "this is a photo of a real photograph of an official government identity document",
    "this is a photo of a genuine authentic passport scan or id card",
    "this is a photo of a legitimate official photograph of an identity document",
    "this is a photo of a real scanned government issued certificate",
]
_FAKE_PROMPTS = [
    "this is a photo of a computer generated fake artificial identity document",
    "this is a photo of an ai synthesized fake passport or identity card",
    "this is a photo of a digitally manipulated fraudulent government document",
    "this is a photo of a synthetically created artificial fake id card",
]
```

**Scoring (sigmoid, NOT softmax):**
```python
logits = outputs.logits_per_image   # [1, N]
probs  = torch.sigmoid(logits[0])   # sigmoid — required by SigLIP-2's training objective

auth_prob   = probs[:n_auth].mean().item()          # mean of 4 authentic prompts
ai_gen_prob = probs[n_auth:n_auth + 2].mean().item()  # mean of first 2 fake prompts

return float(max(0.0, min(1.0, (ai_gen_prob - auth_prob + 1.0) / 2.0)))
```

**Fallback:** When SigLIP-2 unavailable, falls back to CLIP ViT-L/14 zero-shot via UniversalFakeDetect with softmax over same prompts at temperature `0.05`.

---

### 2.7 `detectors/clip_ufd.py` — `UFDAdapter`

**Architecture:** CLIP ViT-L/14 + `nn.Linear(768, 1)` fine-tune head.  
**Training:** CVPR 2023 synthetic image detection dataset (UniversalFakeDetect, Ojha et al.).

| Parameter | Value |
|---|---|
| Backbone | `CLIP:ViT-L/14` |
| Head weights | `weights/UniversalFakeDetect/pretrained_weights/fc_weights.pth` |
| Repo | git clone of `github.com/Yuheng-Li/UniversalFakeDetect` at `weights/UniversalFakeDetect/` |
| Threshold | `0.5` (default) |

**Preprocessing:**
```python
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_PREPROCESS = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD)])
```

**Output:** `sigmoid(logit)` → `fake_prob` ∈ [0,1]

---

### 2.8 `detectors/dino_probe.py` — `dino_feature_score()`

**Not a `BaseDetector` subclass — functional interface.**

Three zero-training distribution-shift indicators from DINOv2 ViT-L/14:

| Indicator | Real | AI | Formula |
|---|---|---|---|
| Patch token entropy | High norm diversity (std ≈ 0.8–2.0) → score→0 | Homogeneous (std < 0.5) → score→1 | `clip((0.8 - diversity) / 0.6, 0, 1)` |
| CLS token magnitude | — | OOD → higher L2 norm | sigmoid z-score vs. prior mean=44, std=3 |
| Attention head disagreement | Heads specialise (range > 0.10) → score→0 | Heads converge → score→1 | `clip((0.10 - disagree) / 0.08, 0, 1)` |

**Combined:** `0.40 * patch_entropy + 0.35 * cls_magnitude + 0.25 * head_disagreement`

**CLS prior** (`_CLS_PRIOR_MEAN=44.0`, `_CLS_PRIOR_STD=3.0`) recalibrated for ID document images.  
**Input:** resized to `224×224` (not 518; separate from RIGID which uses 518).  
**Attention capture:** forward hook monkey-patches `model.blocks[-1].attn.forward` and restores after inference.

---

### 2.9 `detectors/textshield.py` — `TextShieldDetector`

**Reference:** [TextShield-R1](https://github.com/qcf-568/TextShield)  
**Backbone:** `Qwen/Qwen2.5-VL-7B-Instruct` (zero-shot, no fine-tuning).

**Domain prompts:**
- `"invoice"`: numeric field consistency, font uniformity, text alignment
- `"id_document"`: text sharpness, font consistency, field alignment
- `"default"`: generic document manipulation assessment

**Inference:** Greedy decode (`do_sample=False`, `max_new_tokens=300`); score parsed via progressive regex from model text output. Fallback: 0.5 if no parseable number found.

**Limitations:** Refuses to load on CPU (warns and sets `_TS_LOAD_FAILED=True`). Requires CUDA.

---

### 2.10 `detectors/yolo_damage.py` — `DamageDetector`

**Model:** YOLOv11m fine-tuned on CarDD_COCO dataset by ReverendBayes (~20M parameters).  
**Source:** `https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector`  
**Weight file:** `weights/yolo_damage_detect/trained.pt` (downloaded automatically on first load)  
**Classes:** `dent`, `scratch`, `crack`, `shattered glass`, `broken lamp`, `flat tire`

**Parameters:**
- `conf_threshold = 0.25`
- Inference via `ultralytics.YOLO`

**Role in ensemble:** ROI-localization only — score is **never added to `signals{}`**. Provides `composite_roi()` (union of all detections) and `best_roi()` (highest confidence) for downstream 1.4× expanded crop.

---

### 2.11 `signals/ela.py` — `ela_anomaly_score()`

**Algorithm:** JPEG re-compression at quality 85, per-pixel delta analysis.

```python
QUALITY = 85
delta   = |orig_arr - recomp_arr|.mean(axis=2)   # (H, W)

smooth_score  = clip((0.06 - mean_d) / 0.04, 0, 1)   # low ELA mean → AI-generated
hotspot_score = clip((hotspot_ratio - 0.04) / 0.10, 0, 1)  # high ratio → inpainted
score         = 0.5 * smooth_score + 0.5 * hotspot_score

# hotspot_ratio = fraction of pixels > mean + 2×std
```

**Outputs:** `score`, `heatmap` (uint8 grayscale delta map), `ela_mean`, `cv`, `hotspot_ratio`

---

### 2.12 `signals/prnu.py` — `prnu_anomaly_score()`

**Algorithm:** FFT spatial frequency band ratio (not true PRNU camera fingerprint extraction).

```python
fft_shift = np.fft.fftshift(np.fft.fft2(gray))
power     = |fft_shift|^2

# Radial frequency bands (fraction of max radial freq)
mid_ratio  = power[0.08 < r ≤ 0.40].sum() / total   # AI over-represents
high_ratio = power[r > 0.40].sum() / total            # AI under-represents

mid_score  = clip((mid_ratio  - 0.50) / 0.20, 0, 1)
high_score = clip((0.20 - high_ratio) / 0.15, 0, 1)
fft_score  = clip(0.6 * mid_score + 0.4 * high_score, 0, 1)
```

**Note:** The name "PRNU" is a misnomer — this computes FFT band energy ratios, not Photo Response Non-Uniformity. True PRNU requires multiple images from the same camera.

---

### 2.13 `signals/dct_benford.py` — `dct_benford_score()`

**Algorithm:** First-significant-digit distribution of JPEG AC coefficients vs. Benford's Law.

**Requires `jpegio`** (not in requirements.txt) for raw coefficient access. Falls back to pixel-domain blockiness score when given a decoded array.

```python
_BENFORD = [log10(1 + 1/d) for d in range(1, 10)]

# Chi-square test (8 df)
chi2_stat = sum((observed - expected)^2 / (expected + 1e-10))
p_value   = 1.0 - scipy.stats.chi2.cdf(chi2_stat, df=8)

chi2_score = clip(chi2_stat / 30.0, 0, 1)
zero_score = clip((zero_pct - 0.60) / 0.30, 0, 1)  # high zero fraction = AI quantization
score      = 0.6 * chi2_score + 0.4 * zero_score
```

**Fallback blockiness score** (when given decoded array):
```python
# JPEG 8×8 block boundary difference ratio
# Real JPEG scans: ratio ≈ 1.5–4.0 → score → 0
# AI / uncompressed: ratio ≈ 0.8–1.2 → score → 1
score = clip((1.5 - ratio) / 1.0, 0, 1)
```

---

### 2.14 `signals/cfa.py` — `cfa_correlation_score()`

**Algorithm:** Wiener-filter channel noise cross-correlation.

```python
# Remove JPEG 8×8 blocking structure
R_hp = R - uniform_filter(R, size=8)

# Extract noise residual
r_noise = R_hp - wiener(R_hp, mysize=5)

# Pearson correlation between R and B noise residuals
rb_corr = pearson(r_noise.flat, b_noise.flat)

# Authentic band: [0.05, 0.75]
def _pair_score(corr):
    if corr > 0.75:  return clip((corr - 0.75) / 0.25, 0, 1)   # over-correlated → AI
    if corr < 0.05:  return clip((0.05 - corr) / 0.15, 0, 1)   # under-correlated → AI
    return 0.0

score = max(rb_score, rg_score)
```

---

### 2.15 `signals/metadata.py` — `analyze_metadata()`

**Algorithm:** Weighted EXIF check battery (no ML).

| Check | Weight | Notes |
|---|---|---|
| `camera_make_model` | 0.20 | Known camera brands list |
| `exposure_triangle` | 0.20 | f/number, exposure time, ISO |
| `focal_length` | 0.10 | Plausibility range 4–2000mm |
| `datetime_original` | 0.10 | DateTimeOriginal > DateTime |
| `gps_data` | 0.05 | GPSInfo block |
| `software_signature` | 0.10 | AI tool strings → score=0 |
| `thumbnail_present` | 0.10 | EXIF tags 513/514 |
| `pixel_count` | 0.05 | ≥1MP |
| `exif_consistency` | 0.05 | DateTime vs DateTimeOriginal delta |
| `image_format` | 0.05 | JPEG=1.0, PNG=0.2 |

**Ensemble conversion:** `signals["metadata"] = 1.0 - authenticity_score`

**AI software detection strings:** `"stable diffusion"`, `"midjourney"`, `"dall-e"`, `"dalle"`, `"firefly"`, `"imagen"`, `"generative fill"`, `"ai generated"`, `"adobe firefly"`, `"canva"`

---

## 3. Ensemble Scorer (`ensemble/scorer.py`)

### 3.1 Weight Presets

```python
VEHICLE_DAMAGE_WEIGHTS: dict[str, float] = {
    "trufor":      0.35,
    "dual_branch": 0.28,
    "clip_ufd":    0.22,
    "rigid":       0.10,
    "srm":         0.05,
}  # sum = 1.00
# Note: yolo_damage excluded — ROI only, not fraud scored

DOCUMENT_FRAUD_WEIGHTS: dict[str, float] = {
    "trufor":   0.25,   # was 0.30
    "srm":      0.20,   # unchanged
    "siglip2":  0.23,   # +0.03 from freed weak signals
    "dire":     0.18,   # +0.03 from freed weak signals
    "rigid":    0.10,   # unchanged
    "ela":      0.02,   # was 0.05 — demoted, weak on diffusion-era generators
    "prnu":     0.02,   # was 0.05 — demoted
}  # sum = 1.00

ID_CARD_WEIGHTS: dict[str, float] = {
    "trufor":      0.25,   # was 0.30
    "srm":         0.20,   # unchanged
    "dire":        0.19,   # +0.04 from freed weak signals
    "dual_branch": 0.15,   # was 0.20
    "rigid":       0.14,   # +0.04 from freed weak signals
    "ela":         0.02,   # was 0.05 — demoted
    "prnu":        0.02,   # was 0.05 — demoted
    "dct_benford": 0.01,   # was 0.03 — demoted
    "cfa":         0.02,   # unchanged
}  # sum = 1.00
```

### 3.2 Fusion Math

```python
# Filter to available signals with known base weights
active = {k: float(v) for k, v in signals.items() if k in base_weights and v is not None}

# Renormalize weights for missing signals
total_w      = sum(base_weights[k] for k in active)
weights_used = {k: base_weights[k] / total_w for k in active}

ensemble_score = min(max(
    sum(active[k] * weights_used[k] for k in active),
    0.0), 1.0)
```

Missing signals are excluded and remaining weights renormalize to sum=1. Empty signals → `ensemble_score=0.5`.

### 3.3 Welford Online Normalization

**Scope:** Within-session only — state resets on app restart, not persisted to disk.

```python
def _welford_update(self, key: str, value: float) -> float:
    if key not in self._welford_state:
        self._welford_state[key] = {"n": 0, "mean": 0.0, "M2": 0.0}
    state = self._welford_state[key]
    state["n"] += 1
    n     = state["n"]
    delta = value - state["mean"]
    state["mean"] += delta / n
    state["M2"]   += delta * (value - state["mean"])
    if n < 2:
        return float(value)   # first call: pass through unchanged
    variance = state["M2"] / (n - 1)
    std      = math.sqrt(variance) if variance > 1e-12 else 1.0
    z        = (value - state["mean"]) / std
    return float(min(max(0.5 + z / 6.0, 0.0), 1.0))
    # z-score mapped to [0,1] via 0.5 + z/6 (approx ±3σ → [0,1])
```

`RunningStats` class (separate from `_welford_update`) uses the same algorithm but is not connected to `_score_fixed` — the scorer uses the inline dict-based Welford.

### 3.4 Trained Meta-Classifier

```python
_TRAINED_MODEL_PATH = WEIGHTS_DIR / "ensemble_meta_clf_v1.pkl"
_TRAINED_HASH_PATH  = WEIGHTS_DIR / "ensemble_meta_clf_v1.sha256"
```

**Algorithm:** Not specified in code (loaded via joblib from pkl; referenced as "Gradient Boosting" in docstring).  
**Feature vector (5 features):**
```python
feat = [
    signals.get("dual_branch", 0.5),
    signals.get("clip_ufd",    0.5),
    signals.get("trufor",      0.5),
    signals.get("srm",         0.5),
    signals.get("metadata",    0.5),
]
```
Missing signals default to `0.5` (neutral). This creates a silent information loss issue when signals are genuinely unavailable.

**SHA-256 verification:**
```python
expected = _TRAINED_HASH_PATH.read_text().strip().lower()
actual   = hashlib.sha256(_TRAINED_MODEL_PATH.read_bytes()).hexdigest().lower()
if actual != expected:
    raise _ModelSecurityError(...)
```

**Fallback:** File not found or any load exception → falls back to `_score_fixed()` with `mode="fixed_fallback"` and `_fallback_reason` field.

### 3.5 Verdict Threshold Logic

```python
@staticmethod
def _verdict(score: float) -> str:
    if score < THRESH_AUTHENTIC:    return "Authentic"          # < 0.35
    if score < THRESH_SUSPICIOUS:   return "Suspicious"         # 0.35 ≤ score < 0.60
    return "Likely Fraudulent"                                   # ≥ 0.60
```

### 3.6 Confidence/Spread Logic

```python
@staticmethod
def _confidence(breakdown: list) -> str:
    scores = [row["score"] for row in breakdown if row["available"] and row.get("score") is not None]
    if len(scores) < 2:
        return "LOW"
    spread = max(scores) - min(scores)
    if spread < SPREAD_HIGH:    return "HIGH"     # spread < 0.20
    if spread < SPREAD_MEDIUM:  return "MEDIUM"   # spread < 0.40
    return "LOW"
```

Confidence measures inter-signal agreement, not model probability calibration.

---

## 4. Pipeline Definitions (`pipelines.py`)

### 4.1 `_DETECTOR_CACHE` — Module-level singleton cache

```python
_DETECTOR_CACHE: dict[str, object] = {}

def _cached(key: str, factory):
    """Return a cached detector instance, creating it on first call."""
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = factory()
    return _DETECTOR_CACHE[key]
```

Key = detector class name (e.g. `"srm"`, `"trufor"`, `"dual_branch"`). Module-level — persists for the process lifetime. **Not thread-safe** (no lock around `_DETECTOR_CACHE` writes). `run_custom_pipeline` does NOT use `_cached` — creates fresh instances every call.

### 4.2 Image validation

```python
def _validate_image(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w < IMAGE_MIN_SIZE or h < IMAGE_MIN_SIZE:   # IMAGE_MIN_SIZE = 64
        raise ValueError(...)
    return img.convert("RGB") if img.mode != "RGB" else img
```

### 4.3 `run_id_card_pipeline`

| Step | Signal | Input | Weight |
|---|---|---|---|
| 1 | `dual_branch` | full image | 0.15 |
| 2 | `srm` | full image | 0.20; `0.70*srm + 0.30*jpeg` blend |
| 3 | `trufor` | full image | 0.25 |
| 4 | `prnu` | full image (array) | 0.02 |
| 5 | `dct_benford` | full image (array) | 0.01 |
| 6 | `cfa` | full image (array) | 0.02 |
| 7 | `ela` | full image | 0.02 |
| Scorer | `EnsembleScorer(ID_CARD_WEIGHTS)` | | `mode="fixed"` |

Note: `dire` and `rigid` appear in `ID_CARD_WEIGHTS` (weights 0.19, 0.14) and in `PRESET_SIGNALS["id_card"]` but are **absent from `run_id_card_pipeline`'s try/except blocks** — they run only via `run_custom_pipeline`. This is a discrepancy between weight config and pipeline implementation.

### 4.4 `run_document_fraud_pipeline`

| Step | Signal | Input | Weight |
|---|---|---|---|
| 1 | `siglip2` | full image | 0.23 |
| 2 | `trufor` | full image | 0.25; also extracts `noiseprint_localizer`, `mask_spatial_concentration` |
| 3 | `srm` | full image | 0.20; SRM score only (no JPEG blend) |
| 4 | `prnu` | full image (array) | 0.02 |
| 5 | `ela` | full image | 0.02 |
| Scorer | `EnsembleScorer(DOCUMENT_FRAUD_WEIGHTS)` | | `mode="fixed"` |

Same gap as id_card: `dire` (0.18) and `rigid` (0.10) are in weights but not executed in the preset function.

### 4.5 `run_vehicle_damage_pipeline`

| Step | Signal | Input | Notes |
|---|---|---|---|
| 1 | `yolo_damage` | full image | ROI only; not scored |
| 2 | `dual_branch` | full or ROI crop | 0.28 |
| 3 | `trufor` | full or ROI crop | 0.35 |
| 4 | `srm` | full or ROI crop | 0.05 |
| 5 | `metadata` | full image (always) | 0.10; note: metadata runs on full image regardless of ROI crop |

**ROI expansion:** When `use_damage_roi=True` and YOLO detects damage:
```python
cx = comp["x"] + comp["width"]  / 2
cy = comp["y"] + comp["height"] / 2
nw = comp["width"]  * 1.4          # 1.4× expansion
nh = comp["height"] * 1.4
x1, y1 = max(0, cx-nw/2), max(0, cy-nh/2)
x2, y2 = min(iw, cx+nw/2), min(ih, cy+nh/2)
```

**Lazy-import / skip-on-failure pattern:**
```python
try:
    from .detectors.dual_branch import DualBranchDetector
    r = _cached("dual_branch", DualBranchDetector).predict(img)
    signals["dual_branch"] = r["score"]
except (ImportError, RuntimeError, OSError, ValueError) as e:
    skipped.append(f"dual_branch ({type(e).__name__}: {e})")
```

### 4.6 `run_custom_pipeline`

Accepts any combination from `ALL_SIGNAL_KEYS`. Execution order: YOLO first (for ROI) → `dual_branch` → `trufor` → `srm` → `siglip2` → `clip_ufd` → `ela` → `prnu` → `dct_benford` → `cfa` → `metadata` → `dire` → `rigid` → `textshield`.

When `manual_roi` supplied: crops image before any signal runs (takes precedence over `use_damage_roi`).  
When `preset_weights` supplied: uses those; otherwise equal weights (1.0 per selected signal).  
Creates **fresh detector instances** (no `_cached`) — models reload on every call.

---

## 5. Preprocessing (`preprocessing/`)

### 5.1 `preprocessing/image.py` — `preprocess_image()`

Used exclusively by DualBranchDetector.

| Step | Detail |
|---|---|
| Mode | Convert to RGB if not already |
| Resize | `224×224` (torchvision `transforms.Resize`) |
| Normalize | mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] → maps to [-1, 1] |
| Output shape | `[1, 3, 224, 224]` float32 |

### 5.2 `preprocessing/dct.py` — `extract_dct_high_freq()`

Used by DualBranchDetector's DCT branch.

```
RGB → grayscale (cv2.COLOR_RGB2GRAY)
→ resize to 224×224 (cv2.resize)
→ /255.0
→ cv2.dct() (2D DCT-II)
→ zero out [: h//8, : w//8] (suppress DC + low frequencies; 28×28 block zeroed)
→ log(|dct| + 1)
→ add channel dim → (1, 224, 224) float32
```

### 5.3 Detector-specific normalization

| Detector | Mean | Std | Size | Source |
|---|---|---|---|---|
| DualBranch | [0.5, 0.5, 0.5] | [0.5, 0.5, 0.5] | 224×224 | `preprocessing/image.py` |
| CLIP/UFD | [0.4815, 0.4578, 0.4082] | [0.2686, 0.2613, 0.2758] | 224×224 (center crop of 256) | `clip_ufd.py` |
| DINOv2 (RIGID/probe) | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | 518×518 (224×224 in dino_probe) | `dino_probe.py` |
| TruFor | None (raw BGR /255) | — | Native resolution | `trufor.py` |
| DIRE (SDXL) | — | — | 512×512; (arr/127.5)-1.0 | `dire.py` |
| DIRE (Flux) | — | — | 1024×1024; (arr/127.5)-1.0 | `dire.py` |

SRM, ELA, PRNU, DCT-Benford, CFA all work in grayscale float or RGB normalized to [0,1] internally.

---

## 6. Configuration (`config.py`)

All constants with current values:

| Constant | Value | Controls |
|---|---|---|
| `WEIGHTS_DIR` | `<project_root>/weights` | Root for all weight files |
| `DUAL_BRANCH_WEIGHTS` | `weights/dtc_rgb_model_v1.pth` | DualBranchDetector checkpoint |
| `TRUFOR_DIR` | `weights/TruFor` | TruFor repo path |
| `TRUFOR_WEIGHTS` | `weights/TruFor/weights/trufor.pth.tar` | TruFor checkpoint |
| `UFD_REPO` | `weights/UniversalFakeDetect` | CLIP/UFD repo clone |
| `UFD_WEIGHTS` | `weights/UniversalFakeDetect/pretrained_weights/fc_weights.pth` | UFD linear head |
| `YOLO_MODEL_DIR` | `weights/yolo_damage_detect` | YOLO weights dir |
| `YOLO_CACHED_WEIGHTS` | `weights/yolo_damage_detect/trained.pt` | YOLO checkpoint |
| `YOLO_GITHUB_URL` | (ReverendBayes repo raw URL) | Download fallback |
| `THRESH_AUTHENTIC` | `0.35` | ensemble_score < 0.35 → "Authentic" |
| `THRESH_SUSPICIOUS` | `0.60` | ensemble_score ≥ 0.60 → "Likely Fraudulent" |
| `SPREAD_HIGH` | `0.20` | inter-signal spread < 0.20 → "HIGH" confidence |
| `SPREAD_MEDIUM` | `0.40` | inter-signal spread < 0.40 → "MEDIUM" confidence |
| `FULL_IMAGE_THRESHOLD` | `0.10` | DualBranch fake_prob > this → label="Fake" (sensitive) |
| `PATCH_THRESHOLD` | `0.50` | DualBranch patch inference label boundary |
| `PATCH_SIZE` | `224` | DualBranch sliding window patch size |
| `PATCH_STRIDE` | `112` | DualBranch sliding window stride (50% overlap) |
| `PATCH_TOP_K` | `5` | DualBranch top-k patch aggregation count |
| `IMAGE_MIN_SIZE` | `64` | Minimum input dimension for any pipeline |
| `DINO_INPUT_SIZE` | `518` | DINOv2 ViT-L/14 preferred input (37×37 patches) |
| `DINO_FALLBACK_SIZE` | `224` | DINOv2 fallback if 518 causes position-embed error |
| `NOISE_STD` | `0.05` | RIGID Gaussian noise std (normalised pixel space) |
| `RIGID_N_PERTURBATIONS` | `3` | RIGID perturbation count (was 10; reduced for CPU) |
| `DIRE_INVERSION_STEPS` | `10` | DIRE DDIM/RF-Solver inversion steps (was 20) |
| `SRM_FAKE_THRESHOLD` | `0.60` | SRM score ≥ this → "Fake" in `predict()` wrapper |
| `SRM_VAR_NORM` | `0.010` | p95 variance normalization factor for SRM kernels |

All path constants are overridable via environment variables: `DUAL_BRANCH_WEIGHTS_PATH`, `TRUFOR_DIR`, `UFD_REPO`, `YOLO_MODEL_DIR`.

---

## 7. Data Flow

### CLI Entry Point (`main.py`)

```
python main.py --image path/to/image.jpg --pipeline id_card [--json]
```

1. `argparse` parses `--image`, `--pipeline`, `--json`
2. `Image.open(img_path)` → PIL Image (mode unspecified at this point)
3. Dispatch to `run_id_card_pipeline(img)` / `run_document_fraud_pipeline` / `run_vehicle_damage_pipeline`
4. Output: formatted text or JSON (numpy arrays replaced with `"<ndarray>"`)

**No document type auto-classification in CLI.** The `document/classifier.py` module exists but is not invoked by any pipeline or the CLI — it's a standalone utility for OCR-text-based routing.

### Single Image Trace (id_card pipeline)

```
PIL Image (any mode, any size)
    │
    ▼ _validate_image()
    │   • Reject if w < 64 or h < 64 → ValueError
    │   • Convert to RGB if needed
    │
    ▼ convert to np.array for signal functions
    │
    ├──► DualBranchDetector.predict(img)
    │       • preprocess_image() → (rgb_tensor [1,3,224,224], dct_tensor [1,1,224,224])
    │       • DualBranchModel forward → softmax → fake_prob
    │       • signals["dual_branch"] = fake_prob
    │
    ├──► SRMAnalyzer.compute_anomaly_score(img) + detect_jpeg_grid_inconsistency(img)
    │       • extract_residuals() → 5-kernel filter2D on grayscale
    │       • block variance map → p95 → score
    │       • signals["srm"] = 0.70 * srm_score + 0.30 * jpeg_score
    │
    ├──► TruForAnalyzer.analyze(img)
    │       • sys.path insert → load Trainer from TruFor repo
    │       • img_bgr → tensor /255 → trainer.test()
    │       • signals["trufor"] = 1.0 - integrity_score
    │
    ├──► prnu_anomaly_score(img_arr)
    │       • FFT → power spectrum → band ratios
    │       • signals["prnu"] = fft_score
    │
    ├──► dct_benford_score(img_arr)
    │       • Blockiness fallback (not JPEG input here)
    │       • signals["dct_benford"] = block_score
    │
    ├──► cfa_correlation_score(img_arr)
    │       • Wiener residual R/B/G cross-correlation
    │       • signals["cfa"] = max(rb_score, rg_score)
    │
    ├──► ela_anomaly_score(img)
    │       • JPEG re-compress at quality 85 → delta map
    │       • signals["ela"] = 0.5*smooth_score + 0.5*hotspot_score
    │
    ▼ EnsembleScorer(custom_weights=ID_CARD_WEIGHTS).score(signals, mode="fixed")
        • Filter to available signals
        • Renormalize weights
        • Weighted average → ensemble_score
        • _verdict() → "Authentic" / "Suspicious" / "Likely Fraudulent"
        • _confidence() → "HIGH" / "MEDIUM" / "LOW" (inter-signal spread)
        │
        ▼ return dict:
            pipeline, verdict, score, confidence, signals, ensemble, skipped
```

**Caching:** `_DETECTOR_CACHE` keeps `SRMAnalyzer`, `DualBranchDetector`, `TruForAnalyzer` alive across repeated calls within the same process.

**Error handling:** Each signal wrapped in `try/except (ImportError, RuntimeError, OSError, ValueError)` — failures append to `skipped[]` and do not abort the pipeline. Missing signals are auto-renormalized in the ensemble.

**Batching / async:** None. All inference is synchronous, single-image.

**No Streamlit `app.py` found** in the repository. UI components exist in `src/ai_image_detection/ui/components.py` but no top-level app file is present.

---

## 8. Explainability Layer (`explainability/`)

### 8.1 `explainability/gradcam.py` — `generate_gradcam()`

**Target layer:** `model.rgb_encoder.features[8]` — the last EfficientNet-B0 feature block (index 8 of `features` Sequential, which in EfficientNet-B0 is the final MBConv stage before the classifier).

**Hook implementation:**
```python
target_layer = model.rgb_encoder.features[8]
fwd_handle   = target_layer.register_forward_hook(_fwd_hook)   # captures activations
bwd_handle   = target_layer.register_full_backward_hook(_bwd_hook)  # captures gradients
```

**Grad-CAM computation:**
```python
act    = activations[0].squeeze(0)       # (C, H', W')
grad   = gradients[0].squeeze(0)         # (C, H', W')
weights = grad.mean(dim=(1, 2))          # global avg pool → (C,)
cam    = F.relu((weights[:, None, None] * act).sum(dim=0))   # (H', W')
# Normalize, resize to original image size, apply JET colormap
heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
```

**ROI paste-back** (`gradcam_for_roi`): Runs Grad-CAM on the crop, then `result.paste(overlay, (x, y))` into a copy of the full image.

### 8.2 `explainability/heatmap.py`

Two additional visualizations:

1. **`patch_verdict_grid()`** — per-patch inference overlay (112×112 patches, 56 stride). Colors: RED (fake_prob > 0.5) or GREEN (real), opacity proportional to confidence. `PATCH_FLAG_THRESHOLD = 0.50`.

2. **`dct_artifact_map()`** — renders the DCT high-frequency map from `extract_dct_high_freq()` as grayscale and INFERNO colormap.

3. **`build_evidence_panel()`** — convenience wrapper returning `(original, patch_grid, dct_color, stats)`.

### 8.3 UI exposure

`src/ai_image_detection/ui/components.py` contains UI component code. The inventory does not include the full Streamlit app (`app.py` not found in repository). Grad-CAM, patch grid, and DCT map are all plumbed for UI rendering but the app launcher is absent from the codebase at time of this inventory.

---

## 9. External Repositories

| Dependency | Source URL | Integration | Pinned commit | Modifications |
|---|---|---|---|---|
| TruFor | `github.com/grip-unina/TruFor` | git clone into `weights/TruFor/`; `sys.path.insert(0, ...)` | None | None; adapter validates clone has `networks/` dir before path insertion |
| UniversalFakeDetect | `github.com/Yuheng-Li/UniversalFakeDetect` | git clone into `weights/UniversalFakeDetect/`; `sys.path.insert(0, ...)` | None | None; adapter validates clone has `models/` dir |
| YOLO11m-Car-Damage-Detector | `github.com/ReverendBayes/YOLO11m-Car-Damage-Detector` | weights-only download (`trained.pt`) via `urllib.request.urlretrieve` | None | None |
| DINOv2 ViT-L/14 | `facebookresearch/dinov2` via `torch.hub` | `torch.hub.load(pretrained=True)` | None (latest from hub) | Parameters frozen; `blocks[-1].attn.forward` temporarily monkey-patched for attention capture in `dino_probe.py` |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` via HuggingFace | `diffusers.StableDiffusionXLPipeline.from_pretrained` | None | DDIMInverseScheduler config filters out unsupported keys |
| Flux | `black-forest-labs/FLUX.1-dev` via HuggingFace | `diffusers.FluxPipeline.from_pretrained` | None | Custom RF-Solver second-order midpoint integrator |
| SigLIP-2 | `google/siglip2-so400m-patch14-384`, `google/siglip2-so400m-patch16-naflex` via HuggingFace | `transformers.AutoModel.from_pretrained` | None | None |
| Qwen2.5-VL-7B-Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` via HuggingFace | `transformers.AutoModelForVision2Seq.from_pretrained` | None | None |

**Security notes:**
- TruFor and UFD repos are validated before `sys.path` insertion (directory structure check).
- `ensemble_meta_clf_v1.pkl` joblib file is SHA-256 verified before loading.
- No pinned commit hashes — any of the HuggingFace models or GitHub repos could change upstream.

---

## 10. Known TODOs and Caveats

**Result from `grep -rn "TODO|FIXME|HACK|XXX|NOTE" src/ --include="*.py"`:**

| Location | Tag | Text |
|---|---|---|
| `detectors/dire.py:376` | `NOTE` | `# NOTE: the block above either returns early or sets self._loaded = True. No further "if not self._loaded" check is reachable here.` |

**No TODO, FIXME, HACK, or XXX annotations found in the codebase.**

**Undocumented caveats (from code reading):**

1. **`dire` and `rigid` missing from preset pipeline functions**: Both signals have weights in `ID_CARD_WEIGHTS` and `DOCUMENT_FRAUD_WEIGHTS`, and appear in `PRESET_SIGNALS`, but `run_id_card_pipeline()` and `run_document_fraud_pipeline()` do not invoke them. They are only reachable via `run_custom_pipeline()`.

2. **`run_custom_pipeline` does not use `_DETECTOR_CACHE`**: Creates fresh `DualBranchDetector()`, `TruForAnalyzer()`, etc. on every call — model weights reload each invocation.

3. **DIRE CPU guard is pre-load only**: Once loaded on GPU, if device context changes, the guard does not re-trigger.

4. **`dct_benford_score` in pipelines receives a decoded numpy array** (not JPEG bytes/path), so it always falls back to the `_blockiness_score` approximation. The Benford chi-square path requires `jpegio` and a raw JPEG path, which no current pipeline provides.

5. **Welford z-score mapped as `0.5 + z/6`**: At ±3σ, score reaches 0.0/1.0. Signals with very low variance across a session will have std ≈ 0, clamped to 1.0, making normalization a no-op.

6. **`textshield` in `PRESET_SIGNALS`** for id_card and document_fraud but not invoked in the preset pipeline functions.

7. **`document/classifier.py` is a dead utility**: No pipeline or CLI path calls `classify_document()` at runtime — it exists for potential OCR-based pipeline routing but is not wired up.

8. **PRNU signal is misnamed**: Implements FFT band energy ratio, not true Photo Response Non-Uniformity extraction (which requires multiple images from the same camera).

---

## 11. Test Coverage

### Test Files

| File | Description |
|---|---|
| `tests/conftest.py` | Shared fixtures: `sample_image` (224×224 gray), `sample_image_path` (skip if no sample files) |
| `tests/unit/test_detectors.py` | Unit tests for SRMAnalyzer, BaseDetector ABC, RIGIDDetector model-free paths, DIREDetector CPU/Flux heuristics, TextShieldDetector parsing |
| `tests/unit/test_scorer.py` | Unit tests for RunningStats (Welford), EnsembleScorer fixed/Welford/trained-fallback modes |
| `tests/integration/test_pipelines.py` | Integration tests for all 3 preset pipelines + custom pipeline, schema validation, image size guard, detector cache, CLI JSON output |

### Test Descriptions

**`tests/unit/test_detectors.py`:**
- `TestSRMAnalyzer` — 6 tests: predict schema, anomaly score, small image guard (returns 0.5, not crash), residual shape, JPEG grid score, large image
- `TestBaseDetectorContract` — 2 tests: ABC instantiation raises TypeError, incomplete subclass raises TypeError
- `TestRIGIDDetectorSafePaths` — 2 tests: DINOv2 unavailable → score=None; default perturbation count = 3
- `TestDIREDetectorSafePaths` — 4 tests: CPU → unavailable; Flux heuristic on 1024×1024; Flux heuristic on 800×600; singleton identity
- `TestTextShieldSafePaths` — 4 tests: unavailable model → score=None; `_parse_score` with explicit label, no number, score=1.0

**`tests/unit/test_scorer.py`:**
- `TestRunningStats` — 4 tests: single-value std=1.0, mean update, positive std with 2 values, reset clears state
- `TestEnsembleScorerFixed` — 8 tests: all-zero → Authentic, all-one → Fraudulent, valid range, missing signals renormalize, empty → 0.5, confidence field, weights sum to 1, mode field
- `TestEnsembleScorerWelford` — 3 tests: first call passthrough, second call shifts, reset clears
- `TestEnsembleScorerTrainedFallback` — 1 test: pkl absent → graceful fallback

**`tests/integration/test_pipelines.py`:**
- Schema validation for id_card, document_fraud, vehicle_damage, custom pipelines
- Determinism check for id_card pipeline
- Custom pipeline with empty selection → score=0.5
- Image size guard for all 3 preset pipelines
- Detector cache identity test (SRMAnalyzer reused across calls)
- CLI JSON output validity

### Zero-coverage detectors/signals

| Component | Test coverage |
|---|---|
| `DualBranchDetector` | Zero unit tests (integration tests exercise it via pipeline, but fail gracefully if torch absent) |
| `TruForAnalyzer` | Zero tests; fallback path untested directly |
| `SigLIP2Detector` | Zero tests |
| `UFDAdapter` / `clip_ufd` | Zero tests |
| `DINOv2 dino_probe` | Zero tests |
| `TextShieldDetector` (inference path) | Zero tests (model-free path tested, not actual inference) |
| `DIREDetector` (GPU inference path) | Zero tests |
| `DamageDetector` (YOLO) | Zero tests |
| `ela_anomaly_score` | Zero unit tests (exercised indirectly in integration) |
| `prnu_anomaly_score` | Zero unit tests |
| `dct_benford_score` | Zero unit tests |
| `cfa_correlation_score` | Zero unit tests |
| `analyze_metadata` | Zero tests |
| `classify_document` | Zero tests |
| Grad-CAM / heatmap | Zero tests |
| Preprocessing (`dct.py`, `image.py`) | Zero dedicated tests |

### Test fixtures

- All tests use synthetic PIL images (`Image.new`) — no real photographs in the test suite.
- `samples/` directory referenced in `conftest.py` but tests skip if empty (real sample images are gitignored).
- No fixtures for GPU paths, CUDA availability, or external model availability — GPU-dependent tests rely on graceful fallback to `score=None`.
