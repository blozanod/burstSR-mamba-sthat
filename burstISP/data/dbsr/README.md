# Vendored: DBSR synthetic burst pipeline

Official synthetic burst generation, validation-set loading and PSNR evaluation
code for the SyntheticBurst benchmark, vendored so that MambaFusion's L3/L4
experiments (see `PLAN.md`) use the standard protocol wholesale instead of a
re-implementation.

## Provenance

- **Source repository:** https://github.com/goutamgmb/deep-burst-sr
  ("Deep Burst Super-Resolution", Bhat et al., CVPR 2021 — the DBSR toolkit,
  also the basis of the NTIRE burst super-resolution challenge kits)
- **Commit:** `16f158e0553b6702cc1693c43f733c2830973278` (2023-11-26)
- **License:** CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0
  International), Copyright (c) 2021 Huawei Technologies Co., Ltd.
  Released **for academic research use only**; commercial use requires
  contacting Huawei. Every vendored file retains the original license header.
  Note this is more restrictive than the top-level license of this repository —
  the vendored folder remains under CC BY-NC-SA 4.0.

## Files

| File here | Upstream path | Adaptations |
|---|---|---|
| `camera_pipeline.py` | `data/camera_pipeline.py` | none (verbatim) |
| `processing_utils.py` | `data/processing_utils.py` | none (verbatim) |
| `data_format_utils.py` | `utils/data_format_utils.py` | none (verbatim) |
| `synthetic_burst_generation.py` | `data/synthetic_burst_generation.py` | 2 import lines re-pointed to this package (`data.camera_pipeline` → `burstISP.data.dbsr.camera_pipeline`, `utils.data_format_utils` → `burstISP.data.dbsr.data_format_utils`); everything else verbatim |
| `synthetic_burst_val_set.py` | `dataset/synthetic_burst_val_set.py` | removed the `admin.environment` import; `root` is a required argument instead of falling back to `env_settings()` |
| `image_quality.py` | `models/loss/image_quality_v2.py` | only `PixelWiseError` and `PSNR` kept (verbatim bodies); `SSIM`/`LPIPS`/`AlignedL2` omitted to avoid the `msssim`/`lpips`/PWC-Net dependencies not needed for the headline PSNR |

## Official protocol constants (from `train_settings/dbsr/default_synthetic.py` upstream)

- `crop_sz = (384, 384)`, `downsample_factor = 4`
- `burst_transformation_params = {'max_translation': 24.0, 'max_rotation': 1.0,
  'max_shear': 0.0, 'max_scale': 0.0, 'border_crop': 24}`
- `image_processing_params = {'random_ccm': True, 'random_gains': True,
  'smoothstep': True, 'gamma': True, 'add_noise': True}`
- Upstream DBSR trained with `burst_sz = 8`; the benchmark validation set and
  the later burst-SR literature (BIPNet, Burstormer, BSRT, …) use 14 frames,
  which is what MambaFusion's L3/L4 configs use.
- Evaluation (`evaluation/synburst/compute_score.py` upstream): predictions are
  quantized to 14-bit (`(pred.clamp(0, 1) * 2**14).short() / 2**14`) and scored
  with `PSNR(boundary_ignore=40)` against the linear-RGB ground truth.

## Consumers in this repository

- `burstISP/data/synthetic_burst_dataset.py` — `SyntheticBurstDataset`
  (train-time on-the-fly generation from Zurich RAW-to-RGB, val-time official
  300-burst set).
- `burstISP/metrics/synburst_psnr.py` — `calculate_psnr_synburst`, the
  registered metric wrapping `image_quality.PSNR` plus the official
  quantization step.
