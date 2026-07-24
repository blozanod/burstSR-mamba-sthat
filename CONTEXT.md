# MambaFusion — Repository Context

## Project Overview

MambaFusion is a **RAW burst super-resolution** model. Given a burst of N short-exposure, low-quality RAW frames, the model produces a single high-quality RGB image at 4× spatial upscale (8× in the packed RGGB coordinate system used throughout the code).

The project is a research prototype. Training runs on an HPC cluster (4× GPU). This local WSL2 repo is used for code development and inference/analysis only — the full dataset lives at `/groups/rls/blozanod/MambaFusion/dataset/` on the cluster.

**Current direction (July 2026):** see [PLAN.md](PLAN.md) — porting to the standard SyntheticBurst benchmark as the primary instrument, with a diagnosis-led repair of the burst-utilization failure documented below.

---

## Dataset: RealBSR-RAW

- Paper: `papers/RealBSR-RAW.pdf`
- Each sample is a **burst folder** containing:
  - 14 LQ RAW frames: `*_x1_*.png` — stored as 4-channel packed RGGB, shape `[H/2, W/2, 4]`, 16-bit
  - 1 GT RGB image: `*_x4_rgb.png` — 48-bit RGB, at 4× the spatial resolution of the raw frames
  - 1 metadata file: `*.pkl` — holds camera info (`black_level_subtracted`, WB gains, CCM, etc.)
- **Black level**: 512 is subtracted if not already done per `meta_data['black_level_subtracted']`, then normalized to `[0, 1]` by dividing by 16383.
- **Scale convention**: LQ is packed RGGB at `[H, W, 4]` (so real spatial resolution is `[2H, 2W]` after unpacking). GT is at `[4×2H, 4×2W]` real pixels = `[8H, 8W]` in packed coordinates, hence `scale=8` in all configs.
- During training, 5 of the 14 frames are randomly sampled per iteration; the reference frame (index 0) is always placed at the center position `N//2`.
- The only data augmentation is a random RGGB-aware transpose (swaps G1↔G2 channels when transposing).
- Local preview dataset: `dataset/Inference_Set/` — 10 test-set bursts with GT.
- **Known confound**: the GT is captured at a different optical zoom and registered to the reference imperfectly, so pixel losses are partly optimized against misregistration noise (this is why eval crops a 40 px border, and part of the motivation for the SyntheticBurst port).

---

## Dataset: SyntheticBurst (L3/L4)

- Standard benchmark protocol (DBSR/NTIRE), run wholesale via code vendored from the DBSR toolkit at `burstISP/data/dbsr/` (provenance + license in its README; CC BY-NC-SA 4.0, academic use only).
- `SyntheticBurstDataset` (`burstISP/data/synthetic_burst_dataset.py`): **train** generates bursts on the fly from Zurich RAW-to-RGB canon JPGs (official parameters: 432-px padded random crop → inverse ISP → 14 random affine frames, translation ≤24 px / rotation ≤1°, ×4 downsample, RGGB mosaic, shot+read noise → `lq [14, 4, 48, 48]`, `gt [3, 384, 384]` linear RGB); **val** loads the official pre-generated 300-burst set exactly as distributed.
- The generator's reference frame (index 0) is moved to the center slot `N//2` to match this codebase's reference convention (same trick as `BurstImageDataset`); ground-truth `flow_vectors` (LR-RGB resolution, pre-warp geometry) ride along in every train sample.
- `oracle_align: true` (L4) warps each training frame onto the reference with those flows before returning the burst. The official val set ships no flows, so val inputs are never oracle-warped.
- Samples carry **no `meta` key** (no camera pkl); the val loop and `save_img` treat metadata as optional. Headline eval is `calculate_psnr_synburst` — the official quantize-to-14-bit + `PSNR(boundary_ignore=40)` path, vendored verbatim.

---

## Architecture: `MambaFusionNet`

Defined in [burstISP/archs/mambafusion_arch.py](burstISP/archs/mambafusion_arch.py). Three sequential modules:

### 1. BurstAlign (`burstISP/archs/dcn_align_arch.py`)
- Pyramid (2-level) + Cascading + Deformable alignment via **DCNv4** (custom CUDA kernel, compiled at `burstISP/utils/DCNv4/`).
- Extracts features from each LQ frame, computes DCN offsets relative to the center/reference frame, and returns aligned feature maps `[B, N, C, H, W]` plus reference features `ref_feats`.
- Runs in **float32** (forced via `autocast(enabled=False)`) for numerical stability in offset computation.
- **Note:** `ref_feats` feeds the restoration module's zero-init `skip_proj` — `MambaFusionNet.forward` calls `restoration(fused_input, ref_feats)`. It was temporarily unwired (`(fused_input, fused_input)`) as a deliberate experiment to reduce reliance on the reference frame (it did not help); the L1 revert (PLAN.md) restored the wiring.

### 2. ST_HAT Fusion (`burstISP/archs/st_hat_fusion_arch.py`)
- Input: aligned features `[B, N, C, H, W]`, output: single fused feature map `[B, C, H, W]`.
- **Stage 1** (depth_stage1 blocks, each with 3 sub-blocks):
  - `SpatialBlock`: window self-attention within each frame independently
  - `TemporalBlock`: per-pixel self-attention across the N frames (collapses burst dimension into batch dimension)
  - `SpatioTemporalBlock`: joint 3D window attention over (N × H × W) space
- **Stage 2** (dimension collapse):
  - `FusionBlock`: cross-attention where only the reference frame provides queries, all frames provide keys/values → collapses burst to single map
  - `SpatialBlock` for refinement
  - Residual from Stage 1 features via 1×1 conv projection
- **Note:** both Stage-2 residual paths use the **reference-frame slice** (`x_win[:, ref]` in FusionBlock, `x_s1[:, ref]` for the stage-2 skip), matching the class docstring. A deliberate experiment replaced them with a mean over frames to reduce reference reliance (it did not help, and with imperfect alignment a cross-frame mean acts as a low-pass filter); the L1 revert (PLAN.md) restored the reference slices.
- **Stage 3** (depth_stage3 `RefinementBlock`s):
  - Each block: OCAB (overlapping cross-attention) → HAB (hybrid window + channel attention) → OCAB
  - Removes windowing artifacts and reweights features

### 3. MambaIRv2 Restoration (`burstISP/archs/mambairv2_arch.py`)
- Adapted from the public MambaIRv2 codebase (not original to this project).
- Input: fused features `[B, C, H, W]`; second argument is a residual injected through a zero-init `skip_proj` (the aligned reference features `ref_feats` — see note above).
- Mamba-based state space model (SSM) backbone with window attention, produces upscaled output `[B, 3, 8H, 8W]`.
- Upsampler: `pixelshuffle` mode.

### Global Skip Connection (optional)
- Non-learnable Malvar-He-Cutler demosaicing (via kornia) + bicubic 4× upsampling of the center raw frame.
- Model learns residual on top of this baseline. Currently **disabled** (`global_skip: false`) in the active config.

---

## Code Structure

```
MambaFusion/
├── burstISP/              # Core library
│   ├── archs/             # Model architectures
│   │   ├── mambafusion_arch.py   ← Full model entry point
│   │   ├── st_hat_fusion_arch.py ← ST-HAT fusion module
│   │   ├── dcn_align_arch.py     ← BurstAlign with DCNv4
│   │   ├── mambairv2_arch.py     ← Restoration backbone
│   │   └── arch_util.py          ← Shared helpers (DCNv4Block, etc.)
│   ├── data/
│   │   ├── burst_image_dataset.py     ← BurstImageDataset (RealBSR-RAW)
│   │   ├── synthetic_burst_dataset.py ← SyntheticBurstDataset (L3/L4, standard protocol)
│   │   └── dbsr/                      ← Vendored DBSR toolkit code (generation, val set,
│   │                                    official PSNR) — see its README for provenance/license
│   ├── models/
│   │   ├── mambafusion_model.py  ← Training/eval wrapper
│   │   └── sr_model.py           ← Base model class
│   ├── loss/losses.py            ← CharbonnierLoss, GWLoss, SobelLoss, etc.
│   ├── metrics/psnr_ssim.py      ← calculate_psnr_srgb/linear, calculate_ssim_srgb
│   ├── metrics/synburst_psnr.py  ← calculate_psnr_synburst (official SyntheticBurst eval)
│   └── utils/
│       ├── img_util.py           ← ISP pipeline, image I/O
│       ├── options.py            ← YAML config parsing
│       └── DCNv4/                ← DCNv4 CUDA extension (must be compiled)
├── main/
│   ├── train.py                  ← Training entry point
│   ├── test.py                   ← Test/inference entry point
│   ├── config.yml                ← Current reference config
│   ├── mamba_job.sh              ← HPC job submission script; runs analysis/run_analysis.py after training
│   └── _archive/Testing_Files/   ← Stale prototype scripts, superseded by train.py/test.py
├── analysis/                     ← Analysis and visualization scripts
│   ├── visualize_inference.py    ← Run model + ISP + save PNG
│   ├── visualize_progress.py     ← Training progress visualization (all checkpoints, fixed burst set)
│   ├── visualize_dataset.py      ← Dataset inspection
│   ├── analyze_logfile.py        ← Parse training logs — dynamically discovers all losses/metrics
│   ├── run_analysis.py           ← Orchestrator: log analysis + progress viz, triggered at end of training
│   ├── burst_ablation.py         ← Normal vs. all-ref two-pass eval (burst utilization)
│   │                               + frame-drop curves (L5), realbsr/synburst datasets
│   ├── offset_analysis.py        ← DCN offset magnitude across checkpoints
│   ├── fusion_attention_mass.py  ← FusionBlock non-ref attention mass across checkpoints (L5)
│   ├── exposure_drift.py         ← Mean linear output intensity vs GT across checkpoints (L5)
│   ├── synburst_sanity.py        ← SyntheticBurst port smoke tests (CPU) + full-model check (GPU)
│   ├── gate_a_motion.py          ← Phase-correlation inter-frame motion measurement
│   ├── outputs/                  ← Gitignored generated artifacts; per-experiment results under outputs/<name>/
│   └── _archive/                 ← Retired per-run inference dumps (see analysis/README.md)
├── experiments/                  ← Saved runs (configs, checkpoints, logs)
│   ├── STHAT_GW/                 ← Completed; best PSNR-sRGB ~24.09 dB
│   ├── MF_STHAT_P0.x/            ← Completed (see below)
│   ├── MF_STHAT_P1_RefRevert/    ← L1 revert run (PLAN.md), 35k schedule
│   ├── MF_STHAT_L3_SynBase/      ← L3 SyntheticBurst baseline (PLAN.md), 100k
│   ├── MF_STHAT_L4_Oracle{On,Off}/ ← L4 oracle-alignment pair (PLAN.md), 35k each
│   └── _archive/                 ← Superseded runs (see experiments/README.md)
├── dataset/
│   ├── Inference_Set/            ← 10 local test bursts for inference
│   ├── RealBSR_RAW_testpatch/    ← Local mirror of the cluster val/test split
│   ├── RealBSR_RAW_trainpatch/   ← Small local sample of the train split (pipeline testing only)
│   └── _archive/                 ← Retired scratch data from early development
├── papers/                       ← Reference papers (RealBSR-RAW, MambaIR, HAT, etc.)
└── PLAN.md                       ← Plan of record (July 2026)
```

---

## Registry System

All models, datasets, archs, and losses are registered via decorators (e.g. `@ARCH_REGISTRY.register()`). They are selected in YAML configs by their `type` key and instantiated via `build_model()`, `build_dataset()`, etc.

---

## Training Details

- **Optimizer**: AdamW, lr=1e-4, betas=(0.9, 0.99)
- **Scheduler**: MultiStepLR (lr drops at milestones)
- **Loss**: Charbonnier (pixel) + edge term (GWLoss 0.25 in STHAT_GW; SobelLoss 0.5 in MF_STHAT_P0.x), both computed on mu-law companded (μ=24) linear RGB in `MambaFusionModel.optimize_parameters`. Companding is config-gated via `train.compand` (default `true`, so RealBSR configs reproduce exactly; the SyntheticBurst L3/L4 configs set it to `false` for plain losses on linear RGB)
- **Gradient accumulation**: `accumulation_steps=2` → effective batch size = 2 × batch_per_gpu × n_gpus
- **Mixed precision**: bfloat16 autocast in training; BurstAlign forced to float32
- **EMA**: supported via `ema_decay` config key (not enabled in recent runs)
- **Logging**: file logger + optional TensorBoard; Weights & Biases supported

---

## Completed Experiments

### STHAT_GW (completed June 2026)
- 100k iterations, 4 GPUs, Charbonnier + GWLoss (0.25)
- **Best PSNR-sRGB**: ~24.09 dB @ ~50k iter; plateaued ~24.03 dB
- PSNR-Linear declined after ~10k iter (33.0 → 31.8)

### MF_STHAT_P0.x (completed 2026-06-23)
- Follow-up to STHAT_GW, same architecture; edge loss switched to SobelLoss (0.5).
- 100k iterations (the config was standardized as `main/config.yml`; an extended 300k schedule was considered but **not** run).
- **Best PSNR-sRGB**: 24.151 dB @ 40k; final 24.023. Best PSNR-linear 33.017 @ 10k, declining to 31.74. Best SSIM 0.7322 @ 30k.
- Training time: 2 days 2 h on 4× A10.

---

## Diagnostics Summary (June 2026 — basis for current plan)

- **Burst ablation** (`analysis/burst_ablation.py`, P0.x @ 50k, 2377 test bursts): normal vs. all-reference-frames delta = +0.079 dB sRGB / −0.008 dB linear / +0.003 SSIM → **the model behaves as single-image SR**.
- **Offset analysis** (`analysis/offset_analysis.py`): lv1 DCN offsets shrink 0.45 → 0.17 px over training; lv2 ~0.14 px; cascade ~0.57 px.
- **Gate-A motion** (`analysis/gate_a_motion.py`): real inter-frame motion median **0.895 packed px** (≈1.8 raw px, ≈7 GT px), with a tail past 8 packed px — alignment compensates only a small fraction of real motion.
- Outputs are blurry; hypothesized mechanism and repair ladder in [PLAN.md](PLAN.md).

---

## ISP Pipeline (Post-processing for Visualization)

The model outputs linear RAW-domain RGB. For visualization and sRGB PSNR:
1. Auto-exposure normalization (scale by `0.2 / mean`)
2. Clamp to `[1e-6, 1.0]`
3. Gamma correction (`^(1/2.2)`)
4. Smoothstep tone mapping (`3x² - 2x³`)

Implemented in `burstISP/utils/img_util.py: generate_processed_image_channel3`.
