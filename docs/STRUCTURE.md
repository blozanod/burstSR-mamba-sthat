# Repository structure

*Generated from the filesystem on 2026-09-09. Vendored third-party trees
(`burstISP/utils/DCNv4/`, `burstISP/data/dbsr/`) are summarised, not expanded.*

```
burstSR-mamba-sthat/
├── burstISP/                         # core library
│   ├── archs/
│   │   ├── mambafusion_arch.py       # MambaFusionNet — full model entry point
│   │   ├── dcn_align_arch.py         # BurstAlign — flow pyramid + DCNv4
│   │   ├── st_hat_fusion_arch.py     # ST-HAT fusion module
│   │   ├── mambairv2_arch.py         # MambaIRv2 restoration trunk (adapted)
│   │   ├── temporal_fusion_arch.py
│   │   ├── vgg_arch.py
│   │   └── arch_util.py              # shared helpers (DCNv4Block, to_2tuple, ...)
│   ├── data/
│   │   ├── burst_image_dataset.py    # BurstImageDataset (RealBSR-RAW)
│   │   ├── synthetic_burst_dataset.py
│   │   ├── pre_alignment.py
│   │   ├── transforms.py, data_util.py, data_sampler.py
│   │   └── dbsr/                     # VENDORED — DBSR toolkit, CC BY-NC-SA 4.0
│   │       └── README.md             # provenance + license (do not move)
│   ├── models/
│   │   ├── mambafusion_model.py      # training/eval wrapper
│   │   ├── mambairv2_model.py
│   │   ├── sr_model.py, base_model.py, lr_scheduler.py
│   ├── loss/losses.py                # Charbonnier, GWLoss, SobelLoss, ...
│   ├── metrics/
│   │   ├── psnr_ssim.py              # calculate_psnr_linear / _srgb, ssim
│   │   └── synburst_psnr.py          # official SyntheticBurst eval (vendored)
│   └── utils/
│       ├── img_util.py               # ISP pipeline, image I/O
│       ├── options.py                # YAML config parsing
│       └── DCNv4/                    # VENDORED — DCNv4 CUDA extension
│                                     # (must be compiled; upstream READMEs kept)
│
├── main/
│   ├── train.py                      # training entry point
│   ├── test.py                       # test / inference entry point
│   ├── config.yml                    # legacy reference config (RealBSR era)
│   ├── configs/                      # launch configs — ALL configs live here
│   │   ├── MF_STHAT_L3_SynBase.yml
│   │   ├── MF_STHAT_L4_OracleOn.yml
│   │   ├── MF_STHAT_L4_OracleOff.yml
│   │   ├── MF_STHAT_L5_BayerSpace.yml
│   │   ├── MF_STHAT_L5_PackedControl.yml
│   │   ├── MF_STHAT_L6_FlowFusion.yml
│   │   └── MF_STHAT_P1_RefRevert.yml
│   ├── jobs/                         # HPC submission scripts
│   └── _archive/Testing_Files/       # stale prototypes, superseded by train.py
│
├── analysis/
│   ├── run_analysis.py               # orchestrator, called at end of training
│   ├── analyze_logfile.py            # log parsing / dashboards
│   ├── param_budget.py               # analytic parameter counter (no torch)
│   ├── fusion_cost_model.py          # analytic FLOP + activation-memory model
│   ├── burst_ablation.py             # two_pass (all-ref) + frame_drop curves
│   ├── offset_analysis.py            # DCN offset magnitude across checkpoints
│   ├── fusion_attention_mass.py      # FusionBlock non-ref attention mass
│   ├── gate_a_motion.py              # phase-correlation inter-frame motion
│   ├── exposure_drift.py             # mean output intensity vs GT
│   ├── dcn_probe.py, dcn_scatter_check.py
│   ├── synburst_sanity.py, synburst_interp_baseline.py
│   ├── shape_check.py, test_transform.py, burst_data.py
│   ├── visualize_inference.py, visualize_progress.py, visualize_dataset.py
│   └── outputs/                      # gitignored generated artifacts
│
├── experiments/                      # one folder per run, created by train.py
│   ├── MF_STHAT_L3_SynBase/
│   ├── MF_STHAT_L5_BayerSpace/
│   └── MF_STHAT_L5_BayerSpace_100k/
│
├── dataset/                          # local preview data (full sets on cluster)
├── docs/
│   ├── STRUCTURE.md                  # this file
│   ├── HISTORY.md                    # narrative record — NOT ground truth
│   └── _prune/                       # old docs, staged for review/deletion
│
├── requirements.txt
└── LICENSE
```

## Conventions worth knowing

- **Registry.** Archs, datasets, models and losses register via decorators
  (`@ARCH_REGISTRY.register()`) and are selected by the `type:` key in YAML.
- **Configs live in `main/configs/`, never in `experiments/`.** `train.py`'s
  `make_exp_dirs()` renames the whole `experiments/<name>/` folder to
  `..._archived_<timestamp>` at the start of every run, before copying the
  `-opt` config in. A config living inside that folder is swept away first and
  the job dies on `FileNotFoundError`.
- **Scale convention.** LQ is packed RGGB; GT is 8× the packed resolution,
  hence `scale: 8` everywhere. With `pre_align: true`, PreAlign pixel-shuffles
  ×2 to the Bayer grid and the restoration module upsamples by `scale // 2`.
- **Two vendored trees have their own licenses.** `burstISP/data/dbsr/` is
  CC BY-NC-SA 4.0 (academic use only) — its README carries the required
  attribution and must stay next to the code. `burstISP/utils/DCNv4/` keeps its
  upstream READMEs for the same reason.
