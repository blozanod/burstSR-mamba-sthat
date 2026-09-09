# experiments/

Each subfolder here is one training run, created automatically by `train.py` on first launch: checkpoints (`models/` or loose `net_g_*.pth`, gitignored), logs (gitignored), `training_states/` (gitignored), a timestamped snapshot of the config used to launch it, and a `visualization/` folder holding that run's own inference/progress snapshots (gitignored — regenerate with the `analysis/` scripts, don't commit them). The configs actually used to *launch* runs live in [`main/configs/`](../main/configs/) — see "Adding a new run" below.

## Current runs

- **STHAT_Fixed35k** — first stable run of the ST-HAT + MambaIRv2 architecture, 80k iterations.
- **STHAT_GW** — same architecture, Charbonnier + GWLoss (0.25), 100k schedule. Best PSNR-sRGB ~24.09 dB @ ~35k iter. Completed.
- **MF_STHAT_P0.x** — follow-up to STHAT_GW, uses `main/config.yml` (100k iterations) with a SobelLoss edge term instead of GWLoss. Completed 2026-06-23; best PSNR-sRGB 24.151 dB @ 40k (see PLAN.md).
- **MF_STHAT_P1_RefRevert** — L1 revert run (PLAN.md): restores the reference-slice residuals in ST-HAT (`FusionBlock` residual and stage-2 skip) and wires BurstAlign's `ref_feats` into the restoration module's zero-init skip projection. Config (`main/configs/MF_STHAT_P1_RefRevert.yml`) is a copy of `main/config.yml` with `total_iter: 35000`; architecture/loss otherwise identical to MF_STHAT_P0.x. Not yet run — `experiments/MF_STHAT_P1_RefRevert/` will appear once it is.
- **MF_STHAT_L3_SynBase** — L3 SyntheticBurst baseline (PLAN.md), full 100k run on the standard protocol: on-the-fly official synthetic burst generation from Zurich RAW-to-RGB (14 frames, 48-px packed crops), official 300-burst val set, official `psnr_synburst` eval, plain L1 on linear RGB (`compand: false`, no edge term), fusion untouched. Config: `main/configs/MF_STHAT_L3_SynBase.yml`. Not yet run.
- **MF_STHAT_L4_OracleOn / MF_STHAT_L4_OracleOff** — L4 verdict pair (PLAN.md), 35k each: identical to L3 except `oracle_align` on/off — the On arm trains on bursts warped to the reference with the generator's ground-truth flows. The official val set ships no flows, so val inputs are never oracle-warped; read the verdict from the pair's relative numbers plus the all-ref ablation on the oracle model. Configs: `main/configs/MF_STHAT_L4_OracleOn.yml` / `main/configs/MF_STHAT_L4_OracleOff.yml`. Not yet run.

## _archive/

Superseded experiments, kept for reference rather than deleted:

- **v1_arch_milestones/** — the original numbered milestone sequence (`01_InitTest` … `09_FirstSkipAblation`) from before the ST-HAT + MambaIRv2 architecture existed. Internal structure (including informally-named sub-runs like `*_Garbage`, `*Desperation*`, `*_archived_<timestamp>`) is preserved as-is.
- **newarch_early/** — early/aborted iterations of the current architecture (`Fixed_STHAT`, `Fixed_STHAT_35k`, `NewArch`, `NewArch_archived_*`, `NewArch-FixedData`, `NewArch-FixedData_archived_*`) superseded by `STHAT_Fixed35k` / `STHAT_GW` / `MF_STHAT_P0.x` above.

## Adding a new run

Write the run's config directly to [`main/configs/<name>.yml`](../main/configs/) (set `name:` to match) and launch with:

```bash
qsub main/mamba_job.sh main/configs/<name>.yml
```

`train.py` creates `experiments/<name>/` itself — `models/`, `training_states/`, `visualization/`, and a timestamped snapshot of the launch config all appear automatically once the job starts; nothing needs to be pre-created by hand.

**Never put a config inside `experiments/<name>/` before launching, and never point `-opt` at a file that lives there.** `train.py`'s `make_exp_dirs()` unconditionally renames the *entire* `experiments/<name>/` folder to `..._archived_<timestamp>` at the start of every run, before copying the `-opt` config into the freshly recreated folder. A config living inside that same folder gets swept away before it can be copied in, and the job dies immediately with `FileNotFoundError` on `config.yml`. Keeping every launch config under `main/configs/` — never under `experiments/`— avoids this by construction.

When a run is superseded, move its `experiments/<name>/` folder into `experiments/_archive/` rather than deleting it; its own config snapshot travels with it, so there's nothing extra to archive from `main/configs/`.

## Automated post-training analysis

At the end of every HPC training job, `main/mamba_job.sh` calls `analysis/run_analysis.py --config <config>`, which resolves the run from the config's `name:` field, finds its newest log, and writes a loss/metric dashboard + markdown summary + checkpoint progress visualization to `analysis/outputs/<name>/` — no manual script-running required. See [analysis/README.md](../analysis/README.md).
