# experiments/

Each subfolder here is one training run: its config, checkpoints (`models/` or loose `net_g_*.pth`, gitignored), logs (gitignored), `training_states/` (gitignored), and a `visualization/` folder holding that run's own inference/progress snapshots (gitignored — regenerate with the `analysis/` scripts, don't commit them).

## Current runs

- **STHAT_Fixed35k** — first stable run of the ST-HAT + MambaIRv2 architecture, 80k iterations.
- **STHAT_GW** — same architecture, Charbonnier + GWLoss (0.25), 100k schedule. Best PSNR-sRGB ~24.09 dB @ ~35k iter. Completed.
- **MF_STHAT_P0.x** — follow-up to STHAT_GW, uses `main/config.yml` (100k iterations) with a SobelLoss edge term instead of GWLoss. Completed 2026-06-23; best PSNR-sRGB 24.151 dB @ 40k (see PLAN.md).
- **MF_STHAT_P1_RefRevert** — L1 revert run (PLAN.md): restores the reference-slice residuals in ST-HAT (`FusionBlock` residual and stage-2 skip) and wires BurstAlign's `ref_feats` into the restoration module's zero-init skip projection. Config is a copy of `main/config.yml` with `total_iter: 35000`; architecture/loss otherwise identical to MF_STHAT_P0.x. Not yet run.
- **MF_STHAT_L3_SynBase** — L3 SyntheticBurst baseline (PLAN.md), full 100k run on the standard protocol: on-the-fly official synthetic burst generation from Zurich RAW-to-RGB (14 frames, 48-px packed crops), official 300-burst val set, official `psnr_synburst` eval, plain L1 on linear RGB (`compand: false`, no edge term), fusion untouched. Not yet run.
- **MF_STHAT_L4_OracleOn / MF_STHAT_L4_OracleOff** — L4 verdict pair (PLAN.md), 35k each: identical to L3 except `oracle_align` on/off — the On arm trains on bursts warped to the reference with the generator's ground-truth flows. The official val set ships no flows, so val inputs are never oracle-warped; read the verdict from the pair's relative numbers plus the all-ref ablation on the oracle model. Not yet run.

## _archive/

Superseded experiments, kept for reference rather than deleted:

- **v1_arch_milestones/** — the original numbered milestone sequence (`01_InitTest` … `09_FirstSkipAblation`) from before the ST-HAT + MambaIRv2 architecture existed. Internal structure (including informally-named sub-runs like `*_Garbage`, `*Desperation*`, `*_archived_<timestamp>`) is preserved as-is.
- **newarch_early/** — early/aborted iterations of the current architecture (`Fixed_STHAT`, `Fixed_STHAT_35k`, `NewArch`, `NewArch_archived_*`, `NewArch-FixedData`, `NewArch-FixedData_archived_*`) superseded by `STHAT_Fixed35k` / `STHAT_GW` / `MF_STHAT_P0.x` above.

## Adding a new run

Create `experiments/<name>/` directly under `experiments/` (no numbering needed) with its own copy of `main/config.yml` (set `name:` to match) and a `visualization/` subfolder for that run's own inference/progress outputs. When a run is superseded, move it into `experiments/_archive/` rather than deleting it.

## Automated post-training analysis

At the end of every HPC training job, `main/mamba_job.sh` calls `analysis/run_analysis.py --config <config>`, which resolves the run from the config's `name:` field, finds its newest log, and writes a loss/metric dashboard + markdown summary + checkpoint progress visualization to `analysis/outputs/<name>/` — no manual script-running required. See [analysis/README.md](../analysis/README.md).
