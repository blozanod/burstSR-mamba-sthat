# analysis/_archive

Scripts from finished lines of work (ST-HAT / MambaFusionNet L3-L6, the DCN alignment, RealBSR
one-offs), moved here on 2026-10-03 so `analysis/` holds only what the current (KGTSMamba)
pipeline runs. Nothing here is maintained. Each still runs from its new location (repo-root
paths were adjusted by one level, nothing else), but most target `MambaFusionNet` / `BurstAlign`
/ RealBSR data and will not work on a KGTSMamba checkpoint.

Git history follows them (`git log --follow analysis/_archive/<file>`). `**/_archive/` is in
`.gitignore`, so these stay tracked but new files dropped in here are not picked up unless added
with `git add -f`.

| file | was | superseded by / why archived |
|---|---|---|
| `offset_analysis.py` (+ `jobs/offset_analysis_job.sh`) | BurstAlign DCN offset magnitude across checkpoints (L6) | KGTS has no DCN; geometry: `diagnostics/oracle_geometry.py` |
| `dcn_scatter_check.py` | BurstAlign `scatter_flow` vs DCNv4 channel layout (L6) | no DCN in KGTSMamba |
| `dcn_probe.py` | DCNv4 scratch probe | — |
| `fusion_attention_mass.py` | ST-HAT FusionBlock attention on non-ref frames | no FusionBlock; burst use: `diagnostics/error_bands.py`, `burst_length.py` |
| `burst_data.py` | LQ-burst loader shared by the two scripts above | only they used it |
| `fusion_cost_model.py` | analytic FLOP / memory model of ST-HAT stage 1 | `burstISP/archs/KGTSMamba/budget_check.py` |
| `param_budget.py` | analytic MambaFusionNet parameter counter (L6) | `budget_check.py` (exact counts) |
| `exposure_drift.py` | mean output intensity vs GT across checkpoints (L5, RealBSR) | `diagnostics/colour_fit.py` (bias by intensity, colour / tone fits) |
| `gate_a_motion.py` (+ `jobs/gate_a_motion_job.sh`) | phase-correlation motion statistics of RealBSR | one-off gate, done |
| `synburst_sanity.py` | SyntheticBurst port checks + MambaFusionNet forward (L3/L4) | `kgts_cpu_checks.py`, `kgts_sanity.py` |
| `synburst_interp_baseline.py` | bilinear / bicubic SyntheticBurst baseline via MambaFusionNet's skip | one-off number |
| `test_transform.py`, `visualize_dataset.py`, `visualize_inference.py` | RealBSR transform / dataset / MambaFusionNet inference previews | one-offs with hard-coded paths |

Still in `analysis/`: `run_analysis.py` + `analyze_logfile.py` + `visualize_progress.py` (end of every
training job), `kgts_sanity.py` + `kgts_cpu_checks.py` (KGTSMamba pre-flight / CPU checks),
`burst_ablation.py`, `count_macs.py`, `shape_check.py` (used by `main/jobs/rcan3d_job.sh` and the
older ablation jobs), and `diagnostics/`.
