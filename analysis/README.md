# analysis/

Scripts for inspecting the dataset, visualizing model behavior, and analyzing training runs. All generated artifacts (plots, CSVs, comparison images) are gitignored — they belong in `outputs/`, not committed to the repo.

## Automated post-training analysis

`main/mamba_job.sh` runs `run_analysis.py` automatically right after `torchrun` finishes, so nothing needs to be run by hand once a training job completes on the cluster. It:

1. Reads `name:` from the config to resolve `experiments/<name>/`.
2. Finds that run's newest `train_<name>_*.log`.
3. Runs `analyze_logfile.py` on it → `analysis/outputs/<name>/logfile_dashboard.png` + `logfile_summary.md`.
4. Runs `visualize_progress.py` across every checkpoint in `experiments/<name>/models/` → `analysis/outputs/<name>/progress/`.

Each stage is isolated (a GPU-less environment just skips the progress step) and the command always exits 0, so it never marks the training job itself as failed. Run it manually with:

```
python analysis/run_analysis.py --config main/config.yml [--skip-progress]
```

## Scripts

| Script | Purpose |
|---|---|
| `run_analysis.py` | Orchestrator for the automated pipeline above — the single entry point called at the end of training. |
| `visualize_dataset.py` | Sanity-check raw burst/GT pairs from a dataset root. |
| `visualize_inference.py` | Run one checkpoint through the model + ISP pipeline and save a comparison PNG. |
| `visualize_progress.py` | Run every checkpoint in a run's `models/` dir against the same bursts, to see training progress over iterations. Invoked automatically by `run_analysis.py`; can also be run standalone. |
| `test_transform.py` | Check the RGGB-aware transpose augmentation. |
| `analyze_logfile.py` | Parse a training log into a dashboard + markdown summary. Dynamically discovers every loss term (`l_*`) and every validation metric logged, however many are present — nothing is hardcoded to a fixed set of losses/metrics. Reports mean±std per loss for the blocks between validation checkpoints, plus whole-run stats and a best-per-metric table. Usage: `python analyze_logfile.py --log <path> [--output-dir <dir>] [--config <yaml>]`. |
| `offset_analysis.py` / `burst_ablation.py` | Distributed ablation tools; log to `analysis/outputs/ablation_logs/` by default. `burst_ablation.py` has two modes: `two_pass` (normal vs. all-ref, the burst-utilization score) and `frame_drop` (PLAN.md L5 curves over N = 1, 2, 5, 9, 14 distinct frames, remaining slots ref-filled), on either `--dataset realbsr` or `--dataset synburst` (official val set). |
| `fusion_attention_mass.py` | PLAN.md L5: per-checkpoint FusionBlock attention mass on non-reference frames via a forward hook on the fusion softmax; mirrors `offset_analysis.py` (torchrun, log + plot to `analysis/outputs/ablation_logs/`). |
| `exposure_drift.py` | PLAN.md L5: per-checkpoint mean intensity of raw linear outputs vs. GT across the val set (drift ratio), to test whether the PSNR-linear decline is global brightness drift masked by the auto-exposure ISP. |
| `synburst_sanity.py` | SyntheticBurst port checks: `--mode cpu` smoke-tests the vendored generation, dataset contract, oracle warp and official eval without GPU or data; `--mode model` is the one-command full-model forward check for the cluster. |
| `gate_a_motion.py` | Phase-correlation inter-frame motion measurement across the dataset; outputs to `analysis/outputs/gate_a/` by default. |

## outputs/

Gitignored home for anything a script generates. Automated per-experiment results (from `run_analysis.py`) live under `outputs/<experiment name>/` (dashboard, summary, progress images). Non-run-specific artifacts (ablation logs, dataset sanity-check images, gate-A motion stats) live in their own subfolders as before. Per-run visualizations can also be regenerated directly into `experiments/<run>/visualization/` if you run `visualize_progress.py`/`visualize_inference.py` standalone with that `--output_path`.

## _archive/

Retired per-run inference comparison dumps (`inferences/`) from before per-run visualizations were consolidated into `experiments/<run>/visualization/`. Kept for reference, not actively maintained.
