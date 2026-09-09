# MambaFusion — Plan of Record

*Agreed 2026-07-24 (grilling session). Supersedes the ad-hoc notes in README.md's "Planned Changes" section.*

## Goal

Workshop / NTIRE-tier paper. **Story (Option C, hybrid) — premise needs restating:** the diagnostics below are still the instrument, but ST-HAT is no longer an example of the failure they were built to expose (see Evidence base). Lead with the diagnosis — burst SR models can silently collapse into single-image SR while posting respectable PSNR — and propose the diagnostics that expose it (all-ref ablation delta as a "burst utilization" score, offset-magnitude vs. measured inter-frame motion, frame-drop curves). Land it with a repair arc: diagnostics catch the collapse in ST-HAT, localize it (L4), and a targeted fix converts a burst-ignoring model into a burst-using one, posting *competitive* (not necessarily SOTA) numbers on the standard benchmarks. The methodology is the claim; ST-HAT is the case study; the architecture contribution is ST-HAT evolved under diagnosis.

**Novelty constraint (verified 2026-07-24):** "Mamba for burst SR" is already published — QMambaBSR (arXiv 2408.08665, Huawei Noah's Ark/USTC), Burst Image Super-Resolution with Mamba (arXiv 2503.19634, Huawei), and a multi-scan SSM decoder paper (arXiv 2505.19668, Beihang). Mamba is not the contribution. Contribution locus = diagnostics + alignment/fusion (the original code in this repo).

## Constraints

- ~1 full run (100k iters, ~2 days) per week on 4×A10; liberal short runs (35k, ~17 h) — metric ordering historically settles by ~35k.
- No hard deadline; diagnosis phase first, exit when L4 renders its verdict.

## Evidence base (June 2026 runs)

- STHAT_GW and MF_STHAT_P0.x both plateau at ~24.0–24.15 dB PSNR-sRGB / ~31.7 linear / 0.727 SSIM. PSNR-linear peaks at 10k and *declines* thereafter.
- Burst ablation (P0.x @ 50k, RealBSR): **superseded — do not cite.** That reading was taken on the pre-L5 packed-RGGB architecture and no longer describes this model; the current all-ref delta on SyntheticBurst is multiple dB. The instrument (`burst_ablation.py`) stands; the June number does not.
- DCN offsets: lv1 collapses 0.45 → 0.17 px over training; Gate-A real inter-frame motion median **0.895 packed px** (≈7 GT px at ×8), tail past 8 packed px. Alignment leaves ~0.7 px median error uncompensated in the model's own coordinate system.
- Two deliberate interventions to reduce reference reliance (mean-over-frames residuals in ST_HAT; unwiring `ref_feats`) did not help and were reverted in L1. The conclusion drawn from them at the time — that non-reference frames carry no accessible signal at the fusion stage — is **superseded**: L5/L6 fusion does extract multi-dB burst gain.

## Two parallel tracks

- **Track 1 (GPU, now):** L1 revert run on the existing RealBSR pipeline.
- **Track 2 (CPU, now):** SyntheticBurst port. All serious architecture conclusions get drawn there once it lands.

## Experiment ladder

**L1 — RealBSR, 35k.** Revert both intentional interventions: `FusionBlock` residual `x_win.mean(dim=1)` → `x_win[:, ref]`; stage-2 skip `x_s1.mean(dim=1)` → `x_s1[:, ref]`; wire `self.restoration(fused_input, ref_feats)` (zero-init `skip_proj` makes it safe). Bundled deliberately for budget; **if PSNR drops, split into two runs.** Prediction: sharper output, modest PSNR gain, ablation delta still ≈ 0.

**L2 — skipped** (held-out-frame alignment loss on RealBSR; superseded by GT-flow supervision on SyntheticBurst).

**L3 — SyntheticBurst baseline, full run.** Wholesale standard protocol:
- Official synthetic burst generation code + official pre-generated 300-burst val set + their eval function verbatim (check licenses when vendoring).
- 14-frame bursts, 48×48 packed RGGB crops → 384×384 linear RGB GT (matches the repo's scale-8 packed convention).
- **Loss = plain L1 on linear RGB.** No mu-law companding, no edge term — those become labeled ablations later. L3 exists to calibrate the architecture against published numbers; every nonstandard knob goes to its literature setting.
- Fusion untouched (measure the architecture we have). Fall back to trimming stage-1 depth 3→2 only if A10 memory forces it, and log the change.

**L4 — SyntheticBurst, 35k × 2. The verdict experiment.** Oracle pair: (i) inputs warped to the reference with the generator's known flows, (ii) learned alignment as-is. Also run the all-ref ablation on the oracle model.

**L5 — no GPU cost.** Checkpoint forensics: FusionBlock attention mass on non-ref frames; frame-drop curves (N = 1, 2, 5, 9, 14); exposure-drift check for the PSNR-linear decline (hypothesis: companded loss induces brightness drift that the auto-exposure sRGB ISP masks).

## Pre-registered L4 decision rules

- Oracle − learned ≥ **0.5 dB** → alignment is materially guilty.
- All-ref delta on the *oracle* model ≥ **0.3 dB** → fusion can exploit aligned frames (downstream architecture exonerated).
- All-ref delta < **0.15 dB** even under oracle alignment → fusion is guilty → redesign.
- Likely outcome is both guilty in sequence: repair **alignment first** (upstream), re-run the ablation before touching fusion at all.

**Repair preferences:**
- Alignment: BSRT-style flow-guided DCN offsets, plus GT-flow supervision on SyntheticBurst as an auxiliary loss with the decay schedule from the old README spec (0.5 → 0.1 → 0). Note: explicit alignment losses are not standard in the literature; comparability constrains eval only, so no asterisk.
- Fusion: evolve ST-HAT (per-frame contribution paths, no mean shortcuts) rather than replace it — novel design only if the diagnosis specifically motivates one.

## Publication suite

SyntheticBurst + **BurstSR** (real, AlignedPSNR protocol — the field's institutionalized handling of GT misregistration). RealBSR-RAW as optional third (pipeline already exists). BurstSR work starts only after the L4 verdict.

## To verify before the target table

- Exact published SyntheticBurst/BurstSR numbers for DBSR, MFIR, BIPNet, Burstormer, BSRT, QMambaBSR, BISR-Mamba.
- Licenses on the burst-generation toolkit and val set.

## Housekeeping (done this session)

- CONTEXT.md corrected: MF_STHAT_P0.x *completed* 100k iters on 2026-06-23 (best 24.151 dB @ 40k) — the 300k-schedule claim was wrong; `ref_feats` is currently unused (the ControlNet-style injection description was aspirational); the mean residuals were an intentional experiment, reverted in L1.
