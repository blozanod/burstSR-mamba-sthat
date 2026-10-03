# KGTSMamba diagnostics

Where a trained KGTSMamba checkpoint succeeds and fails on SyntheticBurst, and which architecture
change each failure points to. Four measurements, one orchestrator, one job:

```
qsub main/jobs/diagnostics_job.sh M1_KGTSMamba.yml            # latest checkpoint, EMA weights
qsub main/jobs/diagnostics_job.sh M1_KGTSMamba.yml 120000 --draws 3
python analysis/diagnostics/run_all.py --config main/configs/M1_KGTSMamba.yml --ckpt latest   # interactive
python analysis/diagnostics/<script>.py --config ... --ckpt ... --limit 20                     # one, quick
```

Outputs: `analysis/outputs/diagnostics/<run name>/<checkpoint>_ema/` — `SUMMARY.md` plus, per
script, `.txt` (the report), `.json`, `_per_image.csv` and `.png`. One A10, M1 size: roughly
40 min for all four (one `burst_length` draw). Each script's docstring is its full spec.

| script | question | data |
|---|---|---|
| `error_bands.py` | Is the error low-frequency / colour (the body) or high-frequency at edges (sub-pixel extraction and reconstruction)? What does the burst add per band? | SyntheticBurstVal, burst vs all-ref vs aux head |
| `colour_fit.py` | How many dB does a per-image 3×3 + offset colour map recover? | SyntheticBurstVal |
| `oracle_geometry.py` | Does LK's tap-position error, and its tail, cost dB? | generated bursts with flow (Zurich `test`) |
| `burst_length.py` | Does PSNR keep climbing to N = 14, and in which bands? | SyntheticBurstVal, true-length N = 1..14 |

## Shared conventions (common.py)

- **Official metric**: clamp, 14-bit truncation, 40 px border. Every error image is exactly the one
  the headline PSNR is computed from, so band energies add up to its MSE.
- **fp32** (validation runs fp32), **EMA weights** (`--weights raw` for `params`).
- **Paired routing**: MambaIRv2's ASSM samples a Gumbel route even in eval, so two forwards of a
  burst differ (~1e-4 here). The default `--routing paired` seeds that noise per call and shares it
  across the batch, so burst vs all-ref, LK vs oracle and N = 1..14 differ only in what is compared.
- **Bands** in units of the packed-LR Nyquist F_N = 1/16 cycles per HR px:
  `<0.5` tone/colour, `0.5-1` up to what one frame resolves, `1-2` demosaicking range,
  `2-4` and `>4` beyond any single frame: only sub-pixel burst information (or priors) puts anything
  there. The requested 2-band split ("Gaussian low-pass at about one LR pixel") is a Gaussian of
  σ = 3 HR px, half amplitude at F_N.
- **Attenuation vs additive** (error_bands): in each band the error splits into the part explained
  by a gain on the GT's own spectrum (MTF < 1: the output is too smooth, detail is missing) and the
  rest (residual noise, demosaicking/aliasing artifacts, misplaced detail). This matters because
  residual noise is broadband — most of it lands in the high bands too — so "high-frequency error"
  alone does not mean "missing detail". The edge / flat split says the same thing spatially.

## Reading the results → candidates

`SUMMARY.md` prints the evidence for and against each candidate with the thresholds used. The logic:

| result | means | points to |
|---|---|---|
| low-band or chroma error dominates; `colour_fit` affine gain ≥ ~0.2 dB | the body's colour / large-scale denoising is the gap | none of the five; body / loss / noise conditioning (review §6.4) |
| high-band error is **additive**, spread into flat areas | residual noise and artifacts | denoising: noise-level conditioning, body; not the burst-side candidates |
| high-band error is **attenuation**, concentrated on edges | detail is missing | sub-pixel extraction or reconstruction: 1–4 |
| …and burst MTF > all-ref MTF above 2 F_N | the burst delivers detail; rendering it at ×8 loses the rest | 1 (AdaUp), 2 (HR injection) |
| …and burst MTF ≈ all-ref MTF above 2 F_N | the burst is used as a denoiser; sub-pixel content is not extracted | 3 (wavelet scan), 2, 4 |
| aux-head mixing gains ≥ 0.05 dB in a band ≥ 1 F_N | the burst-only path carries detail the main path drops | 2 (HR injection) |
| oracle − LK ≥ 0.1 dB, tailfix recovers ≥ 60% of it | the LK tail costs dB | 5 (fix the tail) |
| oracle − LK ≥ 0.1 dB, tailfix recovers little | bulk LK precision costs dB | 5, but as better registration overall |
| oracle − LK < 0.05 dB | positions are not the bottleneck | drop 5; 4 reads correctly placed samples |
| burst curve still ≥ 0.5 dB / doubling at N 8..14 | every frame still adds information | 4 (more samples per frame), 5 |
| burst curve < 0.2 dB / doubling at N 8..14 | information arrives but is not used | 1–3, not 4 |
| frames 8→14 only reduce the < 1 F_N error | extra frames only denoise | 3, 2 |

Cost note for candidate 4, measured with `budget_check.measure` (CPU FlopCounter, body scans added
analytically; it reproduces M1's 88.0 GMACs): **M1 with `token.k: 3` is 115.9 GMACs** (+27.9) —
KGTS scans 15.1 → 29.0, token-side precompute 8.5 → 19.0, TokenRefine 2.2 → 4.8, bank 0.8 → 1.7.
That is over `mamba_job.sh`'s 90 GMAC gate: the fused kernel made k = 3 fast, not cheap. For scale,
the whole ×8 tail (conv_before_upsample + upsample + conv_last) is 7.6 GMACs, so an HR-side
injection (candidate 2) that splats projected tap samples at their sub-pixel positions can stay
well under 1 GMAC.

Caveats worth keeping in mind:

- **N < 8 is out of distribution.** M1 trains with `burst_aug.min_frames: 8`; the left half of the
  burst curve measures robustness to unseen lengths as much as information. Judge "flattening" on
  N ≥ 8; the script marks the rest.
- **The oracle is a lower bound.** The model was trained on LK positions; better geometry at test
  time is worth at least what it shows (a model trained with it could trust positions more).
- **SyntheticBurst has no colour matrix to invert**: its GT is the camera-space image the RAW was
  mosaicked from. A colour-fit gain there is bias (noise clipping in the darks, exposure, L1's median
  bias), not a failed colour inversion.
- **The aux test can only say yes.** The aux head is a linear decode of call 0, before the body; if
  mixing it in helps, the information is there, if not, nothing is proven.

## Smoke test

`python analysis/diagnostics/smoke_test.py` (CPU, ~15 min, no data or GPU): writes procedural
images, a SyntheticBurstVal-format set and a tiny random M1-shaped checkpoint, unit-checks the
measurement code (band partition exact, blur → attenuation, noise → additive, colour fit recovers a
known map, paired routing independent of batch, `oracle_flow` in LK's convention plus
`kgts_sanity`'s geometry stage), then runs `run_all.py --cpu` on three bursts.
