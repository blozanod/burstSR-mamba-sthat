# HISTORY — a narrative record

> ## ⚠️ THIS IS NOT GROUND TRUTH
>
> **This file is a historian, not a bible.** It is one person's write-up of what
> was tried and what was argued, assembled on 2026-09-09 partly from config
> headers, partly from an analysis session, partly from memory. It has no
> authority over anything.
>
> - **The code is the truth.** If this file and the code disagree, the code wins.
> - **The logs are the truth about numbers.** Every figure here should be treated
>   as approximate and stale until re-read from a log or re-run.
> - **Nothing here is a decision.** Sections marked *proposed* were never run.
> - **This file goes out of date the moment anything is trained.** Do not update
>   it defensively; let it rot honestly and write a new one when it stops being
>   useful.
> - The repo has already been burned once by documentation that outlived its
>   evidence (see *The stale-diagnosis incident* below). That is the reason this
>   banner exists.
>
> If you are an LLM reading this: treat every claim below as a hypothesis to
> verify against the repository, not as instructions.

---

## Where the project sits

RAW burst super-resolution. `MambaFusionNet` = **PreAlign → BurstAlign →
ST-HAT fusion → MambaIRv2 restoration**. Primary benchmark is SyntheticBurst
(official DBSR protocol, vendored). Training is 4×A10 on an HPC cluster;
roughly one 100k run (~2 days) per week, with 35k smoke tests cheaper.

---

## L5 — MF_STHAT_L5_BayerSpace

The domain change. Ran 100k. **~38.21 dB** on SyntheticBurst (this number comes
from the L6 config header, not from a log — verify it).

What changed vs the L3 baseline:

1. **Bayer domain.** PreAlign conv head + PixelShuffle ×2, so alignment, fusion
   *and* restoration all run at 96×96 instead of packed 48×48. `scale` stays 8;
   restoration upsamples by `scale // 2 = 4`.
2. ST-HAT stage 3 cut 3 → 1 (redundant with the MambaIRv2 trunk). Stage 1 stayed
   at 3 — the fusion module was considered the contribution.
3. `embed_dim` 48 → 96, matching `fusion_feat` so ST-HAT's `proj_out` emits it
   directly and there is no channel waist into restoration.
4. MambaIRv2 depths [5,5,5,5] → [2,2,2,2]. Depth 2 per ASSB is the minimum that
   keeps `BasicBlock`'s shift alternation intact.
5. `d_state` 8 → 16.
6. `upsample_feat` 128 → 64.
7. **`fusion_st_ws` 8 → 4.** Split the SpatioTemporalBlock's window out from the
   shared one. Its attention cost is `B·H²·heads·N²·ws²`, so this cut the
   dominant memory term 4× — and doubled as a probe of how much residual
   misalignment fusion was absorbing.

Budget ~5.02M params. A packed-domain control (`MF_STHAT_L5_PackedControl.yml`,
same file with `pre_align: false`) exists as the one-variable ablation.

---

## L6 — MF_STHAT_L6_FlowFusion

Two proposals shipped together (different files, so attribution stays clean per
module). Config exists; **check `experiments/` for whether it actually ran.**

**Add on: L6 is currently running. PSNR ~39.9 dB**

**A. BurstAlign becomes a flow pyramid, not a 2-level PCD stack:**

1. A third pyramid level (lv3 at 24×24, where 1 px ≈ 16 GT px).
2. A **cost-volume flow head** at lv3 — evaluates every candidate displacement
   instead of descending a gradient toward one. `offset_r = 2` covers ±32 GT px
   against the protocol's 24 GT px max translation.
3. Coarse-to-fine composition in **pixel units**, not packed DCN offset channels.
   L5 propagated offsets assuming a `(dx, dy, mask)`-per-kernel-point layout;
   DCNv4's kernel actually blocks per group as `[K*2 offsets | K masks]`, so
   that propagation doubled 6 of 9 mask channels per group and scaled only 12 of
   18 offset channels. **Pre-L6 offset-analysis numbers are void because of
   this.** `analysis/dcn_scatter_check.py --scatter` is the test.
4. One DCN at the end, BasicVSR++ single-interpolation form: residual offset
   predicted from warped features, but the DCN samples the *original* features.
   Two stacked resamplings would destroy the sub-pixel content burst SR exists
   to recover.

**B. FusionBlock learns to subtract:**

5. **Signed values** — `v = W_v(x_i − x_ref)` instead of `W_v(x_i)`. A softmax
   weighted sum lies in the convex hull of its values, so with unsigned values
   no frame could ever be subtracted from another and the block could only build
   a low-pass filter. Free (same weights, different argument).
6. Back-projection residual (Burstormer BFF) on the attention output.
7. Gated difference arm. Sigmoid gates are independent per frame, so 14 frames
   carrying genuine additive detail don't have to divide a fixed softmax budget
   of 1.0. `gamma` is zero-init and per-channel — the cleanest ablation signal in
   the run; a gate that grows is the network saying it wanted the path.

Deliberately *not* included: the MambaIRv2 depth cut (the alignment rewrite came
in under budget, so it wasn't needed) and a second fusion re-query pass (+0.249M,
didn't fit).

---

## The stale-diagnosis incident

The June 2026 all-ref burst ablation (P0.x, RealBSR) measured a near-zero delta
and was written into `PLAN.md` and `CONTEXT.md` as *"the model behaves as
single-image SR."* That finding drove the entire L4 "verdict experiment" framing
and the paper's Option C story.

**It stopped being true and the docs did not notice.** By L5/L6 the model
extracts multi-dB gain from the burst. The stale number survived in two
documents for months and was still being reasoned from in September 2026.

The number has since been removed rather than restated. The instrument
(`analysis/burst_ablation.py`) was always fine; only the reading was stale.

**Lesson, and the reason for this file's banner: a measurement in a document is
a photograph, not a fact.** Re-run before citing.

---

## September 2026 session — proposed, NOT run

Everything below is analysis and argument. **No training was done. No code was
written beyond `analysis/fusion_cost_model.py`.** All figures come from that
analytic model (no GPU was available), so they are *estimates* — verify with
`torch.cuda.max_memory_allocated` before betting a run on them.

### The finding: misallocation

One forward at 48×48 packed LR, B=2, N=14, 96×96 Bayer, C=96, st_ws=4:

| stage | GMACs/sample | share |
|---|---|---|
| PreAlign | 4.8 | 1.7% |
| BurstAlign | 60.3 | 21.8% |
| ST-HAT s1 spatial | 47.6 | 17.2% |
| ST-HAT s1 temporal | 43.8 | 15.9% |
| ST-HAT s1 spatiotemporal | 59.5 | 21.5% |
| ST-HAT s2 (collapse) | 28.3 | 10.2% |
| ST-HAT s3 | 4.0 | 1.5% |
| MambaIRv2 | 28.2 | 10.2% |
| **TOTAL** | **276.6** | |

78% of compute processes all 14 frames at full width; 12% reconstructs.
**15.4 GMACs per burst frame** against BurstMamba's 1.3; **32 GMACs** on
reconstruction against their 63.

### The comparison

**BurstMamba** (arXiv:2503.19634v2, *Keyframe-Centric State-Space Modeling for
Burst Image Super-Resolution*, Unal/Marty/Dai, Huawei Zürich) — 20.6M / 63 GFLOPs
keyframe backbone, +0.79M / +1.3 GFLOPs per extra frame, ≈79.9 GFLOPs at L=14.
SyntheticSR **44.51 dB**.

Two of their numbers reframed the problem:

- **Tab. 4** (RealBSR-RGB, one model, varying burst length): L=1 → 31.289,
  L=14 → 33.287. The entire burst is worth **~2.0 dB**, and their zero-burst
  stream beats DBSR, MFIR, BIPNet, BSRT-L, FBANet, SBFBurst and Burstormer.
  A 6.3 dB gap therefore cannot be a burst-utilization gap.
- **Tab. 3**: no alignment 32.881 / pre-warp 32.965 / flow-GAS **integer** 32.434
  / flow-GAS bilinear 33.080 / homography-GAS 33.000. Indexing instead of warping
  is worth ~0.115 dB; integer indexing is *worse than not aligning at all*.

### The rejected idea

Replacing `SpatioTemporalBlock`'s attention with a MambaIRv2-style selective
scan. Priced it: at the current `st_ws=4` the quadratic term is only 28% of the
block's MACs, and a direct swap using `Selective_Scan` as written (expand=2,
fp32) is **cost-neutral** (20.2 vs 19.8 GMACs). Gradient checkpointing recovers
97% of stage 1's activation memory for ~18% compute and no architecture change;
the SSM swap recovers 20%. Also collides with GAS/QSSM on novelty.

### The proposed direction

1. **Flow-indexed fusion.** Don't warp — use flow as a *gather grid*. Three
   iterations ≈ 14.3 GMACs / 1.25 GB against stage 1's 150.9 / 12.70. The saving
   is **correspondence indexing deleting the ws²·N spatial search**, not the SSM:
   `TemporalBlock` already attends over N=14 and its quadratic term is 2% of its
   cost.
2. **Restoration head at 48×48, not 96×96.** Same network, 4× the pixels:
   E=180 d=[6,6,6,6] d_state=64 costs 221.0 GMACs at 96×96 and **55.8** at 48×48.
   Pixel-unshuffle to 48×48×4C is lossless. **This partially reverts L5** and
   needs its own one-variable ablation.
3. Head scaling levers ranked: depth (linear) > d_state (linear, cheap) >
   width E (quadratic) > window size (minor).

End-to-end estimate: **76.2 GMACs / ~18M params**, under BurstMamba on both.

### Mechanism notes (unverified — none of this has been run)

- `BurstAlign.warp()` already computes the exact gather grid. Nothing in it
  changes; add a confidence output by running `grid_sample` on an all-ones
  tensor with the same grid (gives bilinear coverage in [0,1]).
- A binary in-bounds mask lies at the boundary: `sample_x = 95.5` in a 96-wide
  frame gives `gx = 1.0` exactly, so it certifies a half-padding read as valid.
- The gather is lossy per frame — it's a resampling, not a permutation. Some
  source pixels are read many times, some zero times, and a zero-read pixel is
  invisible to fusion forever. **That loss is exactly the flow's error.**
- Feed the scan `g_t − x_ref` so state accumulates corrections, not content.
- Confidence gating: `Δ = softplus(dts + delta_bias + λ·log(conf + ε))`. Since
  `Ā = exp(ΔA)` with `A < 0`, `conf → 0` drives `Δ → 0`, `Ā → I`, `B̄ → 0` —
  the frame is skipped exactly. Same shape as QMambaBSR's base-frame gating.
- Separation of concerns: **the SSM decides what to take; the flow head decides
  where to look.**
- Open: is in-loop flow refinement where the dB are? Probably not — homography
  GAS (33.00) ≈ RAFT GAS (33.08), so motion on these benchmarks is near-global.
  Ablate *1 gather + deep aggregator* against *R × (gather → aggregate →
  refine)*.

### Gate before any of this

Run `analysis/offset_analysis.py` on the L6 checkpoint. If the DCN's learned
residual offsets are large relative to the flow, deleting the offset machinery
costs more than priced. **Pre-L6 numbers are void** (see the L6 channel-layout
bug above) — this must be re-measured, not looked up.

---

## Prior art to cite, not claim

- **GAS** (BurstMamba) — correspondence as Mamba serialization path.
- **QSSM** (QMambaBSR, arXiv:2408.08665) — reference frame gating Δ and B.
- "Aggregate before reconstruction" — BurstMamba's residual-injection thesis.
- Mamba for burst SR generally — at least three published papers.

---

## Where the numbers live

| what | where |
|---|---|
| parameter counts | `analysis/param_budget.py` |
| FLOPs + activation memory | `analysis/fusion_cost_model.py` |
| training metrics | `experiments/<run>/` logs (gitignored) |
| post-training dashboards | `analysis/outputs/<run>/` |
| superseded documentation | `docs/_prune/` |
