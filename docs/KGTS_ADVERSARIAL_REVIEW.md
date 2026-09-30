# KGTS adversarial review (2026-09-30)

> **What the numbers are.** Everything measured here ran on CPU: the repo's own modules
> (`KGTSMamba`, `KGTSAlign`, `TokenBank`, `KGTS`), bursts from the repo's DBSR generator
> (official SyntheticBurst protocol, BSDS500 images, 256 px crops → 32×32 packed),
> reduced widths, a few hundred iterations. They point in a direction; none of them is a
> benchmark. Where something is exact (a gradient that is identically zero, a permutation
> invariance, a bit-exact reproduction), it is called exact. Like `HISTORY.md`: the code
> is the truth, and every number here should be re-run before it is cited.

Every change is a flag, off by default. **With all flags off the model reproduces HEAD
(236dbba) bit-exactly** — state dict, output, aux flows and all 270 / 273 parameter
gradients on two configurations, also with `W_c`, γ and `W_q` perturbed so the scan is
live. `python analysis/kgts_cpu_checks.py` re-runs that and the other checks on any
machine (no GPU, no mamba_ssm). `main/configs/M1_KGTSMamba.yml` now switches them all on
(the queued M1 run starts with them); 236dbba's M1 is the flags-off model, so the two
differ by exactly this review.

---

## TL;DR

1. **The tap geometry is the ceiling, and it is weakest when it matters most.** KGTS reads
   each tap's sub-pixel position straight off the flow, and nothing downstream corrects it.
   A parameter-free photometric affine registration of the burst itself (`align.global_motion:
   lk`) lands at **0.079 packed px** mean error from step 0, against ~0.35 for the affine
   fit of the learned flow (and ~5 px for the learned flow at init).
2. **Your token-depth point is right, and it is broader than the convs:** the *entire value
   path* (`u = W_u x`, plus `W_δ x`, `W_B x`, `W_k x`) is computed once and is identical at all
   six injections. Deep calls can only re-weight the same shallow vectors. Two fixes:
   deeper per-frame tokens (`align.token_blocks`) and tokens re-expressed in the keyframe
   state mid-body (`refine`), which is cheap because tap tokens and keyframe state share a grid.
3. **Your head-role point is right, and decay alone cannot solve it.** At init all 16
   head×direction pairs are interchangeable integrators (40–47 of 56 taps each). A faster A
   does not make a head discriminate — in frame order it makes it a *recency* detector (up to
   55% of its weight on the last 8 taps). The fix is to give selector heads an **order in
   which recency means relevance**: scan their taps sorted by a relevance key, so a fast state
   keeps a soft top-k, a slow state still reads every tap, and the scan becomes exactly
   invariant to frame order (`kgts.roles`).
4. **The burst branch gets exactly zero gradient at step 0** (`W_c` is zero-init) and then a
   small one while `W_c` grows; an auxiliary burst-only reconstruction (`aux_head`) supervises
   it from step 0.
5. **SyntheticBurst never rewards rejecting a tap** (one affine map per frame, no occlusion,
   no local motion), so a model trained on it has no reason to learn discrimination before
   it meets real motion. Moving-object augmentation (`datasets.train.outliers`), a hybrid
   global+local flow (`align.local`), a photometric flow loss for data without flow ground
   truth (`train.photo_opt`), and burst subset/order augmentation (`train.burst_aug`) make the
   same model usable on real bursts.
6. **Bug:** `flow_lambda` milestones are read in *micro-steps* (`optimize_parameters` gets
   `train.py`'s micro-step counter), so M1's curriculum runs 2× fast and L6's 4× fast.
   `flow_lambda.unit: iter` fixes it without touching running experiments.

---

## How this was checked

- **Harness.** `mamba_ssm` has no CPU build, so a stub provides mamba's own pure-torch
  `selective_scan_ref` (the MambaIRv2 body runs, slowly), and the KGTS end-state scan uses an
  exact closed form, `y_L = Σ_n C_n Σ_t exp(A_n τ_t) dt_t B_nt u_t` with `τ_t = Σ_{s>t} dt_s`,
  verified against `kgts_arch.ref_scan` (max abs error 5e-6 at scale 42, gradients 2e-7
  relative). Training probes swap the six ASSBs for conv groups with the same call
  signature, so everything else — alignment, bank, KGTS, wiring, upsampler — is the real
  `KGTSMamba.forward`.
- **Data.** 3000 training bursts (BSDS500 train+test) and 100 held-out bursts (BSDS500
  val = BSD100) through `rgb2rawburst` with the official transformation / noise parameters,
  keyframe at N // 2, flow_vectors kept.
- **Not done:** no GPU, so no kernel parity for the new code paths, no peak memory, no GMACs
  measured on GPU (the counts in §9 are CPU FlopCounter + analytic scans, the method
  budget_check uses on GPU). `analysis/kgts_sanity.py` and `budget_check.py` are the
  pre-launch gates for that (both updated for the new options);
  `analysis/kgts_cpu_checks.py` is the CPU part, committed.

---

## Summary

| # | Flaw | Evidence | Change | Status |
|---|---|---|---|---|
| 1 | Tap positions come from a learned dense flow on noisy RAW; at x8 every 0.1 packed px of error is 0.8 HR px of misplaced sample, and at init the flow is ~5 px off | LK reaches 0.079 px vs ~0.35 (fit) | `align.global_motion: lk` (`affine_lk`) | implemented, measured |
| 2 | One global model is wrong on real bursts (moving objects, parallax) | mechanism test: patch followed at 0.075 px, background untouched | `align.local` (`local_residual`) | implemented, tested |
| 3 | Real data has no flow ground truth, so FlowAlign is unsupervised there | loss minimised at the true flow | `train.photo_opt` | implemented, tested |
| 4 | Flow curriculum runs in micro-steps | code path | `flow_lambda.unit: iter` | implemented (opt-in) |
| 5 | Shallow, static tokens: 5 convs (RF 11 px, ~77k params) for 13 frames vs a 24-layer body for 1 | code; value path identical at every call | `align.token_blocks` | implemented, probed |
| 6 | Tokens never see what the body learned | code | `refine` (`TokenRefine` + cache rebuild) | implemented, probed |
| 7 | Heads have no roles | init anatomy: all 16 are integrators | `kgts.roles` (int / geo / con) | implemented, measured, probed |
| 8 | A fast decay in frame order is recency, not selection | a_max 8: up to 55% weight on the last 8 taps | relevance-ordered selector scans | implemented, exact invariance |
| 9 | Heads share one LayerNorm | code | `kgts.out_norm: group` | implemented |
| 10 | One tied KGTS serves six different depths; its update cannot grow with the residual stream | ‖s‖ grows 2.3× across the calls at init; the tied `W_c` writes the same size everywhere | `kgts.depth_embed`, `kgts.untie_out` | implemented, identity at init |
| 11 | The first ASSB never sees the burst | code | `inject_first` | implemented, identity at init |
| 12 | Zero gradient into the burst branch at step 0 | exact (W_c = 0) | `aux_head` + `train.aux_opt` | implemented, probed |
| 13 | Synthetic training never rewards rejecting a tap | DBSR motion model | `datasets.train.outliers` (+ flow mask) | implemented, probed |
| 14 | The burst is treated as an ordered, fixed-length sequence | scan is a recurrence; N fixed at 14 | `train.burst_aug` | implemented, tested |
| 15 | No weight EMA | config | `train.ema_decay` in M1 | config |
| 16 | Stochastic routing in the MambaIRv2 body at eval | two eval passes differ | proposed (outside KGTS) | not changed |
| 17 | No noise-level conditioning | fusion weights depend on noise | proposed | not changed |
| 18 | `out_norm` discards how much evidence was pooled | code | proposed | not changed |
| 19 | The body re-learns single-image SR from scratch while the burst branch competes with it | M1 body ≠ M0 body | warm start from M0 (config only) | proposed |
| 20 | `budget_check.py`'s total missed `KGTS.precompute` (outside `kgts.forward`) | M1 at 236dbba: 78.1 GMACs, header said ~75.0 | global FlopCounter total; `mamba_job.sh` gate at 90 | fixed |

---

## 1. Geometry: the burst branch can be no better than its tap positions

### 1.1 The problem

A tap's contribution is its content *and* its position code, and the position code is
read off the flow with nothing downstream to correct it (L6's DCN could refine offsets;
KGTS cannot). The repo's own probes (commit 20bfff3) already showed how steep this is —
burst gain after 400 iters: generator flow +0.041 dB, fitted flow (~0.35 px) +0.019,
dense flow (~0.51 px) +0.006. At x8, 0.35 packed px is 2.8 HR px. And at init the learned
flow is ~5 packed px off (random lv3 head), so for the first thousands of iterations the
tokens are gathered at essentially random positions.

### 1.2 Change: photometric affine registration (`align.global_motion: lk`)

`flow_align_arch.affine_lk`: per frame, a robust Gauss–Newton (Lucas–Kanade) registration
of the frame to the keyframe on the burst itself — channel-mean image, affine motion + an
intensity offset, Tukey biweight on the median-centred residual with a MAD scale, coarse to
fine by blur (σ 2 → 0.7), Levenberg damping, 1 px step clip. Two starts (zero motion; the
affine fit of FlowAlign's dense flow); per frame the lower final residual wins. Detached,
no parameters, ~7 MMACs per 48×48 frame (~0.1 GMACs per burst).

Held-out bursts (100 × 13 frames, 32×32 packed, all noise levels), error vs the
generator's geometry in the convention the taps use (backward field, packed px):

| | mean | median | p95 | frames > 0.5 px |
|---|---|---|---|---|
| affine fit of the learned flow (commit 20bfff3, 2.5k CPU iters) | ~0.35 | – | – | – |
| **`affine_lk`, zero start** | **0.079** | **0.067** | 0.168 | 0.1% |
| by noise tercile (low / mid / high) | 0.069 / 0.073 / 0.093 | | | |
| at the official 48×48 packed size (30 bursts; BSDS resized to fit the 432 px crop, so smoother than Zurich) | 0.044 | 0.039 | | |

It converges from zero motion to the same error as from a start 0.33 px off, so on
SyntheticBurst it does not need the learned flow at all; the fit start is there for
larger real motions. Luma beat per-Bayer-channel residuals (0.10): the four packed channels are each
2× subsampled, and their average aliases less.

**Adversarial check that changed the design.** The first version (free photometric gain,
Cauchy weights, mean-|r| scale) was pulled to 0.20 px by a moving object covering 1/16 of
three frames, and to 3.5 px by one covering 1/4 — a free gain can shrink the image term to
zero and make motion unidentifiable. Offset only + Tukey + MAD: 0.090 and 1.3 px. The
quarter-frame case is not solved by a better start either (a phase-correlation start was
worse, 5.3 px: its window weights the centre, where the object sat). That case is what §1.3
is for.

Caveat: with `lk` the tokens' flow is detached, so FlowAlign's flow heads train only from
`flow_opt` / `photo_opt`. Keep one of them on at every step (M1 holds `flow_lambda` at a
floor) or DDP with `find_unused_parameters: false` will fail. On SyntheticBurst the dense
flow then only seeds LK (and feeds lv1 features to the token encoder in `packed` mode); an
ablation dropping FlowAlign's lv2/lv3 and flow heads there would recover most of its ~4
GMACs.

### 1.3 Real bursts: global model + gated local residual (`align.local`)

`local_residual(dense, global, tau, win)`: the dense estimate's error is mostly
incoherent noise; a moving object is a coherent residual. A `win×win` box filter keeps the
latter, averages the former down, and a gate ramping from `tau` to `2·tau` hands those
regions to the (smoothed) dense flow. Test: global model + a 10×10 patch moving (2, −1) px
+ 0.3 px dense noise → EPE inside the patch 0.075 (global alone 2.24), outside 0.000 (dense
alone 0.38). A wrong global estimate (the quarter-frame case above) is itself a coherent
residual, so this is also its safety net. ~Closed on SyntheticBurst.

### 1.4 Real bursts have no flow ground truth (`train.photo_opt`)

`MambaFusionModel.photo_loss`: keyframe vs each frame warped by FlowAlign's lv1 flow
(folded to the packed grid for `bayer`), on the blurred channel mean, masked to in-view
pixels, plus first-order smoothness. Used on batches without `flow_vectors` (or always,
`photo_when: always`). On real DBSR bursts it is minimised at the true flow: 0.0040 /
0.0071 / 0.0121 / 0.0224 at 0 / 0.25 / 0.5 / 1 px offsets. Without it, on RealBSR the flow
heads get gradient only through the tap position codes.

### 1.5 Bug: the flow curriculum runs in micro-steps

`train.py` calls `model.optimize_parameters(micro_step)`, and
`MambaFusionModel.flow_lambda` reads its milestones against that argument. With
`accumulation_steps: 2` (M1) λ reaches 0.1 at optimizer iteration 15k, not 30k; with 4
(L6) at 7.5k. `train.flow_lambda.unit: iter` converts (the same key works for the new
`aux_lambda` / `photo_lambda`); the default stays `micro` so a resumed L6 does not change
schedule mid-run. M1 now uses `iter`.

---

## 2. Token depth (your first point)

### 2.1 What is static is more than the convs

Per frame, the tokens come from 5 convs (FlowAlign lv1 ×3 + `token_enc` ×2; receptive
field 11 packed px, ~77k parameters). The keyframe gets `conv_first` + 24 AttentiveLayers
(windowed attention + a global semantic scan each; 17.8M parameters). In M1's own budget,
align is 4.06 GMACs for all 14 frames; the body is 47.7 GMACs for one.

And it is not only the features. `KGTS.precompute` computes `u = W_u x` (the values),
`W_δ x`, `W_B x` and `W_k x` **once**, and all six calls read that cache. A call can only
change *how much* of each cached value channel it pools (dt, B through γ/β, the affinity) —
never *what* it pools. So the deep calls, whose keyframe state has seen the whole image,
still receive weighted averages of the same per-channel projections of shallow,
11-px-context features. As training improves the body, the gap widens.

### 2.2 Change A: deeper per-frame tokens (`align.token_blocks`)

`TokenBlock`: depthwise 7×7 → LayerNorm → 2× pointwise MLP, last layer zero-init (a stack
is an identity at init). Two blocks: RF 11 → 23 px, +40k params, ~1.3 GMACs for 14 frames
at 48×48 (0.045 GMACs per block per frame).

### 2.3 Change B: tokens re-expressed in the keyframe state (`refine`)

A tap token of pixel p and the keyframe state at p live on the *same* reference grid —
the taps were gathered there. So the whole keyframe context at p can condition every tap
of p without any warping: `TokenRefine`, `x ← x + W2·gelu(W1·LN(x) + V·LN(s))`, W2 zero-init,
invalid taps kept at 0. `refine.at: [i, …]` runs it after call i and rebuilds KGTS's token
cache from the refined bank, so later calls pool *different* values than earlier ones.
Cost per entry at M1 width: TokenRefine ~2.2 GMACs + a second cache ~4.2 GMACs, ~40k params,
~0.7 GB more activations at batch 4 (bf16). Checkpointed with the rest under
`use_checkpoint`. M1 refines once, after call 2, so calls 3–6 read refined tokens.

What it does not do: give a frame spatial context *in its own coordinates* beyond the
encoder's RF — that is 2.2's job. The two are complementary.

Alternatives considered:
- **Per-call linear value adapters** (`u_l = A_l u`): useless. The pooled end state is linear
  in the values, so `Σ_t w_t (A_l u_t) = A_l Σ_t w_t u_t` — a per-call linear map on the values
  is a per-call output projection, which `untie_out` already is. Value adaptation has to be
  *nonlinear and per token* to add anything, which is what TokenRefine is.
- **Running the body's first ASSB on every frame** (Siamese): the most faithful "same
  features for all frames", but ~8 GMACs per frame per ASSB at M1 width, i.e. +100 GMACs
  for 13 frames.
- **Refining at every call**: each refinement pays for a new token cache (~4.2 GMACs at M1
  width); one mid-body refinement is where M1 starts. `refine.at: [2, 4]` is the next step
  if the per-call ablation (§8) still shows the deep calls contributing least.

---

## 3. Head roles (your second point)

### 3.1 Measured: at init every head is the same integrator

M1 shapes (d 64, d_inner 128, n 16, 8 heads, both directions), tokens from real DBSR
bursts at generator geometry. Per head × direction: effective taps read (participation
ratio of the end state's per-tap weights, of 56) / share of weight on the last 8 taps of
the scan (uniform = 14%):

```
fwd 45/15% 40/20% 40/18% 43/17% 44/19% 45/15% 44/18% 47/17%
bwd 42/17% 42/22% 47/16% 46/17% 43/19% 45/17% 40/20% 45/16%
```

Sixteen interchangeable accumulators. Nothing in the init (every channel gets the same A
spectrum and the same dt range) gives any of them a different job, and nothing in the loss
asks for one. Pooled states change 2.3% under a permutation of the 13 non-key frames.

### 3.2 Why tuning A cannot make a head discriminate on its own

With a_max 0.5 and dt ≈ 0.01–0.1, |A|·Σdt is small and the end state is essentially
`y_c = Σ_t dt_ct · u_ct · ⟨C_c ⊙ γ(s), W_B x_t⟩ + …` — linear attention over the 56 taps
with a *fixed* query `C_c` that the keyframe rescales per state dimension. That is a fine
integrator, and it is order-free precisely because the decay is negligible. The decay
spectrum is the only thing a scan has that attention does not — but only the *last* state
is read, so a fast state weights tap t by `exp(A·Σ_{s>t} dt_s)`: **recency**. In frame
order, recency is arbitrary: with a_max 8 the same measurement gives 18–35 effective taps
and **28–55% of the weight on the last 8 taps**, and a frame permutation changes the pooled
states by 9% (4× M1). A fast head in frame order is a detector of whichever frames happen
to come last. This is the "it must consider all scans" problem.

### 3.3 Change: relevance-ordered scans (`kgts.roles`)

Give selector heads an order in which recency *means* relevance. Each head gets a role:

- **`int` — integrate / inject.** Frame order, a_max (0.5): the accumulator the scan
  already was — denoising, gathering every sample.
- **`geo` — geometric selector.** Taps scanned in ascending `−|pos − c|²`, so the samples
  whose RGGB quad is nearest the centre of pixel p's HR block come *last*. (c = 3/16 packed
  px from the keyframe's R sample: the x8 block of p spans [p, p + 7/8], centre 7/16; a tap's
  quad centroid sits 1/4 from its R sample. The keyframe's R sample of p is at HR 8p.)
- **`con` — consistency selector.** Taps scanned in ascending `cons` = −‖content − mean of
  the keyframe's own taps at p‖² (content before the position code, LayerNorm'd), so the taps
  that best match the keyframe come last. Parameter-free, so it ranks meaningfully from step 0.

Selectors use `a_max_sel` (8): their fast states keep a soft top-few of the key, their
slow states a soft top-dozens, and C mixes the scales. How this answers "must consider all
scans":

- **Every head still scans all 56 taps** (all 52 non-key taps for selectors); nothing is
  dropped, only ordered.
- **The order no longer depends on the burst's frame order**: the sort canonicalises it, so
  a selector's end state is *exactly* invariant to permuting the frames (test: max |Δ| = 0
  with every head a selector; frame-order heads changed 12.7% in that test's small random
  setting, 2.3% at M1 init on real bursts). The set is read as a set.
- **Both directions still run**: the backward copy scans the reverse order, so its fast
  states summarise the *least* relevant taps — an outlier / novelty readout (where the burst
  disagrees with the keyframe: occlusion, motion, misregistration), which the gate and W_c
  can use to trust the keyframe more there.
- **The keyframe still steers per call**: orders are token-side and fixed per forward, but
  dt (admission *and* forgetting) still carries `U_δ s` and the per-head affinity, so a deep
  call can zero out a high-ranked tap it disbelieves. Re-sorting per call by the affinity
  was rejected: it costs a gather of the whole `(P, 2·d_inner, L)` value tensor per call,
  and W_q is zero-init, so the order would be arbitrary early on.
- **Selectors never read the keyframe's own taps** (the body has them; ranked by consistency
  they would trivially win the top-k).
- **`out_norm: group`**: one LayerNorm per (direction, head), so selectors (a few taps'
  worth of state) and integrators (56 taps' worth) are normalised separately before W_c
  mixes them, instead of one scale dominating a joint LayerNorm.

Cost: one gather per cached tensor per forward (`KGTS.orders` / `take`), nothing per call.
Needs `TokenBank(extras=True)` (KGTSMamba does it automatically when roles are set).

Alternatives considered: (a) softmax-normalised dt with a per-head temperature (order-free
selection, but it discards the scan's multi-timescale state for a single temperature);
(b) cyclic frame orders per head (every frame is "recent" for some head, but the selection
is then by frame index, not content); (c) per-call re-sorting (above). The relevance order
keeps the scan and makes its one distinctive property, the decay spectrum, useful.

---

## 4. One tied KGTS serves six different depths

- **Tied weights, untied representations.** `U_δ`, `W_q`, γ, β, `W_z`, `W_c` and `W_g` all read
  `norm_s(s)`; `norm_s` fixes the *scale* of the residual stream across depths but not what
  it represents — and it only fixes the input side. At M1 init the mean per-pixel ‖s‖
  entering the six calls is 16.8, 19.4, 23.4, 27.4, 31.8, 38.1 (2.3× growth). The tied
  `W_c` reads a normalised input, so it writes an update of the same size at every depth,
  and the sigmoid gate can only shrink it: the deepest injection is structurally ≥ 2.3×
  weaker *relative to the stream* than the first, whatever the loss wants (in the probe,
  §8, the per-call injection ratio ‖Δs‖/‖s‖ was 2.2% / 0.9% / 0.35% across three calls). **`kgts.depth_embed`** adds a learned zero-init embedding of the call index
  to the normalised state (every keyframe-side projection knows which depth it serves; 1.3k
  params). **`kgts.untie_out`** gives calls 1… their own `W_c` (zero-init) and `W_g` (call 0
  keeps the originals), so each depth writes into its own subspace (0.67M params at M1
  width — the priciest item in M1's new flags; depth_embed alone is the cheap variant). Token-side
  weights, and so the cache, stay shared.
- **The first ASSB never sees the burst.** Injections come after each ASSB; the first one
  (1/6 of the body) denoises and builds features from the single noisy keyframe.
  **`inject_first`** adds a call on the embedded keyframe before it (+1 call, ~2.2 GMACs).

All three are exact identities at init (tested: same output as without them).

---

## 5. Training dynamics across the run

### 5.1 Steps 0 → a few thousand: the burst branch is starved

`W_c` is zero-init, so at step 0 the gradient reaching the scan, the tokens, the token
encoder and (through the position codes) the flow from the SR loss is **exactly zero**, and
it stays proportional to |W_c| while W_c grows by ~lr per Adam step under a 2k-iteration
linear warmup. Meanwhile the body trains as single-image SR and the flow is ~5 px off. The
previous `out_norm` fix shortened this; it cannot remove it.

**`aux_head` + `train.aux_opt`**: a 1×1 conv + pixel shuffle decodes the first call's pooled
states straight to HR, L1 against GT. The whole burst branch is supervised from step 0,
independently of W_c, and the aux PSNR is a live readout of what the burst branch *alone*
carries (the "is the burst being used" question, answered every log line). M1: weight 1.0 →
0.1 over the first 20k iterations, then held (at 0 its parameters would be unused and DDP
would fail; the model refuses an aux head without `aux_opt` for the same reason).

### 5.2 Schedules and averaging

- `flow_lambda.unit: iter` (§1.5).
- `ema_decay: 0.999` in M1 (validation and saving already use `net_g_ema` when present).
- With `lk`, the flow floor (0.05) matters less for the tokens but keeps FlowAlign's heads
  in the graph (§1.2 caveat).

### 5.3 Proposed: warm-start the body from M0

Every new module is zero-init / identity-at-init on purpose, and `W_c` was zero-init from
the start: a KGTSMamba whose body matches a trained single-image model *is* that model at
step 0. M1's body is not M0's (`[4]*6`, inner_rank 64, num_tokens 128 vs `[6]*4`, 32, 64 —
the M1 header says so), so today 300k iterations go partly to re-learning single-image SR
while the burst branch competes with a body that learns faster. With M0's body values (4
injections; 5 with `inject_first`) and `path.pretrain_network_g: <M0 net_g>` +
`strict_load_g: false`, the body starts converged and the burst branch starts in the regime
where the body's remaining error is exactly what the burst can fix (aliasing and noise);
the aux head keeps the branch's gradient alive meanwhile. `load_network` fills the matching
body keys and reports the burst branch as missing. Worth one 35k smoke test against the
from-scratch run.

### 5.4 Late training

What limits the late phase is sub-pixel accuracy (geometry, §1) and how much of the
burst's extra samples survive into the output: the x8 tail squeezes the body's 180
channels to 64 at the packed resolution before three conv + pixel-shuffle stages. Not
changed; a zero-init HR skip from the last call's pooled states (`PS(W·y)`, 0.1 GMACs) is
the cheap experiment if the late-phase burst gain stalls.

---

## 6. Robust for synthetic and real

### 6.1 Synthetic data never rewards rejecting a tap (`datasets.train.outliers`)

DBSR moves each frame by one affine map: no occlusion, no local motion, no brightness
change. Every tap is an inlier, so the consistency machinery (affinity, `con` heads, the
gate) is never *rewarded* for rejecting anything, and `global_motion: affine|lk` even
hard-codes the assumption. The first moving object is met at test time.
`SyntheticBurstDataset(outliers={prob, max_frames, size, shift})` replaces a random box in
1..max_frames non-key frames with that frame's own content displaced by 2–6 packed px (an
independently moving object the keyframe does not show). The GT is untouched, and a
`flow_mask` removes those pixels from the flow loss (`MambaFusionModel.flow_loss(mask=…)`;
a level's pixel counts only if its whole footprint is valid). On in M1 (30% of bursts): in
the fusion probe it cost nothing on clean bursts and halved the damage of corrupted frames
(§8).

### 6.2 The burst is a set of variable size (`train.burst_aug`)

Per batch: shuffle the non-key frames; with probability `prob` keep a random subset of ≥
`min_frames`. The keyframe is moved to the slot the network reads for the new length, the
flow ground truth (and mask) indexed alongside. Integrator heads see random orders (so they
stay order-free); the whole model sees burst lengths it will meet on real data.

### 6.3 Real-data config

`main/configs/M1_KGTSMamba_RealBSR.yml` (untested on data): M1 + `align.local`,
`photo_opt` instead of `flow_opt`, `compand: true`, `burst_aug` from 4 frames, fine-tuned
from an M1 SyntheticBurst run trained with `outliers` on.

### 6.4 Proposed, not implemented: noise-level conditioning

Optimal fusion weights depend on the noise level (low noise: trust the keyframe's own
samples; high noise: average), and SyntheticBurst's shot noise spans two decades. The
model has to infer it. It can in principle — `B_t` and `u_t` are both linear in the token, so
the pooled `Σ_t dt_t (C·B_t) u_t` contains second moments across taps — but only through a
product it has to discover, and the per-pixel LayerNorms before the gather and after the
pool remove the most direct cue (feature magnitude). A per-pixel noise-std map
(`σ(p) = sqrt(shot·I(p) + read)`) as an extra input to `conv_first` and the token encoder is
the standard, cheap fix (KPN). SyntheticBurstVal's
`meta_info.pkl` does carry the levels, but using them at test time would not be comparable
with methods that don't; estimate them blind (e.g. from the temporal variance of the
LK-aligned frames) instead.

---

## 7. Other findings, not changed

- **Stochastic routing in eval.** Both MambaIRv2 copies call `F.gumbel_softmax(..., hard=True)`
  in `ASSM.forward` regardless of `self.training` (as upstream does), so two evaluations of
  the same burst differ (max |Δ| 1e-3 at init in a small CPU model). Validation PSNR is
  noisy at the level of the ablation deltas this project cares about. Argmax routing in
  eval (or a fixed seed in `test()`) is a one-line experiment in the body, outside KGTS.
- **`out_norm` discards how much evidence was pooled.** A pixel at the border (fewer valid
  taps) or a burst where nothing was admitted gives a unit-norm pooled vector like any
  other. Appending log Σdt (or the valid-tap count) before `W_c` would restore a confidence.
- **FlowAlign under `lk` on SyntheticBurst** only seeds LK and supplies lv1 features — see
  §1.2 for the compute it could return.
- **The x8 tail** (§5.4).

---

## 8. Probe results

**Setup.** The real `KGTSMamba` wiring with the ASSBs swapped for 3 conv groups (2
ResBlocks, 48 ch) → 3 KGTS calls; packed align (flow_feat 16), tokens c = d = 32 with M1's
corrections, KGTS n 4 / expand 2 / 4 heads, x8 `pixelshuffledirect`; 0.49M params. The
tokens get the generator's flow (oracle geometry — FlowAlign still trains on the flow loss),
so fusion mechanisms are compared without flow noise. AdamW 4e-4, warmup 100, cosine to 5%
over 600 iterations, batch 4 of 24×24 packed crops (3000 training bursts). Eval on 60
held-out bursts at 32×32, 8 px border: burst gain = PSNR(real) − PSNR(every frame replaced
by the keyframe); robustness = PSNR change when 3 / 6 non-key frames carry a moving-object
box (a quarter of the frame, displaced 4–8 px); at the end, the PSNR lost by skipping each
KGTS call. Arms share seed and data order.

**Caveat — read the paired columns, not absolute PSNR.** The same base configuration read
24.84 dB at iteration 200 on 2 CPU threads and 25.40 dB on 4 (float reduction order changes
the trajectory). Burst gain, the corruption drop and the per-call ablation are measured
within one model and are the comparable numbers. One seed, 600 iterations, a 0.5M-param
body: directions only.

**Status when this section was written: in progress.** Arms, all at the setup above:
A0 base, A1 `roles` [int, int, geo, con] + `out_norm: group`, A2 `token_blocks: 2` +
`refine.at: [0]`, A3 `inject_first` + `aux_head` + `depth_embed` + `untie_out`, R0 / R1 = A0 / A1
trained with moving-object outliers in half the batches. A follow-up commit replaces this
paragraph with the table. Readings so far (iteration 200 of 600, the branch just switching
on — both gains are at the noise floor, nothing to conclude yet):

| arm | burst gain | Δ PSNR, 3 / 6 corrupted frames | \|W_c\| | ‖Δs‖/‖s‖ per call |
|---|---|---|---|---|
| A0 base | +0.002 | −0.000 / −0.001 | 0.220 | 2.1% / 0.9% / 0.3% |
| A1 roles | +0.002 | −0.000 / −0.001 | 0.273 | 2.4% / 0.9% / 0.4% |

---

## 9. Configs, budget, launch order

`main/configs/M1_KGTSMamba.yml` carries every flag of this review (the queued M1 run
starts with them), including `datasets.train.outliers` (30% of bursts): in the fusion
probe (§8) it cost nothing on clean bursts and halved the damage of corrupted frames, and
it is the only thing in synthetic training that rewards rejecting a tap. The same run is
then the starting point for `main/configs/M1_KGTSMamba_RealBSR.yml`.

| at (1, 14, 4, 48, 48) | M1 at 236dbba | M1 now |
|---|---|---|
| params (exact) | 19.121M | 19.919M |
| KGTS calls | 6 | 7 |
| GMACs (FlopCounter + analytic scans) | 78.1 | 88.0 (+0.1 aux head, training only) |
| activations saved for backward, bf16, per sample (estimate) | 2.77 GiB | 3.64 GiB (0.39 with `use_checkpoint`) |

The GMACs were counted on CPU exactly as `budget_check.py` counts on GPU — the selective
scans stubbed to zero-FLOP ops (on GPU they are kernels the counter cannot see) plus its
analytic scan terms; the `layers` row comes out at 47.74 GMACs, the GPU number in M1's old
header. **236dbba's M1 was 78.1 GMACs, not the ~75.0 its header gave**: `budget_check.py`
summed per-module counts, and `KGTS.precompute`'s token-side Linears (4.2 GMACs; 8.5 now,
with the refined second cache) run outside `kgts.forward`, so no module row held them. It
now reports the counter's global total with an `other` row. The deltas of the new M1: +2.2
`inject_first`, +6.4 `refine` (TokenRefine 2.2 + second cache 4.2), +1.3 `token_blocks`,
+0.1 LK.

`main/mamba_job.sh` now gates KGTSMamba configs on it before anything else:
`budget_check --config <cfg> --budget 90` (`MAX_GMACS=90`), then `kgts_sanity`'s scan-parity
and memory stages as before. At batch 4 the activation estimate is ~14.6 GiB (vs ~11.1)
before transients; the memory stage measures the real peak on the GPU and stops the job if
the configured setting OOMs — the fallback is `use_checkpoint: true`, then batch 2 × 4.

Ablations worth running against the new M1 (flags off one at a time, same seed, 35k
smoke tests first), in order of expected information:

1. `align.global_motion: affine` (the previous geometry) — the largest expected effect.
2. `kgts.roles: ~`, `out_norm: true` — watch burst gain and the corrupted-frame drop.
3. `refine: ~` and, separately, `align.token_blocks: 0` — refine is the costly one (6.4 GMACs).
4. `aux_head: false` (+ drop `train.aux_opt`) — watch how early the burst gain appears.
5. `inject_first: false`, `depth_embed: false`, `untie_out: false`.

Then the RealBSR fine-tune from the SyntheticBurst M1 (trained with `outliers` on).

---

## 10. Files changed

- `burstISP/archs/KGTSMamba/kgts_arch.py` — `TokenBank(extras=…)` side information,
  `TokenRefine`, `GroupLayerNorm`, `KGTS(roles, a_max_sel, out_norm='group', n_calls,
  depth_embed, untie_out)`, `forward(idx, return_pooled)`.
- `burstISP/archs/KGTSMamba/kgts_mamba_arch.py` — `inject_first`, `refine`, `aux_head`, call
  indices, bank passed through `forward_features`.
- `burstISP/archs/KGTSMamba/flow_align_arch.py` — `affine_lk`, `gaussian_blur`,
  `local_residual`, `TokenBlock`, `KGTSAlign(global_motion='lk', lk, local, token_blocks)`.
- `burstISP/models/mambafusion_model.py` — schedule `unit`, `aux_opt`, `photo_opt`,
  `burst_aug`, masked `flow_loss(gt, mask)`.
- `burstISP/data/synthetic_burst_dataset.py` — `outliers` + `flow_mask`.
- `analysis/kgts_sanity.py` — role side information in the scan stage, tuple-safe hook, aux
  loss in the overfit stage. `budget_check.py` — `--config` / `--budget` mode, total =
  the counter's global count (the old per-part sum missed `KGTS.precompute`), presets
  `M1-wide` (= the config) and `M1-wide-236dbba`.
- `main/mamba_job.sh` — KGTSMamba pre-flight refuses configs over `MAX_GMACS=90`.
- `analysis/kgts_cpu_checks.py` (new) — the CPU checks below, re-runnable anywhere:
  `python analysis/kgts_cpu_checks.py [--rev 236dbba] [--images <dir of natural images>]`.
- `main/configs/M1_KGTSMamba.yml` (every flag on), `main/configs/M1_KGTSMamba_RealBSR.yml`
  (new: the real-data variant).

Checks run (CPU): flags-off bit-exact vs HEAD on two configurations (packed/affine/M1
token+scan settings; bayer/dt_norm/target), each also with W_c / γ / W_q perturbed (this
caught a 3e-11 difference from a reordered `precompute` — autograd sums a tensor's gradient
in creation order — now restored); every new option builds, runs forward+backward with
every parameter receiving a gradient, and the zero-init ones are exact identities at init;
selector permutation invariance exact; `affine_lk` / `local_residual` / `photo_loss` /
`flow_loss(mask)` / `augment_burst` / schedule units / `outliers` behave as documented;
`kgts_sanity`'s scan stage runs with roles.
