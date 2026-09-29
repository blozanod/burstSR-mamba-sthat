# KGTS / KGTSMamba — Architecture Context

Self-contained description of the **Keyframe-Gated Tap Scan (KGTS)** module and the **KGTSMamba** (config `M1_KGTSMamba`) burst super-resolution model it lives in. Written so another model can understand the design without reading the code.

Source files (repo `burstSR-mamba-sthat`, all under `burstISP/archs/KGTSMamba/`):
`kgts_mamba_arch.py` (full model), `kgts_arch.py` (`tap_gather`, `TokenBank`, `KGTS`), `flow_align_arch.py` (`KGTSAlign`, `FlowAlign`, `PreAlign`, `PostAlign`), `mambairv2_arch.py` (base trunk), `budget_check.py`. Config: `main/configs/M1_KGTSMamba.yml`. Pre-flight test: `analysis/kgts_sanity.py`.

---

## 1. Task and idea

- **Task:** multi-frame raw burst → RGB super-resolution (x8 on the packed grid) + demosaicing. Input is a burst `(B, N=14, 4, H, W)` of packed RGGB frames (synthetic burst dataset, Zurich-RAW-to-RGB). Output `(B, 3, 8H, 8W)` linear RGB. Loss: L1 on linear RGB, plus a flow-supervision curriculum.
- **Core design choice:** nearly all compute goes into **single-image SR on the keyframe** (frame `N//2`, a MambaIRv2 trunk = the M0 baseline). The other 13 frames are **not** processed by a fusion trunk. They reach the model only through KGTS, a small weight-tied module the keyframe *queries* after every trunk block. The burst is treated as side information.

## 2. Novelty (what is distinctive)

1. **Fusion = a selective (Mamba S6) scan over a pixel's tap sequence.** For each packed pixel, all flow-indexed samples from all frames (`L = N·k·k`, e.g. 14·2·2 = 56) form a short sequence. A selective scan pools it. This replaces cross-frame attention or 3D convolution.
2. **The keyframe state gates the scan.** The scan's admission/decay `Δ` and input matrix `B` are conditioned on the current keyframe feature `s`, not only on the tokens:
   - `Δ = softplus(W_δ·x_tok + U_δ·s + b)` (admission gate)
   - `B = (W_B·x_tok) ⊙ γ(s) + β(s)` (FiLM; identity at init)
   So the keyframe decides *which taps each channel admits* and how strongly. Standard Mamba conditions only on its own tokens.
3. **Sub-pixel phase is a token feature.** `tap_gather` snaps the flow target to integer pixels, gathers a k×k neighbourhood, and keeps the fractional remainder as a positional embedding, in fp32 (bf16 would quantise sub-pixel offsets to ≥0.25 px). Out-of-frame taps are masked (`valid`) and given `Δ≈0`, so they neither decay nor inject.
4. **One weight-tied recurrent module, called after every ASSB.** The same KGTS weights run `len(depths)` times (6). Token-only terms are computed once per forward (`precompute`) and cached; each call adds only keyframe-side terms. KGTS is ~0.26M params vs ~17.8M for the trunk.
5. **Exact-identity start.** The injection projection `W_c` is zero-initialised, so the model starts as exactly the keyframe-only M0 and the burst path grows in during training. (Consequence: no gradient reaches alignment until KGTS begins injecting, hence the flow-supervision curriculum.)

## 3. Data flow

```
burst (B,N,4,H,W)  packed RGGB
  │  pad whole burst to multiple of window_size (reflect), subtract mean, * img_range
  ▼
KGTSAlign ─► feats (B,N,c=64,h,w), flow (B,N,2,h,w) [packed px, fp32], flows{lv1,lv2,lv3}
  ▼
TokenBank ─► tokens (B·h·w, N·k², d=64), valid (B·h·w, N·k²)
  ▼
KGTS.precompute ─► cache = (u, W_δ x, W_B x, valid)          # once per forward

keyframe = burst[:, ref]  (ref = N//2)
  ▼ conv_first
  ▼ [ ASSB layer_i → KGTS(s, cache) ] × 6        # same KGTS weights each time
  ▼ norm → patch_unembed → conv_after_body (+ skip from conv_first)
  ▼ pixelshuffle upsampler (x8)  [or pixelshuffledirect]
  ▼ / img_range + mean, crop padding
RGB (B,3,8H,8W)     [optional aux: {'flows': ...} for the flow loss]
```

`KGTSMamba` subclasses `MambaIRv2`; the burst branch (`align`, `bank`, `kgts`) is built *after* `super().__init__` so MambaIRv2's `self.apply(_init_weights)` cannot overwrite KGTS's zero-init `W_c` / identity FiLM. Never call `self.apply(...)` on the model afterwards. KGTS's `d` is forced to `token.d` and its `ds` to `embed_dim`.

## 4. Components

### 4.1 KGTSAlign (front-end)
Produces per-frame token features and a flow, both on the **packed grid in packed-pixel units** (sign convention: content the reference sees at `p` sits at `p − flow(p)`).

- **FlowAlign:** 3-level PCD-style pyramid (stride-2 extractors), no DCN. Cost volume at lv3 (normalised correlation, radius `r`, `(2r+1)²` channels) → 2-channel flow head; lv2 and lv1 add residual flow (zero-init refinement convs, so the lv3 head drives early training) using internally warped features; one extra lv1 cascade refinement. Frame-count agnostic. Returns `flows{lv1,lv2,lv3}` and unwarped lv1 features. Feature warping for fusion is *not* done here; it is implicit in the tap gather.
- **`type: packed` (used by M1):** FlowAlign runs directly on the 48×48 packed frames (~4× cheaper). Token features come from a small conv encoder over `[raw frame ; FlowAlign lv1 features]`, so `flow_feat` only needs to be sized for matching, while raw samples (the sub-pixel content KGTS delivers) reach the tokens directly. One lv3 px = 4 packed px = 32 GT px, so `r=1` already covers the 24 GT px max translation; M1 uses `r=2`.
- **`type: bayer`:** PreAlign (conv + PixelShuffle 2 → Bayer grid) → FlowAlign at 2× resolution → PostAlign (PixelUnshuffle + convs for features; parameter-free avg-pool-2 and ×0.5 for flow). More precise flow, ~4× the cost.

### 4.2 tap_gather + TokenBank
- `tap_gather(feat, flow, k)`: `s = xy − flow` (fp32). Snap with `floor` (even k) or `round` (odd k) to get integer base `n` and remainder `δ = s − n`. For each offset in a k×k window, gather `feat[n+o]`, zero it if out of bounds, and record position `(ox − δx, oy − δy)`. Returns taps `(B,N,k²,C,h,w)`, pos `(…,2,h,w)` fp32, valid `(…,h,w)`. Nothing is interpolated; the geometry is handed to the scan as a positional embedding.
- `TokenBank(c, d, k)`: 1×1 conv `c→d`, `tap_gather`, reshape to `(P=B·h·w, L=N·k², d)`, add `pos_encode` (Linear(2,d)–GELU–Linear(d,d)) of the fractional position, zero invalid taps. Built once; shared by every KGTS call.

### 4.3 KGTS (the module)
Inputs: keyframe feature `s` `(P, ds)` (the trunk's residual stream at one pixel, `ds = embed_dim = 180`) and the cache. One instance, called after each ASSB.

Per call:
1. `sn = LayerNorm(s)` (`norm_s`; needed because tied weights otherwise see a residual stream whose scale grows with depth, and the `W_g` sigmoid saturates on deep calls).
2. **Conditioning** (channels `d_inner = expand·d`, keys `heads·n`):
   - `Δ = W_δ x + U_δ sn` (token side cached, keyframe side per call); invalid taps masked to −30 *after* the sum (softplus ≈ 0).
   - `B = (W_B x) ⊙ γ(sn) + β(sn)`; `γ` init weight 0 / bias 1, `β` init 0.
   - `u = W_u x` (identity if `expand=1`); `A = −exp(A_log)` (S4D-real init); `C` a fixed learned readout (the keyframe enters through γ/β/Δ, not C); `Δ` bias initialised so `dt ∈ [dt_min, dt_max]`.
3. **Bidirectional scan in ONE kernel call:** the backward copy is flipped and stacked on the channel axis (`2·d_inner` channels, `2·heads` B-groups, VMamba cross-scan style) and run through `mamba_ssm.selective_scan_fn` (grouped-B layout). Only the **final state** of each direction is used: `y ∈ (P, 2·d_inner)` — a per-channel, keyframe-conditioned pooling over the L taps. CPU/testing path uses a pure-torch `ref_scan`.
4. **Output gate** (`out_gate`): `y ← y ⊙ silu(W_z sn)` — Mamba's z-branch, driven by the keyframe.
5. **Injection:** `d_s = W_c y` (**zero-init**), `g = sigmoid(W_g [sn ; d_s])`, return `s + g ⊙ d_s`.

Capacity knobs: `expand` (pooled width; `d_inner`, not `d`, is the number of distinct things one injection can deliver, and it grows the scan without growing the cached token bank), `heads` (independent key spaces so channels can rank taps by different criteria, e.g. occlusion/brightness mismatch vs sub-pixel phase; no extra kernel launches), `n` (state / key-query dimension per channel).

## 5. M1 configuration (`M1_KGTSMamba.yml`)

| Group | Values |
|---|---|
| Body (MambaIRv2) | `embed_dim 180`, `depths [4]*6`, `num_heads [6]*6`, `window 16`, `d_state 64`, `inner_rank 64`, `num_tokens 128`, `mlp_ratio 2`, `img_size 48`, `upscale 8`, pixelshuffle upsampler (`upsample_feat 64`) |
| Align | `type packed`, `flow_feat 32`, `r 2`, `ref_idx` = `N//2` |
| Token | `c 64`, `d 64`, `k 2` → `L = 14·4 = 56` |
| KGTS | `n 16`, `expand 2` (d_inner 128 → 256 pooled dims/injection), `heads 8`, `norm_s`, `out_gate`, `dt 0.01–0.1` |
| Budget (measured, GPU, (1,14,4,48,48)) | 19.09M params, 73.4 GMACs, 0.82 GB eval peak; align 4.06 GMACs / 0.172M; KGTS 12.64 GMACs / 0.260M; layers 47.74 GMACs / 17.80M |
| Training | AdamW 1e-4, cosine 300k iters, warmup 2k, effective batch 32 (4/GPU × 4 GPUs × accumulation 2), bf16 autocast, grad-clip 1 |
| Losses | L1 on linear RGB; Charbonnier flow loss on FlowAlign's pyramid (`lv1..lv3`) vs generator `flow_vectors`, λ schedule 0.5 → 0.1 → 0 at iters 0 / 30k / 60k |

## 6. Caveats

- M1's body (`[4]*6`, `inner_rank 64`, `num_tokens 128`) differs from M0's (`[6]*4`, `inner_rank 32`, `num_tokens 64`), so M1 − M0 is **not** a pure "burst on/off" ablation. Run the config with M0's body values for that.
- Training memory unverified: KGTS holds `(B·48·48, 2·d_inner, 56)` scan inputs per injection. If OOM, set `use_checkpoint: true` (checkpoints every ASSB and KGTS call), then reduce batch/increase accumulation.
- Mixed precision: coordinates and flow are fp32 throughout; `delta` and `B` are cast to `u`'s dtype only for the kernel.
- Requires `mamba_ssm` on CUDA; without it the scan raises (the pure-torch `ref_scan` is for tests only).
