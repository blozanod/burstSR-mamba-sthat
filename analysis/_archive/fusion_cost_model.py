#!/usr/bin/env python3
"""Analytical FLOP / activation-memory model for ST-HAT's stage-1 blocks.

Companion to `param_budget.py`, which prices parameters. This prices the two
things parameters do not predict: multiply-accumulates per forward pass, and
bytes of activation held for the backward pass. Pure arithmetic, no torch, so
config variants and hypothetical blocks can be priced without a GPU.

Written to answer one question: what would replacing `SpatioTemporalBlock`'s
`WindowAttention3D` with a MambaIRv2-style selective scan actually buy?

Conventions
-----------
* "FLOPs" means multiply-accumulates, matching thop/fvcore and therefore the
  numbers BurstMamba (arXiv 2503.19634) and QMambaBSR (arXiv 2408.08665)
  report. Elementwise ops (norms, activations, residual adds) are excluded
  from the MAC count, as in those papers, but their *activations* are counted.
* Activation bytes = tensors autograd holds until backward. This, not the
  parameter count, is what sets `batch_size_per_gpu`.
* Byte widths follow `torch.autocast(bfloat16)` as actually used in
  `mambafusion_model.py:141`. Matmul/conv/linear inputs are held at 2 bytes.
  `softmax`, `log_softmax` and `layer_norm` are on autocast's **fp32 cast
  list**, so they run in fp32 and hold a 4-byte copy of what they saved --
  and where an fp32 result then feeds a matmul, autograd additionally holds
  the 2-byte cast copy. The attention matrix therefore costs 6 bytes per
  element, not 2. `Selective_Scan.forward_core` hardcodes `.float()` on its
  scan inputs (mambairv2_arch.py:388-393), so the scan is priced at 4 bytes.
"""
import math
from dataclasses import dataclass, field

BF16, FP32 = 2, 4


@dataclass
class Cost:
    """MACs and activation bytes, with a per-line breakdown for reporting."""
    macs: int = 0
    act: int = 0
    params: int = 0
    lines: list = field(default_factory=list)

    def add(self, name, macs=0, act=0, params=0):
        self.macs += macs
        self.act += act
        self.params += params
        if macs or act:
            self.lines.append((name, macs, act))
        return self

    def __iadd__(self, other):
        self.macs += other.macs
        self.act += other.act
        self.params += other.params
        self.lines += other.lines
        return self


# ---------------------------------------------------------------- primitives
# Each returns (macs, act_bytes). `act` is what the op saves for backward:
# a matmul/linear/conv saves its input; softmax saves its output; layer_norm
# and the elementwise ops save their input.

def linear(tok, cin, cout, w=BF16):
    return tok * cin * cout, tok * cin * w


def conv(px, cin, cout, k, w=BF16):
    return px * cin * cout * k * k, px * cin * w


def layernorm(tok, dim):
    # fp32 under autocast: holds an fp32 copy of the input on top of the bf16
    # tensor its producer already owns.
    return 0, tok * dim * FP32


def elemwise(tok, dim, n_saved=1, w=BF16):
    return 0, tok * dim * w * n_saved


def softmax_attn(elems):
    # fp32 saved output (softmax backward) + bf16 cast copy (the attn @ v matmul).
    return 0, elems * (FP32 + BF16)


def mlp(tok, dim, ratio):
    """LayerNorm -> Linear -> GELU -> Linear, the ST-HAT stage-1 FFN."""
    c = Cost()
    h = dim * ratio
    c.add('  norm', *layernorm(tok, dim))
    c.add('  fc1', *linear(tok, dim, h), params=dim * h + h)
    c.add('  gelu', *elemwise(tok, h))
    c.add('  fc2', *linear(tok, h, dim), params=h * dim + dim)
    return c


# ------------------------------------------------------------- ST-HAT blocks

def spatiotemporal_block(B, N, C, H, W, ws, heads, ratio):
    """Current block: 3D window attention over (N x ws x ws) tubes."""
    tok = B * N * H * W
    nwin = B * (H // ws) * (W // ws)
    n = N * ws * ws                       # tokens per window
    c = Cost()
    c.add('norm1', *layernorm(tok, C), params=2 * C)
    c.add('wqkv', *linear(tok, C, 3 * C), params=C * 3 * C + 3 * C)
    # q,k,v are slices of one contiguous permute; count the tensor once.
    c.add('qkv (saved by both matmuls)', 0, tok * 3 * C * BF16)
    c.add('q @ k^T', nwin * heads * n * n * (C // heads), 0)
    c.add('softmax(attn)', *softmax_attn(nwin * heads * n * n))
    c.add('attn @ v', nwin * heads * n * n * (C // heads), 0)
    c.add('proj', *linear(tok, C, C), params=C * C + C)
    c.add('residual', *elemwise(tok, C, 0))
    c += mlp(tok, C, ratio)
    c.params += 2 * C + (2 * ws - 1) ** 2 * (2 * N - 1) * heads
    return c


def spatial_block(B, N, C, H, W, ws, heads, ratio):
    tok = B * N * H * W
    nwin = B * N * (H // ws) * (W // ws)
    n = ws * ws
    c = Cost()
    c.add('norm1', *layernorm(tok, C), params=2 * C)
    c.add('wqkv', *linear(tok, C, 3 * C), params=C * 3 * C + 3 * C)
    c.add('qkv', 0, tok * 3 * C * BF16)
    c.add('q @ k^T', nwin * heads * n * n * (C // heads), 0)
    c.add('softmax', *softmax_attn(nwin * heads * n * n))
    c.add('attn @ v', nwin * heads * n * n * (C // heads), 0)
    c.add('proj', *linear(tok, C, C), params=C * C + C)
    c += mlp(tok, C, ratio)
    c.params += 2 * C + (2 * ws - 1) ** 2 * heads
    return c


def temporal_block(B, N, C, H, W, heads, ratio):
    """Per-pixel attention across frames: B*H*W sequences of length N."""
    tok = B * N * H * W
    nseq = B * H * W
    c = Cost()
    c.add('norm1', *layernorm(tok, C), params=2 * C)
    c.add('in_proj_qkv', *linear(tok, C, 3 * C), params=C * 3 * C + 3 * C)
    c.add('qkv', 0, tok * 3 * C * BF16)
    c.add('q @ k^T', nseq * heads * N * N * (C // heads), 0)
    c.add('softmax', *softmax_attn(nseq * heads * N * N))
    c.add('attn @ v', nseq * heads * N * N * (C // heads), 0)
    c.add('out_proj', *linear(tok, C, C), params=C * C + C)
    c += mlp(tok, C, ratio)
    c.params += 2 * C + N * C
    return c


# ------------------------------------------------------- SSM replacement(s)

def selective_scan(tok, d_inner, d_state, w=FP32):
    """mamba_ssm selective_scan_fn + the x_proj/dt_proj that feed it.

    The CUDA kernel recomputes states in backward, so it holds O(tok*d_inner),
    not O(tok*d_inner*d_state): u, delta and out at `w` bytes each, plus the
    per-token B and C. Scan MACs follow the official 9*L*D*N accounting.
    """
    c = Cost()
    dt_rank = math.ceil(d_inner / 16)
    c.add('  x_proj', tok * d_inner * (dt_rank + 2 * d_state), tok * d_inner * w,
          params=d_inner * (dt_rank + 2 * d_state))
    c.add('  dt_proj', tok * dt_rank * d_inner, tok * dt_rank * w,
          params=dt_rank * d_inner + d_inner)
    c.add('  scan (u, delta, out)', 9 * tok * d_inner * d_state, 3 * tok * d_inner * w)
    c.add('  scan (B, C)', 0, 2 * tok * d_state * w)
    c.params += d_inner * d_state + d_inner          # A_logs, Ds
    return c


def assm(tok, C, d_state, expand, num_tokens, inner_rank, scan_w=FP32):
    """MambaIRv2 ASSM: route -> gumbel -> SGN sort -> selective scan -> unsort."""
    c = Cost()
    h = int(C * expand)
    r = C // 3
    # --- SGN router (the 'learn the order' half of the proposal)
    c.add('route fc1', *linear(tok, C, r), params=C * r + r)
    c.add('route gelu', *elemwise(tok, r))
    c.add('route fc2', *linear(tok, r, num_tokens), params=r * num_tokens + num_tokens)
    c.add('log_softmax', 0, tok * num_tokens * FP32)
    c.add('gumbel_softmax(hard)', 0, tok * num_tokens * BF16)
    c.add('prompt = policy @ emb', tok * num_tokens * d_state, 0)
    c.params += num_tokens * inner_rank + inner_rank * d_state
    # --- body
    c.add('in_proj 1x1', *conv(tok, C, h, 1), params=C * h + h)
    c.add('CPE dwconv', *conv(tok, h, 1, 3), params=h * 9 + h)
    c.add('gate x * sigmoid(CPE)', 0, 2 * tok * h * BF16)
    c.add('SGN gather', 0, 0)                       # index only; tensor counted by scan
    c += selective_scan(tok, h, d_state, scan_w)
    c.add('out_norm', *layernorm(tok, h), params=2 * h)
    c.add('out_proj', *linear(tok, h, C), params=h * C + C)
    return c


def st_mamba_block(B, N, C, H, W, heads, ratio, d_state=16, expand=2,
                   num_tokens=64, inner_rank=32, scan_w=FP32):
    """Proposed swap: WindowAttention3D -> ASSM over the spatio-temporal volume.

    Same skeleton as SpatioTemporalBlock (norm -> mixer -> residual -> FFN),
    so the diff against the baseline isolates the mixer.
    """
    tok = B * N * H * W
    c = Cost()
    c.add('norm1', *layernorm(tok, C), params=2 * C)
    c += assm(tok, C, d_state, expand, num_tokens, inner_rank, scan_w)
    c.add('residual', *elemwise(tok, C, 0))
    c += mlp(tok, C, ratio)
    return c


# --------------------------------------------------------------- reporting

def gb(x):
    return x / 1024 ** 3


def report(name, c, per_sample_B=None):
    print(f"\n{name}")
    print(f"  {'':<34}{'GMACs':>10}{'act MB':>11}")
    for ln, m, a in c.lines:
        if m or a:
            print(f"  {ln:<34}{m/1e9:10.2f}{a/1024**2:11.1f}")
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<34}{c.macs/1e9:10.2f}{c.act/1024**2:11.1f}"
          f"   params {c.params/1e3:8.1f}k")
    if per_sample_B:
        print(f"  {'per sample':<34}{c.macs/1e9/per_sample_B:10.2f}"
              f"{c.act/1024**2/per_sample_B:11.1f}")


# ------------------------------------------- restoration head, priced properly
# After FusionBlock collapses the burst, everything downstream runs at N=1.
# That is a 14x token discount, and it is what makes reallocating compute from
# stage 1 into a deeper/wider restoration head so favourable.

def convffn(tok, E, ratio, kf):
    c = Cost()
    h = int(E * ratio)
    c.add('  norm', *layernorm(tok, E), params=2 * E)
    c.add('  fc1', *linear(tok, E, h), params=E * h + h)
    c.add('  dwconv', *conv(tok, h, 1, kf), params=h * kf * kf + h)
    c.add('  act', *elemwise(tok, h))
    c.add('  fc2', *linear(tok, h, E), params=h * E + E)
    return c


def attentive_layer(B, E, H, W, ws, heads, d_state, ratio, num_tokens,
                    inner_rank, kf, scan_w=FP32):
    """One MambaIRv2 AttentiveLayer: window-MHSA + ConvFFN, then ASSM + ConvFFN."""
    tok = B * H * W
    nwin = B * (H // ws) * (W // ws)
    n = ws * ws
    c = Cost()
    c.add('norm1', *layernorm(tok, E), params=2 * E)
    c.add('wqkv', *linear(tok, E, 3 * E), params=E * 3 * E + 3 * E)
    c.add('qkv', 0, tok * 3 * E * BF16)
    c.add('q @ k^T', nwin * heads * n * n * (E // heads), 0)
    c.add('softmax', *softmax_attn(nwin * heads * n * n))
    c.add('attn @ v', nwin * heads * n * n * (E // heads), 0)
    c.add('win proj', *linear(tok, E, E), params=E * E + E)
    c += convffn(tok, E, ratio, kf)
    c.add('norm3', *layernorm(tok, E), params=2 * E)
    c += assm(tok, E, d_state, ratio, num_tokens, inner_rank, scan_w)
    c += convffn(tok, E, ratio, kf)
    c.add('layer_scale x2', *elemwise(tok, E, 2))
    c.params += 2 * E + inner_rank * d_state
    return c


def restoration_head(B, E, H, W, depths, ws=16, heads=4, d_state=16, ratio=2,
                     num_tokens=64, inner_rank=32, kf=5, upsample_feat=64,
                     upscale=4, scan_w=FP32):
    """Full MambaIRv2 trunk: sum(depths) AttentiveLayers + tail."""
    tok = B * H * W
    c = Cost()
    for _ in range(sum(depths)):
        c += attentive_layer(B, E, H, W, ws, heads, d_state, ratio,
                             num_tokens, inner_rank, kf, scan_w)
    for _ in depths:                                   # per-ASSB 3x3 + residual
        c.add('assb conv', *conv(tok, E, E, 3), params=E * E * 9 + E)
    c.add('conv_after_body', *conv(tok, E, E, 3), params=E * E * 9 + E)
    c.add('conv_before_upsample', *conv(tok, E, upsample_feat, 3),
          params=E * upsample_feat * 9 + upsample_feat)
    uf = upsample_feat
    for _ in range(int(math.log2(upscale))):
        c.add('upsample', *conv(tok, uf, 4 * uf, 3), params=uf * 4 * uf * 9 + 4 * uf)
    c.add('conv_last', *conv(B * (upscale * H) * (upscale * W), uf, 3, 3),
          params=uf * 3 * 9 + 3)
    return c


# ------------------------------- flow-indexed recurrent fusion (the proposal)
# Replaces stage 1's spatial search with correspondence indexing: instead of
# attending over a ws x ws x N tube to *find* the matching pixel, sample each
# frame at the flow-predicted location and aggregate along the frame axis only.
# The reference parameterises the state update (QMambaBSR's QSSM gating), so
# the scan asks "what do these frames add that the reference lacks?".
#
# Two things this makes explicit:
#   - The frame axis was never the expensive one. TemporalBlock already mixes
#     N=14 with full attention and its quadratic term is ~0.3 GMACs; the cost
#     was always the ws^2 * N spatial search. Indexing deletes that search, and
#     once it is gone, scan vs. attention along N is a wash on cost -- pick the
#     operator for expressiveness, not FLOPs.
#   - Everything downstream of the aggregation runs on the reference alone
#     (N=1), so the FFN and the flow-refinement head cost 1/14 of a stage-1 FFN.

def flow_fusion_iter(B, N, C, H, W, d_state=16, expand=1, K=1, ffn_ratio=2,
                     refine_flow=True, scan_w=BF16):
    """One gather -> reference-gated scan -> inject iteration.

    K is the gather neighbourhood: K=1 samples only the flow-predicted point,
    K=3 samples a 3x3 patch around it, buying back a little search robustness
    at K^2 the scan length. BurstMamba's Tab. 3 is the warning label here --
    integer-rounded indexing scored *below* no alignment at all (32.434 vs
    32.881), so the gather must stay bilinear whatever K is.
    """
    tok = B * N * H * W * K * K          # gathered tokens entering the scan
    ref = B * H * W                      # reference-only tokens
    d = int(C * expand)
    dt_rank = math.ceil(d / 16)
    c = Cost()
    c.add('grid_sample gather (bilinear)', 0, B * N * H * W * C * BF16
          + B * N * H * W * 2 * FP32)
    c.add('diff vs reference', *elemwise(tok, C, 1))
    c.add('in_proj', *linear(tok, C, d), params=C * d + d)
    # Delta / B / C come from the reference, not the burst: 1/N the tokens.
    c.add('ref-gated dt,B,C proj', *linear(ref, C, dt_rank + 2 * d_state),
          params=C * (dt_rank + 2 * d_state))
    c.add('dt_proj', *linear(ref, dt_rank, d), params=dt_rank * d + d)
    c.add('scan over frame axis', 9 * tok * d * d_state, 3 * tok * d * scan_w)
    c.add('scan (B, C)', 0, 2 * tok * d_state * scan_w)
    c.add('out_proj', *linear(tok, d, C), params=d * C + C)
    c.params += d * d_state + d
    c.add('inject into reference', *elemwise(ref, C, 1))
    c += mlp(ref, C, ffn_ratio)                     # FFN on the reference only
    if refine_flow:
        # Without this the recurrence can only re-weight the same pixels: the
        # gather locations are identical every iteration, so nothing new can
        # enter. A 2-channel residual flow head off the per-frame aggregation
        # error is what makes iteration 2 able to look somewhere else.
        c.add('flow refine head', *conv(B * N * H * W, C, 2, 3),
              params=C * 2 * 9 + 2)
    return c


def flow_fusion(B, N, C, H, W, iters=3, **kw):
    c = Cost()
    for _ in range(iters):
        c += flow_fusion_iter(B, N, C, H, W, **kw)
    return c


# ------------------------------------- rest of the model, for whole-net FLOPs
# Coarser than the stage-1 blocks above (MACs only, no activation breakdown):
# these exist to put the SpatioTemporalBlock's share in context and to compare
# the whole network against BurstMamba's published 63 + 1.3(L-1) GFLOPs.

def fusion_block_macs(B, N, C, H, W, ws, heads, ratio):
    tok = B * N * H * W
    nwin = B * (H // ws) * (W // ws)
    P, n = ws * ws, N * ws * ws
    m = tok * C * C                                   # wk over all frames
    m += (B * H * W) * C * C * 2                      # wq, proj on ref only
    m += tok * C * C                                  # wv on the signed diffs
    m += nwin * heads * P * n * (C // heads) * 2      # q@k^T and attn@v
    ref_tok = B * H * W
    m += ref_tok * (C * (C // 2) + (C // 2) * C + C * C)   # back-projection
    m += ref_tok * (C * ratio * C * 2)                # mlp on the collapsed map
    m += tok * (2 * C * C * 9) + ref_tok * C * C * 9  # gate_conv, diff_fuse
    return m


def ocab_macs(B, C, H, W, ws, heads, ratio, overlap):
    tok = B * H * W
    ows = int(ws * overlap) + ws
    nwin = B * (H // ws) * (W // ws)
    m = tok * C * 3 * C + tok * C * C
    m += nwin * heads * (ws * ws) * (ows * ows) * (C // heads) * 2
    m += tok * C * ratio * C * 2
    return m


def hab_macs(B, C, H, W, ws, heads, ratio):
    tok = B * H * W
    nwin = B * (H // ws) * (W // ws)
    n = ws * ws
    m = tok * C * 3 * C + tok * C * C
    m += nwin * heads * n * n * (C // heads) * 2
    m += tok * C * ratio * C * 2
    m += tok * (C * (C // 3) * 9 * 2 + C * (C // 30) + (C // 30) * C)   # CAB
    return m


def attentive_layer_macs(B, E, H, W, ws, heads, d_state, ratio, inner_rank,
                         num_tokens, kf):
    tok = B * H * W
    nwin = B * (H // ws) * (W // ws)
    n = ws * ws
    h = int(E * ratio)
    dt_rank = math.ceil(h / 16)
    m = tok * E * 3 * E + tok * E * E                        # wqkv, proj
    m += nwin * heads * n * n * (E // heads) * 2             # window attention
    m += tok * (E * (E // 3) + (E // 3) * num_tokens)        # route
    m += tok * num_tokens * d_state
    m += tok * (E * h + h * 9 + h * E)                       # in_proj, CPE, out_proj
    m += tok * (h * (dt_rank + 2 * d_state) + dt_rank * h)   # x_proj, dt_proj
    m += 9 * tok * h * d_state                               # scan
    m += 2 * tok * (E * h + h * kf * kf + h * E)             # two ConvFFNs
    return m


def whole_model_macs(cfg, B, st_block_macs):
    """Total MACs for one forward at the config's training resolution."""
    N, Hb, C = cfg['N'], cfg['H'], cfg['C']
    F, E, ws = cfg['num_feat'], cfg['embed_dim'], cfg['ws']
    heads, ratio = cfg['heads'], cfg['ratio']
    parts = {}
    # PreAlign: conv3x3(4->F) + conv3x3(F->4F) on the packed 48x48 grid, then x2.
    px_lr = B * N * (Hb // 2) * (Hb // 2)
    parts['PreAlign'] = px_lr * (4 * F * 9 + F * 4 * F * 9)
    # BurstAlign: 7 feature convs + flow heads + one DCN, mostly at 96x96.
    px = B * N * Hb * Hb
    cin = F if cfg['pre_align'] else 4
    px2, px3 = px // 4, px // 16                  # lv2 and lv3 are stride-2 each
    al = px * (cin * F * 9 + 2 * F * F * 9)       # feat_extractor_lv1 (3 convs)
    al += px2 * (2 * F * F * 9)                   # feat_extractor_lv2
    al += px3 * (2 * F * F * 9)                   # feat_extractor_lv3
    al += px3 * ((2 * cfg.get('offset_r', 2) + 1) ** 2 * 2 * 9)   # flow_head_lv3
    al += px2 * (2 * F * F * 9 + F * 2 * 9)       # offset_conv_lv2
    al += px * 2 * (2 * F * F * 9 + F * 2 * 9)    # offset_conv_lv1 + _casc
    al += px * (2 * F * F * 9)                    # offset_conv_dcn
    al += px * (F * 144 * 9)                      # offset_proj
    al += px * (F * F * 2 + F * 9)                # DCNv4 value/out proj + sampling
    parts['BurstAlign'] = al
    s1 = cfg['depth_s1']
    parts['ST-HAT s1 spatial'] = s1 * spatial_block(B, N, C, Hb, Hb, ws, heads, ratio).macs
    parts['ST-HAT s1 temporal'] = s1 * temporal_block(B, N, C, Hb, Hb, heads, ratio).macs
    parts['ST-HAT s1 spatiotemporal'] = s1 * st_block_macs
    parts['ST-HAT s2'] = (fusion_block_macs(B, N, C, Hb, Hb, ws, heads, ratio)
                          + spatial_block(B, 1, C, Hb, Hb, ws, heads, ratio).macs)
    parts['ST-HAT s3'] = cfg['depth_s3'] * (
        2 * ocab_macs(B, C, Hb, Hb, ws, heads, ratio, cfg['overlap'])
        + hab_macs(B, C, Hb, Hb, ws, heads, ratio))
    mb = sum(cfg['depths']) * attentive_layer_macs(
        B, E, Hb, Hb, cfg['mb_ws'], cfg['mb_heads'], cfg['d_state'],
        cfg['mb_ratio'], cfg['inner_rank'], cfg['num_tokens'], cfg['kf'])
    mb += len(cfg['depths']) * B * Hb * Hb * E * E * 9 * 2
    up = cfg['upsample_feat']
    mb += B * Hb * Hb * (E * up * 9 + 2 * up * 4 * up * 9) + B * (4 * Hb) ** 2 * up * 3 * 9
    parts['MambaIRv2'] = mb
    return parts


# ------------------------------------------------------------------- driver

L6 = dict(N=14, H=96, C=96, ws=8, st_ws=4, heads=4, ratio=4, overlap=0.25,
          depth_s1=3, depth_s3=1, num_feat=64, embed_dim=96, pre_align=True,
          depths=[2, 2, 2, 2], mb_ws=16, mb_heads=4, mb_ratio=2, d_state=16,
          inner_rank=32, num_tokens=64, kf=5, upsample_feat=64)


def main():
    cfg, B = L6, 2                      # batch_size_per_gpu from the L6 config
    N, H, C, heads, ratio = cfg['N'], cfg['H'], cfg['C'], cfg['heads'], cfg['ratio']
    print(f"MF_STHAT_L6  |  B={B}/gpu  N={N}  C={C}  {H}x{H} (Bayer)  "
          f"{B*N*H*H:,} tokens/fwd")
    print("FLOPs = multiply-accumulates (thop/fvcore convention). "
          "Activations = bytes held for backward under bf16 autocast.")

    base8 = spatiotemporal_block(B, N, C, H, H, 8, heads, ratio)
    base4 = spatiotemporal_block(B, N, C, H, H, cfg['st_ws'], heads, ratio)
    report("SpatioTemporalBlock, ws=8  (L3/L5 setting)", base8, B)
    report(f"SpatioTemporalBlock, ws={cfg['st_ws']}  (current L6 setting)", base4, B)

    variants = [
        ("ST-Mamba, expand=2, fp32 scan (repo's Selective_Scan as-is)",
         st_mamba_block(B, N, C, H, H, heads, ratio, expand=2, scan_w=FP32)),
        ("ST-Mamba, expand=2, bf16 scan (drop the .float() casts)",
         st_mamba_block(B, N, C, H, H, heads, ratio, expand=2, scan_w=BF16)),
        ("ST-Mamba, expand=1, bf16 scan (leanest defensible)",
         st_mamba_block(B, N, C, H, H, heads, ratio, expand=1, scan_w=BF16)),
    ]
    for name, c in variants:
        report(name, c, B)

    print("\n" + "=" * 72)
    print("STAGE-1 TOTALS (x3 blocks, the whole of stage 1's ST path)")
    print(f"  {'variant':<52}{'GMACs':>9}{'act GB':>9}")
    rows = [(f"attention ws=8", base8), (f"attention ws={cfg['st_ws']}", base4)] + variants
    for name, c in rows:
        print(f"  {name:<52}{3*c.macs/1e9:9.1f}{gb(3*c.act):9.2f}")

    print("\n  Reference points at the same B/N/resolution:")
    for name, c in [("SpatialBlock (x3)", spatial_block(B, N, C, H, H, cfg['ws'], heads, ratio)),
                    ("TemporalBlock (x3)", temporal_block(B, N, C, H, H, heads, ratio))]:
        print(f"  {name:<52}{3*c.macs/1e9:9.1f}{gb(3*c.act):9.2f}")

    print("\n" + "=" * 72)
    print("STAGE 1 AS A WHOLE (3 x [spatial + temporal + spatiotemporal])")
    sp = spatial_block(B, N, C, H, H, cfg['ws'], heads, ratio)
    tp = temporal_block(B, N, C, H, H, heads, ratio)
    s1_mac = 3 * (sp.macs + tp.macs + base4.macs)
    s1_act = 3 * (sp.act + tp.act + base4.act)
    # Everything after FusionBlock runs at N=1, so stage 1 holds essentially all
    # of the fusion module's activation memory.
    boundary = 3 * 3 * B * N * H * H * C * BF16      # one saved tensor per segment
    rows = [
        ("baseline (attention, st_ws=4)", s1_mac, s1_act),
        ("drop SpatioTemporalBlock entirely", 3 * (sp.macs + tp.macs),
         3 * (sp.act + tp.act)),
        ("checkpoint stage 1 (no arch change)", s1_mac * 4 // 3, boundary),
        ("ST-Mamba swap (expand=1, bf16 scan)",
         3 * (sp.macs + tp.macs + variants[2][1].macs),
         3 * (sp.act + tp.act + variants[2][1].act)),
        ("ST-Mamba swap + checkpoint stage 1",
         (3 * (sp.macs + tp.macs + variants[2][1].macs)) * 4 // 3, boundary),
    ]
    print(f"  {'intervention':<40}{'GMACs/smp':>11}{'act GB':>9}{'vs base':>9}")
    for nm, m, a in rows:
        print(f"  {nm:<40}{m/1e9/B:11.1f}{gb(a):9.2f}{gb(a)/gb(s1_act):8.0%}")

    print("\n" + "=" * 72)
    print("WHOLE NETWORK, one forward, 48x48 packed LR / 14 frames")
    for label, stc in [("attention ws=4 (current)", base4),
                       ("ST-Mamba expand=1 bf16", variants[2][1])]:
        parts = whole_model_macs(cfg, B, stc.macs)
        tot = sum(parts.values())
        print(f"\n  {label}")
        for k, v in parts.items():
            print(f"    {k:<30}{v/1e9/B:9.1f} GMACs/sample {100*v/tot:6.1f}%")
        print(f"    {'TOTAL':<30}{tot/1e9/B:9.1f} GMACs/sample")
        print(f"    {'BurstMamba (2503.19634), L=14':<30}{79.9:9.1f} GFLOPs/sample")

    # Sensitivity: if softmax turns out to run in bf16 rather than autocast's
    # fp32 (verify with torch.cuda.max_memory_allocated before trusting either),
    # the attention matrix costs 2 B/element instead of 6. That shrinks the
    # baseline and therefore shrinks what any mixer swap can recover -- the
    # ranking of the interventions below is unchanged.
    sm = next(a for nm, _, a in base4.lines if nm.startswith('softmax'))
    lean = s1_act - 3 * (sm * (FP32 + BF16 - BF16) // (FP32 + BF16))
    print(f"\n  Sensitivity: bf16 softmax -> stage 1 = {gb(lean):.2f} GB "
          f"(vs {gb(s1_act):.2f}); the swap recovers proportionally less.")


def reallocation():
    """What a compute budget buys on each side of the FusionBlock collapse.

    Stage 1 runs at N=14; the restoration head runs at N=1. Per AttentiveLayer
    the head is therefore ~14x cheaper than a stage-1 block of the same width,
    which is the whole arithmetic behind moving compute downstream.
    """
    cfg, B, H = L6, 2, 96
    print("\n" + "=" * 78)
    print("RESTORATION HEAD SCALING (N=1, 96x96, window 16, mlp_ratio 2)")
    print(f"  {'config':<44}{'GMACs/smp':>11}{'act GB':>9}{'params':>10}")
    for label, E, depths, ds, ws in [
        ("current: E=96  d=[2,2,2,2]  d_state=16", 96, [2, 2, 2, 2], 16, 16),
        ("deeper:  E=96  d=[6,6,6,6]  d_state=16", 96, [6, 6, 6, 6], 16, 16),
        ("wider:   E=180 d=[2,2,2,2]  d_state=16", 180, [2, 2, 2, 2], 16, 16),
        ("both:    E=180 d=[6,6,6,6]  d_state=16", 180, [6, 6, 6, 6], 16, 16),
        ("both:    E=180 d=[6,6,6,6]  d_state=64", 180, [6, 6, 6, 6], 64, 16),
        ("both:    E=180 d=[8,8,8,8]  d_state=64", 180, [8, 8, 8, 8], 64, 16),
        ("  ^ with window 16 -> 8 (quarters the attn term)", 180, [8, 8, 8, 8], 64, 8),
    ]:
        c = restoration_head(B, E, H, H, depths, ws=ws, d_state=ds,
                             scan_w=BF16)
        print(f"  {label:<44}{c.macs/1e9/B:11.1f}{gb(c.act):9.2f}"
              f"{c.params/1e6:9.2f}M")

    print("\n  One stage-1 SpatioTemporalBlock (N=14) = "
          f"{spatiotemporal_block(B, 14, 96, H, H, 4, 4, 4).macs/1e9/B:.1f} GMACs/sample.")
    al = attentive_layer(B, 96, H, H, 16, 4, 16, 2, 64, 32, 5, BF16)
    print(f"  One AttentiveLayer (N=1, E=96)          = {al.macs/1e9/B:.1f} "
          f"GMACs/sample  ->  1 ST block buys "
          f"{spatiotemporal_block(B, 14, 96, H, H, 4, 4, 4).macs // al.macs} of them.")

    print("\n" + "=" * 78)
    print("BURST-SIDE BREAKDOWN (what the 'lean question' path costs today)")
    parts = whole_model_macs(cfg, B, spatiotemporal_block(B, 14, 96, H, H, 4, 4, 4).macs)
    burst = ['PreAlign', 'BurstAlign', 'ST-HAT s1 spatial', 'ST-HAT s1 temporal',
             'ST-HAT s1 spatiotemporal']
    bt = sum(parts[k] for k in burst)
    print(f"  {'burst-wide path (N=14)':<44}{bt/1e9/B:11.1f}"
          f"  = {bt/1e9/B/14:.1f} GMACs per frame")
    print(f"  {'  BurstMamba burst stream, per frame':<44}{1.3:11.1f}")
    print(f"  {'collapse (FusionBlock + s2 spatial)':<44}"
          f"{parts['ST-HAT s2']/1e9/B:11.1f}")
    print(f"  {'reconstruction (s3 + MambaIRv2)':<44}"
          f"{(parts['ST-HAT s3'] + parts['MambaIRv2'])/1e9/B:11.1f}")
    print(f"  {'  BurstMamba keyframe stream':<44}{63.0:11.1f}")

    # Where the 60.3 GMACs of BurstAlign go. The offset machinery -- three
    # 2F->F->2 refiners, offset_conv_dcn and offset_proj, all at 96x96 x 14
    # frames -- is 65% of it, spent predicting a displacement field that on
    # SyntheticBurst is by construction a global affine (the generator draws
    # translation <= 24 GT px, rotation <= 1 deg, and ships flow_vectors).
    px, F = B * 14 * H * H, cfg['num_feat']
    align = [('feat_extractor_lv1', px * 3 * F * F * 9),
             ('feat_extractor_lv2 (48x48)', (px // 4) * 2 * F * F * 9),
             ('feat_extractor_lv3 (24x24)', (px // 16) * 2 * F * F * 9),
             ('offset_conv_lv1 + _casc', px * 2 * (2 * F * F * 9 + F * 2 * 9)),
             ('offset_conv_lv2 (48x48)', (px // 4) * (2 * F * F * 9 + F * 2 * 9)),
             ('offset_conv_dcn', px * 2 * F * F * 9),
             ('offset_proj', px * F * 144 * 9),
             ('DCNv4 projs + sampling', px * (F * F * 2 + F * 9))]
    print("\n  BurstAlign internals:")
    for nm, m in align:
        print(f"    {nm:<42}{m/1e9/B:9.1f}")
    off = sum(m for nm, m in align if 'offset' in nm)
    print(f"    {'-> offset prediction machinery':<42}{off/1e9/B:9.1f}"
          f"  ({100*off/sum(m for _, m in align):.0f}% of alignment)")

    print("\n" + "=" * 78)
    print("END-TO-END BUDGETS (GMACs/sample, 48x48 LR, L=14)")
    st = spatiotemporal_block(B, 14, 96, H, H, 4, 4, 4).macs
    sp = spatial_block(B, 14, 96, H, H, 8, 4, 4).macs
    tp = temporal_block(B, 14, 96, H, H, 4, 4).macs
    head = lambda E, d, ws=16: restoration_head(B, E, H, H, d, ws=ws,
                                                scan_w=BF16).macs
    pa, ba, s2 = parts['PreAlign'], parts['BurstAlign'], parts['ST-HAT s2']
    scen = [
        ("A. current (L6)",
         pa + ba + 3 * (sp + tp + st), s2, parts['ST-HAT s3'] + head(96, [2]*4)),
        ("B. drop ST block, s1 3->1, head [6,6,6,6]",
         pa + ba + (sp + tp), s2, head(96, [6]*4)),
        ("C. B + affine correspondence, head E=180",
         pa + (ba - off) + (sp + tp), s2, head(180, [6]*4)),
        ("D. C + head depth 8, window 16->8",
         pa + (ba - off) + (sp + tp), s2, head(180, [8]*4, ws=8)),
    ]
    print(f"  {'':<44}{'burst':>8}{'fuse':>7}{'recon':>8}{'total':>8}{'recon%':>8}")
    for nm, b, f, r in scen:
        t = b + f + r
        print(f"  {nm:<44}{b/1e9/B:8.0f}{f/1e9/B:7.0f}{r/1e9/B:8.0f}"
              f"{t/1e9/B:8.0f}{100*r/t:7.0f}%")
    print(f"  {'BurstMamba, L=14 (44.51 dB SyntheticSR)':<44}"
          f"{16.9:8.0f}{0:7.0f}{63.0:8.0f}{79.9:8.0f}{79:7.0f}%")


def flow_fusion_report():
    B, N, C, H = 2, 14, 96, 96
    print("\n" + "=" * 78)
    print("FLOW-INDEXED RECURRENT FUSION vs ST-HAT STAGE 1")
    st = spatiotemporal_block(B, N, C, H, H, 4, 4, 4)
    sp = spatial_block(B, N, C, H, H, 8, 4, 4)
    tp = temporal_block(B, N, C, H, H, 4, 4)
    s1m, s1a = 3 * (st.macs + sp.macs + tp.macs), 3 * (st.act + sp.act + tp.act)
    print(f"  {'':<46}{'GMACs/smp':>11}{'act GB':>9}")
    print(f"  {'ST-HAT stage 1 (3 x 3 blocks)':<46}{s1m/1e9/B:11.1f}{gb(s1a):9.2f}")
    for it in (1, 2, 3, 4):
        c = flow_fusion(B, N, C, H, H, iters=it, K=1)
        print(f"  {'flow-indexed fusion, K=1, ' + str(it) + ' iterations':<46}"
              f"{c.macs/1e9/B:11.1f}{gb(c.act):9.2f}")
    for it in (2, 3):
        c = flow_fusion(B, N, C, H, H, iters=it, K=3)
        print(f"  {'flow-indexed fusion, K=3 (3x3 gather), ' + str(it) + ' iters':<46}"
              f"{c.macs/1e9/B:11.1f}{gb(c.act):9.2f}")

    print("\n  Why the frame axis was never the problem:")
    for nm, blk, q in [("SpatioTemporalBlock (ws=4)", st,
                        st.lines[3][1] + st.lines[5][1]),
                       ("TemporalBlock (N=14 attention)", tp,
                        tp.lines[3][1] + tp.lines[5][1])]:
        print(f"    {nm:<44}{blk.macs/1e9/B:8.1f} total, "
              f"{q/1e9/B:5.2f} of it quadratic ({100*q/blk.macs:.0f}%)")

    print("\n" + "=" * 78)
    print("END-TO-END: flow-indexed fusion + expanded head")
    cfg = L6
    parts = whole_model_macs(cfg, B, st.macs)
    pa, ba = parts['PreAlign'], parts['BurstAlign']
    px, F = B * N * H * H, cfg['num_feat']
    # Keep the flow half of BurstAlign (feature pyramid + cost-volume head),
    # drop the offset half (offset_conv_lv1/_casc/_dcn, offset_proj, DCN).
    flow_only = (px * 3 * F * F * 9 + (px // 4) * 2 * F * F * 9
                 + (px // 16) * 2 * F * F * 9 + (px // 16) * 25 * 2 * 9
                 + (px // 4) * (2 * F * F * 9 + F * 2 * 9))
    ff = flow_fusion(B, N, C, H, H, iters=3, K=1)
    print(f"  {'':<46}{'GMACs/smp':>11}{'act GB':>9}")
    rows = [
        ("PreAlign", pa, None),
        ("BurstAlign, flow head only (offsets deleted)", flow_only, None),
        ("flow-indexed fusion, 3 iterations", ff.macs, ff.act),
        ("MambaIRv2 head E=180 d=[6,6,6,6] d_state=64",
         restoration_head(B, 180, H, H, [6]*4, d_state=64, scan_w=BF16).macs,
         restoration_head(B, 180, H, H, [6]*4, d_state=64, scan_w=BF16).act),
    ]
    tot = sum(m for _, m, _ in rows)
    for nm, m, a in rows:
        act = f"{gb(a):9.2f}" if a else f"{'-':>9}"
        print(f"  {nm:<46}{m/1e9/B:11.1f}{act}")
    print(f"  {'-' * 66}")
    print(f"  {'TOTAL':<46}{tot/1e9/B:11.1f}")
    print(f"  {'current L6':<46}{sum(parts.values())/1e9/B:11.1f}")
    print(f"  {'BurstMamba L=14 (44.51 dB)':<46}{79.9:11.1f}")
    rec = rows[3][1]
    print(f"\n  reconstruction share: {100*rec/tot:.0f}%  "
          f"(current 11%, BurstMamba 79%)")


def budget_match():
    """Can the proposed shape fit inside BurstMamba's 79.9 GFLOPs / 21.4M?

    The lever that decides it is resolution, not architecture. MambaFusion runs
    its restoration head on the Bayer 96x96 grid; BurstMamba runs its backbone
    at the 48x48 LR input and upsamples at the end. That is 4x the pixels per
    layer, and it is why 20.6M params at 63 GFLOPs looks unreachable from here:
    at 96x96 the same FLOPs buy a quarter of the network.

    Alignment and fusion have a reason to be on the fine grid -- sub-pixel
    motion lives there. Reconstruction does not: a pixel-unshuffle back to
    48x48 x 4C is lossless, and running the head there is what SwinIR and
    MambaIRv2 do anyway. Note this partially reverts L5, which moved
    restoration onto the Bayer grid along with alignment and fusion.
    """
    B, N, C, H = 2, 14, 96, 96

    def flow_path(res, F):
        px = B * N * res * res
        return (px * F * F * 9 * 3 + (px // 4) * 2 * F * F * 9
                + (px // 16) * 2 * F * F * 9 + (px // 16) * 25 * 2 * 9
                + (px // 4) * (2 * F * F * 9 + F * 2 * 9))

    pre = B * N * 24 * 24 * (4 * 64 * 9 + 64 * 4 * 64 * 9)
    print("\n" + "=" * 78)
    print("BUDGET MATCH vs BurstMamba (79.9 GFLOPs / 21.4M params)")
    print(f"  {'':<42}{'GMACs':>8}{'params':>9}{'act GB':>9}")
    for label, fres, F, it, E, d, ds, hres, up in [
        ("flow@96 F=64, fuse x3, head@96 E=180", 96, 64, 3, 180, [6]*4, 64, 96, 4),
        ("flow@96 F=64, fuse x3, head@48 E=180", 96, 64, 3, 180, [6]*4, 64, 48, 8),
        ("flow@48 F=64, fuse x3, head@48 E=180", 48, 64, 3, 180, [6]*4, 64, 48, 8),
        ("flow@48 F=32, fuse x3, head@48 E=180", 48, 32, 3, 180, [6]*4, 64, 48, 8),
        ("flow@48 F=32, fuse x2, head@48 E=180 d8", 48, 32, 2, 180, [8]*4, 64, 48, 8),
    ]:
        fl = flow_path(fres, F)
        fu = flow_fusion(B, N, C, H, H, iters=it, K=1)
        hd = restoration_head(B, E, hres, hres, d, d_state=ds, scan_w=BF16,
                              upscale=up)
        tot = pre + fl + fu.macs + hd.macs
        print(f"  {label:<42}{tot/1e9/B:8.1f}{(fu.params+hd.params)/1e6:8.1f}M"
              f"{gb(fu.act + hd.act):9.2f}")
        split = (f"pre {pre/1e9/B:.1f} + flow {fl/1e9/B:.1f} + "
                 f"fuse {fu.macs/1e9/B:.1f} + head {hd.macs/1e9/B:.1f}")
        print(f"    {split}")
    print(f"  {'current L6':<42}{276.6:8.1f}{5.0:8.1f}M")
    print(f"  {'BurstMamba L=14 (44.51 dB)':<42}{79.9:8.1f}{21.4:8.1f}M")


if __name__ == '__main__':
    main()
    reallocation()
    flow_fusion_report()
    budget_match()
