import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from burstISP.utils.registry import ARCH_REGISTRY
from einops import rearrange

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

def tap_gather(feat, flow, k=2, tap_pos='target'):
    """
    x: [B, N, C, h, w]
    flow: [B, N, 2, h, w]

    Returns:
    taps: [B, N, k*k, C, h, w]
    pos: [B, N, k*k, 2, h, w]   fp32, where the tap's sample sits in the reference, relative to p
    valid: [B, N, k*k, h, w]

    The window is always found at p - flow(p). tap_pos says where each tap's
    sample then sits, and must match what the flow IS:
    'target': T - (p - flow(p)), the flow read at p (the original). First-order
        only: off by |grad flow| * |flow| for a forward field, by
        |grad flow| * |T - p| for a backward one.
    'forward': the flow is a forward field -- frame i's pixel T shows what the
        reference shows at T + flow(T) (the generator's flow_vectors). Exact.
    'backward': the flow is a backward field -- the reference's p shows what
        frame i shows at p - flow(p) (FlowAlign.warp's convention). The tap's
        sample sits at the q with q - flow(q) = T: two fixed-point steps,
        q = T + flow(T + flow(T)), the second one bilinear. Error |grad flow|^2 |flow|.
    Max error against the generator's true sample positions (SyntheticBurst:
    1 deg rotations, 24 GT px shifts; analysis/kgts_sanity.py stage geometry),
    packed px: 'forward' on its field 0.004, 'backward' on its field 0.005 (both
    ~ the Bayer block centre vs its R sample); 'target' 0.072 on the forward
    field (0.57 HR px), 0.022 on the backward one; a precise mode on the wrong
    field ~0.05.

    Coordinates are always fp32. Under bf16 autocast the flow arrives as bf16,
    whose spacing is 0.25 px between 32 and 64 (0.5 px up to 128): doing the
    xs - flow arithmetic in bf16 would quantise the sub-pixel offsets that
    `pos` exists to carry to a quarter of a packed pixel -- coarser than the
    1/8 px an x8 model has to resolve.
    """
    if tap_pos not in ('target', 'forward', 'backward'):
        raise ValueError(f"tap_pos must be 'target', 'forward' or 'backward', got {tap_pos!r}")
    B, N, C, h, w = feat.shape
    feat = feat.reshape(B * N, C, h * w)
    flow = flow.reshape(B * N, 2, h, w).float()

    ys, xs = torch.meshgrid(torch.arange(h, device=feat.device, dtype=flow.dtype),
                            torch.arange(w, device=feat.device, dtype=flow.dtype), indexing='ij')

    # raw tap
    sx = xs - flow[:,0]
    sy = ys - flow[:,1]
    # whole number
    snap = torch.floor if k % 2 == 0 else torch.round
    nx, ny = snap(sx), snap(sy)
    # remainder
    dx, dy = sx - nx, sy - ny

    offs = range(-((k - 1) // 2), k // 2 + 1)
    taps, pos, valid = [], [], []
    fpad = None   # 'backward': the flow, linearly extrapolated past the border (exact for affine fields)
    for oy in offs:
        for ox in offs:
            tx, ty = (nx + ox).long(), (ny + oy).long()

            # mask for valid
            ok = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
            valid.append(ok)

            # gather pixels
            flat = (ty.clamp(0, h - 1) * w + tx.clamp(0, w - 1)).view(B * N, 1, h * w)
            taps.append(feat.gather(2, flat.expand(-1, C, -1)).view(B * N, C, h, w) * ok.unsqueeze(1))

            # position
            if tap_pos != 'target':
                ft = flow.view(B * N, 2, h * w).gather(2, flat.expand(-1, 2, -1)).view(B * N, 2, h, w)
                qx, qy = tx + ft[:, 0], ty + ft[:, 1]                # the flow read at T
                if tap_pos == 'backward':                            # q = T + flow(q): once more, at T + flow(T)
                    if fpad is None:                                 # valid taps land within ~1 px of the grid
                        m = min(2, h - 1, w - 1)
                        fpad = 2 * F.pad(flow, (m,) * 4, mode='replicate') - F.pad(flow, (m,) * 4, mode='reflect')
                    grid = torch.stack(((2 * (qx + m) + 1) / (w + 2 * m) - 1, (2 * (qy + m) + 1) / (h + 2 * m) - 1), -1)
                    fq = F.grid_sample(fpad, grid, mode='bilinear', padding_mode='border', align_corners=False)
                    qx, qy = tx + fq[:, 0], ty + fq[:, 1]
                pos.append(torch.stack((qx - xs, qy - ys), 1))
            else:
                pos.append(torch.stack((ox - dx, oy - dy), 1))

    return (torch.stack(taps, 1).view(B, N, k * k, C, h, w),
            torch.stack(pos, 1).view(B, N, k * k, 2, h, w),
            torch.stack(valid, 1).view(B, N, k * k, h, w))

class TokenBank(nn.Module):
    """
    c: channels
    d: CFQ token dimension
    k: taps for tap_gather (for ablation, 2 default)
    pos_freqs: Fourier frequencies, in cycles per packed px, of the tap-offset
        encoding: pos -> [pos, sin 2pi f pos, cos 2pi f pos] before the MLP.
        A tap's offset is the sub-pixel position of its sample, the thing an x8
        model has to resolve to 1/8 px; an MLP on the raw offset is spectrally
        biased to smooth functions of it (fit to 8x8 sigma=1/8 px kernels:
        R^2 0.42 raw vs 0.98 with f up to 4 = the HR Nyquist). Integer f cannot
        tell the two taps of a k=2 pair apart (their offsets differ by exactly 1);
        pos itself and f=0.5 do. () -> the original raw-offset encoder.
    norm: LayerNorm the content features (per pixel, over channels) before
        the gather. The token encoder's default-init output is ~0.03 RMS and
        varies across the taps of a pixel far less than the position code
        does: 98.6% of a token's tap-to-tap variance at init was position and
        1.3% content, so the scan pooled where the samples were, not what they
        were, and the injection had nothing to learn from (norming content +
        position together: 2.8%; this: 71%). The pooled end states rise from
        ~2e-3 to ~1 RMS at init with it (cf. out_norm). It also keeps dt's
        temperature from tracking whatever scale the encoder drifts to. Before
        the gather it is k*k times cheaper, and gathering commutes with it.
    mark_ref: add a learned (zero-init) embedding to the keyframe's own taps.
        Otherwise they are indistinguishable from any other frame's, and the
        scan cannot learn to discount content the body already has.
    pin_ref: gather the keyframe's taps at flow 0. Its estimated self-flow is
        noise around an exact 0, and the noise's sign flips floor() between
        the {p-1, p} and {p, p+1} tap pairs. The flow loss still sees the
        estimate (KGTSMamba returns FlowAlign's own pyramid).
    tap_pos: see tap_gather. It must name the field the flow is
        trained as: KGTSMamba's FlowAlign is supervised by MambaFusionModel.flow_loss,
        whose train.flow_target picks 'forward' (the generator's flow_vectors) or
        'backward' (the field FlowAlign's own warps assume).

    Tokenizes the aligned taps once, used for every CFQ

    Out:
    x: [P, L, d]
    valid: [P, L]
    with extras=True also a dict of per-tap side information, for KGTS.roles:
        pos: [P, L, 2] fp32, the tap's sample position relative to p (packed px)
        cons: [P, L] fp32, -mean_c (content - keyframe content)^2: how well the tap's
            content (before the position code) matches the mean of the keyframe's own
            taps at p. Parameter-free, so it ranks taps meaningfully from step 0.
        is_ref: [L] bool, the keyframe's own taps
    """
    def __init__(self, c, d, k=2, pos_freqs=(0.5, 1, 2, 4), norm=True, mark_ref=True, pin_ref=True,
                 tap_pos='backward'):
        super().__init__()
        self.pin_ref = pin_ref
        self.tap_pos = tap_pos
        self.proj = nn.Conv2d(c, d, 1)
        self.register_buffer('pos_freqs', torch.tensor(pos_freqs, dtype=torch.float32).view(-1),
                             persistent=False)
        self.pos_encode = nn.Sequential(nn.Linear(2 + 4 * len(self.pos_freqs), d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d) if norm else nn.Identity()
        self.ref_embed = nn.Parameter(torch.zeros(d)) if mark_ref else None
        self.k = k

    def fourier(self, pos):
        """(..., 2) fp32 offsets -> (..., 2 + 4F) features."""
        if not len(self.pos_freqs):
            return pos
        a = 2 * math.pi * pos[..., None] * self.pos_freqs                  # (..., 2, F)
        return torch.cat([pos, a.sin().flatten(-2), a.cos().flatten(-2)], -1)

    def forward(self, feats, flow, ref=None, extras=False):
        B, N, C, H, W = feats.shape

        feats = rearrange(feats, 'b n c h w -> (b n) c h w')
        feats = self.proj(feats)
        # autocast runs LN in fp32: back to the incoming dtype, or the fp32 tokens would be
        # re-cast to bf16 by every Linear in KGTS.precompute, each copy saved for backward
        feats = self.norm(feats.permute(0, 2, 3, 1)).to(feats.dtype)          # (bn, h, w, d)
        feats = rearrange(feats, '(b n) h w c -> b n c h w', b=B)

        if self.pin_ref and ref is not None:
            flow = flow * (torch.arange(N, device=flow.device) != ref).view(1, N, 1, 1, 1)
        taps, pos, valid = tap_gather(feats, flow, self.k, self.tap_pos)

        x = rearrange(taps, 'b n k c h w -> (b h w) (n k) c')
        pos = rearrange(pos, 'b n k c h w -> (b h w) (n k) c') # c = 2
        valid = rearrange(valid, 'b n k h w -> (b h w) (n k)')

        side = self.side_info(x, pos, valid, ref) if extras else None
        pos = self.pos_encode(self.fourier(pos))

        x = x + pos
        if self.ref_embed is not None and ref is not None:
            is_ref = torch.arange(x.shape[1], device=x.device) // self.k ** 2 == ref
            x = x + is_ref.to(x.dtype)[:, None] * self.ref_embed.to(x.dtype)   # no fp32 promotion
        x = x * valid[..., None]

        return (x, valid) if side is None else (x, valid, side)

    @torch.no_grad()
    def side_info(self, content, pos, valid, ref):
        """See the class docstring (extras). content: gathered taps before the position code.
        Only ever used as sort keys, so no autograd."""
        if ref is None:
            raise ValueError('TokenBank extras need the keyframe index')
        is_ref = torch.arange(content.shape[1], device=content.device) // self.k ** 2 == ref
        c = content.float()
        vr = (valid & is_ref).float()[..., None]                           # the keyframe's valid taps
        mean = (c * vr).sum(1, keepdim=True) / vr.sum(1, keepdim=True).clamp_min(1)
        return {'pos': pos.detach().float(), 'cons': -(c - mean).pow(2).mean(-1), 'is_ref': is_ref}


class TokenRefine(nn.Module):
    """Keyframe-conditioned token update: x <- x + W2 gelu(W1 LN(x) + V LN(s)).

    TokenBank's tokens come from a shallow per-frame encoder and are computed once,
    while the keyframe state they are pooled into deepens through the body. A tap token
    and the keyframe state at its pixel p sit on the same (reference) grid, so the whole
    keyframe context at p -- the body's receptive field, not the encoder's -- can
    re-express every tap of p without any warping. W2 is zero-init (identity at init).
    Invalid taps stay 0. KGTS's token-side cache must be recomputed afterwards.
    """
    def __init__(self, d, ds, hidden=None):
        super().__init__()
        hidden = hidden or 2 * d
        self.norm_x, self.norm_s = nn.LayerNorm(d), nn.LayerNorm(ds)
        self.W1 = nn.Linear(d, hidden)
        self.V = nn.Linear(ds, hidden, bias=False)
        self.W2 = nn.Linear(hidden, d)
        nn.init.zeros_(self.W2.weight); nn.init.zeros_(self.W2.bias)

    def forward(self, x, valid, s):
        """x: (P, L, d) tokens, valid: (P, L), s: (P, ds) keyframe state."""
        h = self.W1(self.norm_x(x)) + self.V(self.norm_s(s)).unsqueeze(1)
        return (x + self.W2(F.gelu(h))) * valid[..., None]


class GroupLayerNorm(nn.Module):
    """LayerNorm over each of `groups` equal slices of the last dim, then one affine over all of it."""
    def __init__(self, groups, dim, eps=1e-5):
        super().__init__()
        self.groups, self.eps = groups, eps
        self.weight, self.bias = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        g = x.view(*x.shape[:-1], self.groups, -1)
        return F.layer_norm(g, g.shape[-1:], eps=self.eps).view_as(x) * self.weight + self.bias

def ref_scan(u, delta, A, B, C, delta_bias=None, delta_softplus=True):
    """Pure-torch selective scan (S6). Slow; this is the ground truth.
    u, delta: (P, D, L)   A: (D, n), negative   B: (P, G, n, L)   C: (D, n)   delta_bias: (D,) or None
    Channel c reads B's group c // (D // G) -- mamba_ssm's grouped-B layout.
    Returns y: (P, D, L). Discretization matches mamba_ssm: A_bar = exp(dt*A), B_bar*u = dt*B*u."""
    dt = delta if delta_bias is None else delta + delta_bias[:, None]
    dt = F.softplus(dt) if delta_softplus else dt                         # (P, D, L)
    B = B.repeat_interleave(u.shape[1] // B.shape[1], dim=1)               # (P, D, n, L)
    h = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])                    # (P, D, n)
    ys = []
    for t in range(u.shape[-1]):
        A_bar = torch.exp(dt[:, :, t, None] * A)                           # (P, D, 1)*(D, n) -> (P, D, n)
        Bu = (dt[:, :, t] * u[:, :, t])[:, :, None] * B[..., t]            # (P, D, 1)*(P, D, n) -> (P, D, n)
        h = A_bar * h + Bu
        ys.append((h * C).sum(-1))                                         # (P, D)
    return torch.stack(ys, -1)

class TapNormSoftplus(torch.autograd.Function):
    """dt = softplus(z) / sum over taps of softplus(z), 0 on invalid taps. z, valid: (P, L, D), (P, L, 1).

    Plain autograd would save two fp32 (P, L, D) tensors per call (softplus's input and the
    numerator of the division): ~0.5 GB per KGTS call at M1's batch 4. This saves only the
    output, in out_dtype, and the (P, 1, D) sums, and rebuilds softplus(z) = out * S and
    softplus'(z) = sigmoid(z) = 1 - exp(-softplus(z)) from them."""
    @staticmethod
    def forward(ctx, z, valid, out_dtype):
        cd = torch.promote_types(z.dtype, torch.float32)                  # at least fp32
        sp = F.softplus(z.to(cd)).masked_fill(~valid, 0.0)
        S = sp.sum(1, keepdim=True).clamp_min(1e-6)
        out = (sp / S).to(out_dtype)
        ctx.save_for_backward(out, S, valid)
        ctx.z_dtype = z.dtype
        return out

    @staticmethod
    def backward(ctx, g):
        out, S, valid = ctx.saved_tensors
        o, g = out.to(S.dtype), g.to(S.dtype)
        g_sp = (g - (g * o).sum(1, keepdim=True)) / S                     # through the normalisation
        g_z = g_sp * -torch.expm1(-o * S)                                  # * sigmoid(z)
        return g_z.masked_fill(~valid, 0.0).to(ctx.z_dtype), None, None

def scan(u, delta, A, B, C, delta_bias=None, delta_softplus=True):
    if u.is_cuda:
        if selective_scan_fn is None:
            raise RuntimeError("mamba_ssm not installed; ref_scan is for tests only")
        return selective_scan_fn(u, delta, A, B, C, delta_bias=delta_bias, delta_softplus=delta_softplus)
    return ref_scan(u.float(), delta.float(), A, B.float(), C, delta_bias, delta_softplus).to(u.dtype)

@ARCH_REGISTRY.register()
class KGTS(nn.Module):
    """
    Keyframe-Gated Tap Scan (KGTS) Class

    Consumes TokenBank output: x [P, L, d], valid [P, L]
    where P = (b h w) and L = (n k).

    One instance is called after each ASSG in MambaIRv2.
    Recurrent class - same weights.

    Capacity knobs (expand / heads / n default to the original single-head,
    d-wide design; norm_s / out_gate / out_norm / a_max / affinity default to
    the corrected one. norm_s=out_gate=out_norm=affinity=False, a_max=n, with
    TokenBank pos_freqs=(), norm=False, mark_ref=pin_ref=False, reproduce the
    original KGTS exactly, forward and backward):

    expand: scan width d_inner = expand * d. Each scan channel is its own
        pooling over the L taps (its own dt, so its own admission/decay
        profile) of its own value channel, and the injection is W_c applied to
        the 2*d_inner final states. So d_inner -- not d -- is the number of
        distinct things one injection can deliver to the keyframe. expand > 1
        adds a value projection W_u: d -> d_inner and grows the scan without
        growing the cached token bank (P*L*d), which is the memory term.
    heads: independent key spaces. B is (heads, n) instead of (n): channel
        block h uses key B_h = FiLM_h(W_B,h x), and (with affinity) its own
        admission key W_k,h x. What a tap writes is weighted by an n-dim
        bilinear form (C . gamma(s) . W_B x); whether it is admitted at all by
        <W_q,h s, W_k,h x>. One head means every channel ranks taps by the same
        criteria; several heads can specialise (e.g. one on occlusion/brightness
        mismatch, one on sub-pixel phase). This is mamba_ssm's grouped-B
        layout, so it costs no extra kernel launches. Needs d_inner % heads == 0.
    n: state size per channel, i.e. the dimension of that key/query space.
    norm_s: LayerNorm on the keyframe state before it conditions anything.
        The weights are tied across every call site, but the residual stream
        they read grows through the ASSBs; without it the same U_delta /
        gamma / beta / W_g see a different input scale at each call, and W_g's
        sigmoid saturates on the deep ones.
    out_gate: keyframe-conditioned output gate y * silu(W_z s) on the pooled
        states before W_c -- Mamba's z-branch, driven by the keyframe instead
        of the tokens, so each call can choose which pooled channels it reads.
    out_norm: LayerNorm on the pooled end states before the gate and W_c
        (Mamba2 norms here too). With the original tokens (~0.2 RMS, see
        TokenBank.norm) they were ~1e-3 RMS at init against a keyframe state of
        ~1, and Adam moves each W_c weight by ~lr per step whatever the
        gradient scale, so the injection stayed negligible for thousands of
        steps while the body trained as single-image SR. With TokenBank.norm
        they start at ~1 RMS anyway; out_norm now pins that scale through
        training. W_c stays zero-init: an exact identity at step 0.
    a_max: |A| of the fastest state at init, S4D-real shape A_i = -a_max*i/n
        (a_max = n is the original A_i = -i). The taps are a SET -- frame order
        in a burst carries nothing -- but only the last state is read, so tap t
        is weighted by exp(A * sum_{s>t} dt_s). Mamba's A = -(1..n) is tuned
        for long sequences; over L = 56 taps at dt ~ 0.05 it forgets all but
        the last few, and at init the fwd scan put 81% of its sensitivity on
        the last frame, the bwd 80% on the first: 3.0 effective frames of 14,
        5% of it on frames 4-10. a_max = 0.5 alone gives 13.5 of 14; with the
        other corrections 12.8 (39% on frames 4-10) -- unit-scale tokens
        spread dt wider. A stays learnable, so recency is still available if
        the loss wants it.
    affinity: per-head keyframe-token term <W_q s, W_k x_t> / sqrt(n) in the
        dt logit. dt = softplus(W_delta x_t + U_delta s + b) cannot let the
        keyframe choose taps: U_delta s is the same for every tap of a pixel,
        so it shifts all their logits together and softplus is monotone -- the
        tap ranking by dt (admission AND forgetting) is set by the tokens alone
        (measured: 99.99% of ranks unchanged under a different keyframe). The
        keyframe only reached the taps through B, which scales what a tap
        writes but not what it erases. W_q is zero-init: exact no-op at step 0.
        W_k x is cached with the other token terms.
    dt_norm: normalise dt over the taps of each (pixel, channel) so they sum
        to 1: the scan then integrates over unit time, every state's decay
        across the whole sequence is at most exp(-a_max) however dt grows in
        training (a_max alone only fixes the init), and admission becomes
        competitive -- the end state is a dt-weighted average over the taps
        instead of a sum, so its scale no longer depends on the dt level or on
        how many taps are valid. Invalid taps get dt = 0 exactly.
    roles: None, or one role per head -- gives the heads different JOBS instead of
        identical inits (measured at M1 init: all 16 head x direction pairs read
        40-47 of the 56 taps with 15-22% of their weight on the last 8, i.e. they are
        interchangeable integrators). A fast decay alone cannot make a head selective:
        in frame order it only makes it recency-biased (a_max 8 puts up to 55% on
        the last 8 taps and quadruples the change under a frame permutation). So a
        selector head also gets an ORDER in which recency means relevance: its taps
        are scanned in ascending order of a per-tap key, so a fast state keeps a
        soft top-k of the key, a slow one still reads everything, and the scan
        becomes exactly invariant to the order of the burst's frames.
        'int' (integrate / inject): frame order, A spectrum from a_max -- the
            accumulator the scan already was (denoising, gathering every sample).
        'geo' (geometric selector): ascending -|sample pos - block centre|^2, so the
            samples nearest the pixel's HR block come last -- a learned narrow
            interpolation kernel over the whole burst.
        'con' (consistency selector): ascending TokenBank's `cons`, the tap's match to
            the keyframe's own content at p, so the most consistent taps come last --
            robust fusion that can reject occluded / misaligned / moving content.
        Selectors use a_max_sel (fast states = top few taps, slow = top dozens) and
        never read the keyframe's own taps (the body has them; ranked by consistency
        they would trivially win). The backward copy of a selector scans the reverse
        order, so its fast states summarise the LEAST relevant taps (an outlier
        readout). Orders are fixed per forward (keys are token-side, cached); the
        keyframe still re-weights taps per call through dt and the affinity.
        Needs TokenBank(extras=True) side information; costs one gather per cached
        tensor per forward, nothing per call.
    a_max_sel: a_max of the selector heads' A spectrum.
    out_norm: True (LayerNorm over all 2*d_inner pooled dims), False, or 'group':
        one LayerNorm per (direction, head), so heads with different jobs (and
        different state scales) are normalised separately before W_c mixes them.
    n_calls, depth_embed, untie_out: the weights are tied across the n_calls
        injection points although the keyframe state is a different representation
        at each depth. depth_embed adds a learned zero-init embedding of the call
        index to the normalised keyframe state (every keyframe-side projection sees
        which call it is serving); untie_out gives every call its own W_c (zero-init)
        and W_g, so each depth writes into its own subspace of the residual stream.
        Token-side weights (and so the cache) stay shared.

    Both scan directions run as ONE selective_scan call: the backward copy is
    stacked on the channel axis (2*d_inner channels, 2*heads B-groups), as in
    VMamba's cross-scan.
    """
    ROLES = ('int', 'geo', 'con')
    # 'geo' key anchor, in tap-position units (0 = where the keyframe's R sample of p sits, HR 8p).
    # The x8 HR block of p spans [p, p + 7/8], centre 7/16; a tap's RGGB quad spans [0, 1/2]
    # from its R sample, centroid 1/4. Quad centroid on block centre: pos = 7/16 - 1/4.
    ANCHOR = 3 / 16

    def __init__(self, ds, d=16, n=8, expand=1, heads=1, norm_s=True, out_gate=True, out_norm=True,
                 dt_min=1e-2, dt_max=1e-1, a_max=0.5, affinity=True, dt_norm=False,
                 roles=None, a_max_sel=8.0, n_calls=1, depth_embed=False, untie_out=False):
        super().__init__()
        di = expand * d
        if di % heads:
            raise ValueError(f'd_inner = expand*d = {di} must be divisible by heads = {heads}')
        if roles is not None:
            roles = list(roles)
            if len(roles) != heads or any(r not in self.ROLES for r in roles):
                raise ValueError(f'kgts.roles must list one of {self.ROLES} per head ({heads}), got {roles}')
            if all(r == 'int' for r in roles):
                roles = None                                               # nothing to reorder
        if out_norm not in (True, False, 'group'):
            raise ValueError(f"kgts.out_norm must be true, false or 'group', got {out_norm!r}")
        self.d, self.di, self.n, self.heads = d, di, n, heads
        self.roles, self.n_calls, self.untie_out = roles, n_calls, untie_out

        self.norm_s = nn.LayerNorm(ds) if norm_s else nn.Identity()

        # value: u = W_u x. Identity at expand=1 (the original design: u = x).
        self.W_u = nn.Linear(d, di, bias=False) if di != d else nn.Identity()

        # admission gate: dt = softplus(W_delta x + U_delta s + <W_q s, W_k x> + b)
        self.W_delta = nn.Linear(d, di, bias=False)     # token side, cached
        self.U_delta = nn.Linear(ds, di, bias=False)    # keyframe side, per ASSG
        dt = torch.exp(torch.rand(di) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.delta_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus -> dt at init

        # input matrix, per head: B = (W_B x) * gamma(s) + beta(s)   (FiLM, identity at init)
        self.W_B = nn.Linear(d, heads * n, bias=False)
        self.gamma, self.beta = nn.Linear(ds, heads * n), nn.Linear(ds, heads * n)
        nn.init.zeros_(self.gamma.weight); nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

        # keyframe-token affinity in the dt logit, per head: <W_q s, W_k x> (zero-init W_q -> off at init)
        self.W_k = nn.Linear(d, heads * n, bias=False) if affinity else None      # token side, cached
        self.W_q = nn.Linear(ds, heads * n, bias=False) if affinity else None     # keyframe side, per ASSG
        if affinity:
            nn.init.zeros_(self.W_q.weight)
        self.dt_norm = dt_norm

        # A = -exp(A_log) stays negative; S4D-real shape, scaled so the set of taps is
        # not read with a recency bias (see a_max). Index 0 = fwd, 1 = bwd.
        if roles is None:
            A = (a_max / n) * torch.arange(1, n + 1, dtype=torch.float32).repeat(di, 1)   # (di, n)
        else:                                                              # per head: its role's spectrum
            A = torch.cat([((a_max if r == 'int' else a_max_sel) / n)
                           * torch.arange(1, n + 1, dtype=torch.float32).repeat(di // heads, 1) for r in roles])
        self.A_log = nn.Parameter(torch.log(A)[None].repeat(2, 1, 1))      # (2, di, n)
        self.C = nn.Parameter(torch.randn(2, di, n) / math.sqrt(n))        # fixed readout; s enters via gamma

        # unit-scale pooled states, then the keyframe-conditioned output gate
        self.out_norm = (GroupLayerNorm(2 * heads, 2 * di) if out_norm == 'group' else
                         nn.LayerNorm(2 * di) if out_norm else nn.Identity())
        self.W_z = nn.Linear(ds, 2 * di) if out_gate else None

        # injection: s <- s + g * W_c [y_fwd; y_bwd],  g = sigmoid(W_g [s; ds])
        self.W_c = nn.Linear(2 * di, ds)
        nn.init.zeros_(self.W_c.weight); nn.init.zeros_(self.W_c.bias)     # exact identity at init
        self.W_g = nn.Linear(2 * ds, ds)

        # per-call conditioning (built last: flags off leave the parameter RNG stream untouched)
        self.depth_embed = nn.Parameter(torch.zeros(n_calls, ds)) if depth_embed else None
        if untie_out:                        # call 0 keeps W_c / W_g, calls 1.. get their own
            self.W_cs = nn.ModuleList(nn.Linear(2 * di, ds) for _ in range(n_calls - 1))
            self.W_gs = nn.ModuleList(nn.Linear(2 * ds, ds) for _ in range(n_calls - 1))
            for W in self.W_cs:
                nn.init.zeros_(W.weight); nn.init.zeros_(W.bias)

    def precompute(self, x, valid, extras=None):
        """Token-only terms, once per forward. Invalid taps get dt ~ 0: no decay AND no admission.

        u is built here in its final scan layout -- (P, 2*d_inner, L), backward
        copy stacked on channels -- since it is the same tensor for every call.
        With roles, every per-tap tensor is put in its head's scan order here (the
        keys are token-side), and valid becomes per head, (P, L, heads).
        """
        u = self.W_u(x)                                                    # (P, L, di)
        kx = self.W_k(x) if self.W_k is not None else None                 # (P, L, heads*n)
        dx, bx, valid = self.W_delta(x), self.W_B(x), valid.bool()
        if self.roles is not None:
            if extras is None:
                raise ValueError('kgts.roles needs TokenBank side information (bank(..., extras=True))')
            perm, valid = self.orders(valid, extras)
            u, dx, bx = self.take(u, perm), self.take(dx, perm), self.take(bx, perm)
            kx = self.take(kx, perm) if kx is not None else None
        u = u.transpose(1, 2)                                              # (P, di, L)
        u = torch.cat([u, u.flip(-1)], 1).contiguous()                     # (P, 2di, L)
        return u, dx, bx, kx, valid

    @torch.no_grad()
    def orders(self, valid, extras):
        """Per-head scan order (P, L, heads), ascending key, and the per-head valid mask in that order."""
        P, L = valid.shape
        key = torch.zeros(P, L, self.heads, device=valid.device)          # 'int': all ties -> frame order
        sel = torch.tensor([r != 'int' for r in self.roles], device=valid.device)
        for h, r in enumerate(self.roles):
            if r == 'geo':
                key[..., h] = -(extras['pos'].float() - self.ANCHOR).pow(2).sum(-1)
            elif r == 'con':
                key[..., h] = extras['cons'].float()
        # selectors skip the keyframe's taps; their invalid taps go first (dt = 0 there anyway)
        vh = valid.unsqueeze(-1) & ~(sel & extras['is_ref'].to(valid.device)[:, None])
        key = key.masked_fill(sel & ~vh, float('-inf'))
        perm = key.argsort(dim=1, stable=True)
        return perm, vh.gather(1, perm)

    def take(self, t, perm):
        """Reorder the taps of a (P, L, heads*m) tensor, head block h by perm[..., h]."""
        P, L, H = perm.shape
        t = t.view(P, L, H, -1)
        return t.gather(1, perm.unsqueeze(-1).expand(-1, -1, -1, t.shape[-1])).view(P, L, -1)

    def conditioned(self, s, cache):
        """s must already be normalised (norm_s). Returns the scan inputs; delta is the
        pre-softplus logit, or with dt_norm the final dt."""
        u, dx, bx, kx, valid = cache
        P, L = dx.shape[:2]
        delta = dx + self.U_delta(s).unsqueeze(1)                          # (P, L, di)
        if kx is not None:
            # the only term that lets the keyframe re-rank taps; channel block h reads head h
            q = self.W_q(s).view(P, 1, self.heads, self.n)
            a = (kx.view(P, L, self.heads, self.n) * q).sum(-1) / math.sqrt(self.n)   # (P, L, heads)
            delta = (delta.view(P, L, self.heads, -1) + a.unsqueeze(-1)).view(P, L, self.di)
        if valid.dim() == 3:                                               # per-head orders (roles)
            vmask = valid.unsqueeze(-1).expand(-1, -1, -1, self.di // self.heads).reshape(P, L, self.di)
        else:
            vmask = valid.unsqueeze(-1)
        if self.dt_norm:                                                   # sum over taps = 1
            delta = TapNormSoftplus.apply(delta + self.delta_bias, vmask, u.dtype)
        else:
            delta = delta.masked_fill(~vmask, -30.0)                       # mask AFTER the sum
        B = bx * self.gamma(s).unsqueeze(1) + self.beta(s).unsqueeze(1)    # (P, L, heads*n)

        delta = delta.transpose(1, 2)                                      # (P, di, L)
        delta = torch.cat([delta, delta.flip(-1)], 1)                      # (P, 2di, L)
        B = B.view(P, L, self.heads, self.n).permute(0, 2, 3, 1)           # (P, heads, n, L)
        B = torch.cat([B, B.flip(-1)], 1)                                  # (P, 2heads, n, L)
        # mamba_ssm wants delta and a variable B in u's dtype
        return u, delta.to(u.dtype).contiguous(), B.to(u.dtype).contiguous()

    def pooled(self, sn, cache):
        """End states of both scans, (P, 2di) = [fwd; bwd], before out_norm. sn = norm_s(s)."""
        u, delta, B = self.conditioned(sn, cache)
        A = -torch.exp(self.A_log.float()).flatten(0, 1)                   # (2di, n)
        bias = None if self.dt_norm else self.delta_bias.float().repeat(2)  # dt_norm: delta is dt already
        y = scan(u, delta, A, B, self.C.float().flatten(0, 1), bias, delta_softplus=not self.dt_norm)
        # a copy, not a view: out_norm saves its input for backward, and a view would pin the
        # whole (P, 2di, L) scan output (L = 56x the end state) until then, at every call
        return y[..., -1].contiguous()

    def forward(self, s, cache, idx=0, return_pooled=False):
        """idx: which injection point this call serves (depth_embed / untie_out).
        return_pooled: also return the gated pooled states y (P, 2di) that W_c reads."""
        sn = self.norm_s(s)
        if self.depth_embed is not None:
            sn = sn + self.depth_embed[idx]
        y = self.out_norm(self.pooled(sn, cache))                          # (P, 2di)
        if self.W_z is not None:
            y = y * F.silu(self.W_z(sn))
        W_c, W_g = (self.W_cs[idx - 1], self.W_gs[idx - 1]) if self.untie_out and idx else (self.W_c, self.W_g)
        d_s = W_c(y)                                                       # (P, ds)
        g = torch.sigmoid(W_g(torch.cat([sn, d_s], -1)))
        return (s + g * d_s, y) if return_pooled else s + g * d_s
