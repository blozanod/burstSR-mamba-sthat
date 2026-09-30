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

def tap_gather(feat, flow, k=2):
    """
    x: [B, N, C, h, w]
    flow: [B, N, 2, h, w]

    Returns:
    taps: [B, N, k*k, C, h, w]
    pos: [B, N, k*k, 2, h, w]   fp32
    valid: [B, N, k*k, h, w]

    Coordinates are always fp32. Under bf16 autocast the flow arrives as bf16,
    whose spacing is 0.25 px between 32 and 64 (0.5 px up to 128): doing the
    xs - flow arithmetic in bf16 would quantise the sub-pixel offsets that
    `pos` exists to carry to a quarter of a packed pixel -- coarser than the
    1/8 px an x8 model has to resolve.
    """
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
    for oy in offs:
        for ox in offs:
            tx, ty = (nx + ox).long(), (ny + oy).long()

            # mask for valid
            ok = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
            valid.append(ok)

            # gather pixels
            idx = (ty.clamp(0, h - 1) * w + tx.clamp(0, w - 1)).view(B * N, 1, h * w).expand(-1, C, -1)
            taps.append(feat.gather(2, idx).view(B * N, C, h, w) * ok.unsqueeze(1))

            # position
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

    Tokenizes the aligned taps once, used for every CFQ

    Out:
    x: [P, L, d]
    valid: [P, L]
    """
    def __init__(self, c, d, k=2, pos_freqs=(0.5, 1, 2, 4), norm=True, mark_ref=True, pin_ref=True):
        super().__init__()
        self.pin_ref = pin_ref
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

    def forward(self, feats, flow, ref=None):
        B, N, C, H, W = feats.shape

        feats = rearrange(feats, 'b n c h w -> (b n) c h w')
        feats = self.proj(feats)
        # autocast runs LN in fp32: back to the incoming dtype, or the fp32 tokens would be
        # re-cast to bf16 by every Linear in KGTS.precompute, each copy saved for backward
        feats = self.norm(feats.permute(0, 2, 3, 1)).to(feats.dtype)          # (bn, h, w, d)
        feats = rearrange(feats, '(b n) h w c -> b n c h w', b=B)

        if self.pin_ref and ref is not None:
            flow = flow * (torch.arange(N, device=flow.device) != ref).view(1, N, 1, 1, 1)
        taps, pos, valid = tap_gather(feats, flow, self.k)

        x = rearrange(taps, 'b n k c h w -> (b h w) (n k) c')
        pos = rearrange(pos, 'b n k c h w -> (b h w) (n k) c') # c = 2
        valid = rearrange(valid, 'b n k h w -> (b h w) (n k)')

        pos = self.pos_encode(self.fourier(pos))

        x = x + pos
        if self.ref_embed is not None and ref is not None:
            is_ref = torch.arange(x.shape[1], device=x.device) // self.k ** 2 == ref
            x = x + is_ref.to(x.dtype)[:, None] * self.ref_embed.to(x.dtype)   # no fp32 promotion
        x = x * valid[..., None]

        return x, valid

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

    Both scan directions run as ONE selective_scan call: the backward copy is
    stacked on the channel axis (2*d_inner channels, 2*heads B-groups), as in
    VMamba's cross-scan.
    """
    def __init__(self, ds, d=16, n=8, expand=1, heads=1, norm_s=True, out_gate=True, out_norm=True,
                 dt_min=1e-2, dt_max=1e-1, a_max=0.5, affinity=True, dt_norm=False):
        super().__init__()
        di = expand * d
        if di % heads:
            raise ValueError(f'd_inner = expand*d = {di} must be divisible by heads = {heads}')
        self.d, self.di, self.n, self.heads = d, di, n, heads

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
        A = (a_max / n) * torch.arange(1, n + 1, dtype=torch.float32).repeat(di, 1)   # (di, n)
        self.A_log = nn.Parameter(torch.log(A)[None].repeat(2, 1, 1))      # (2, di, n)
        self.C = nn.Parameter(torch.randn(2, di, n) / math.sqrt(n))        # fixed readout; s enters via gamma

        # unit-scale pooled states, then the keyframe-conditioned output gate
        self.out_norm = nn.LayerNorm(2 * di) if out_norm else nn.Identity()
        self.W_z = nn.Linear(ds, 2 * di) if out_gate else None

        # injection: s <- s + g * W_c [y_fwd; y_bwd],  g = sigmoid(W_g [s; ds])
        self.W_c = nn.Linear(2 * di, ds)
        nn.init.zeros_(self.W_c.weight); nn.init.zeros_(self.W_c.bias)     # exact identity at init
        self.W_g = nn.Linear(2 * ds, ds)

    def precompute(self, x, valid):
        """Token-only terms, once per forward. Invalid taps get dt ~ 0: no decay AND no admission.

        u is built here in its final scan layout -- (P, 2*d_inner, L), backward
        copy stacked on channels -- since it is the same tensor for every call.
        """
        u = self.W_u(x).transpose(1, 2)                                    # (P, di, L)
        u = torch.cat([u, u.flip(-1)], 1).contiguous()                     # (P, 2di, L)
        kx = self.W_k(x) if self.W_k is not None else None                 # (P, L, heads*n)
        return u, self.W_delta(x), self.W_B(x), kx, valid.bool()

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
        if self.dt_norm:                                                   # sum over taps = 1
            delta = TapNormSoftplus.apply(delta + self.delta_bias, valid.unsqueeze(-1), u.dtype)
        else:
            delta = delta.masked_fill(~valid.unsqueeze(-1), -30.0)         # mask AFTER the sum
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
        return y[..., -1]

    def forward(self, s, cache):
        sn = self.norm_s(s)
        y = self.out_norm(self.pooled(sn, cache))                          # (P, 2di)
        if self.W_z is not None:
            y = y * F.silu(self.W_z(sn))
        d_s = self.W_c(y)                                                  # (P, ds)
        g = torch.sigmoid(self.W_g(torch.cat([sn, d_s], -1)))
        return s + g * d_s
