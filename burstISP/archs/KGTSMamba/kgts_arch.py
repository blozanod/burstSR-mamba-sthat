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

    Tokenizes the aligned taps once, used for every CFQ

    Out:
    x: [P, L, d]
    valid: [P, L]
    """
    def __init__(self, c, d, k=2):
        super().__init__()
        self.proj = nn.Conv2d(c, d, 1)
        self.pos_encode = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d,d))
        self.k = k

    def forward(self, feats, flow):
        B, N, C, H, W = feats.shape
        
        feats = rearrange(feats, 'b n c h w -> (b n) c h w')
        feats = self.proj(feats)
        feats = rearrange(feats, '(b n) c h w -> b n c h w', b=B)

        taps, pos, valid = tap_gather(feats, flow, self.k)

        x = rearrange(taps, 'b n k c h w -> (b h w) (n k) c')
        pos = rearrange(pos, 'b n k c h w -> (b h w) (n k) c') # c = 2
        valid = rearrange(valid, 'b n k h w -> (b h w) (n k)')

        pos = self.pos_encode(pos)

        x = x + pos
        x = x * valid[..., None]

        return x, valid

def ref_scan(u, delta, A, B, C, delta_bias):
    """Pure-torch selective scan (S6). Slow; this is the ground truth.
    u, delta: (P, D, L)   A: (D, n), negative   B: (P, G, n, L)   C: (D, n)   delta_bias: (D,)
    Channel c reads B's group c // (D // G) -- mamba_ssm's grouped-B layout.
    Returns y: (P, D, L). Discretization matches mamba_ssm: A_bar = exp(dt*A), B_bar*u = dt*B*u."""
    dt = F.softplus(delta + delta_bias[:, None])                          # (P, D, L)
    B = B.repeat_interleave(u.shape[1] // B.shape[1], dim=1)               # (P, D, n, L)
    h = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])                    # (P, D, n)
    ys = []
    for t in range(u.shape[-1]):
        A_bar = torch.exp(dt[:, :, t, None] * A)                           # (P, D, 1)*(D, n) -> (P, D, n)
        Bu = (dt[:, :, t] * u[:, :, t])[:, :, None] * B[..., t]            # (P, D, 1)*(P, D, n) -> (P, D, n)
        h = A_bar * h + Bu
        ys.append((h * C).sum(-1))                                         # (P, D)
    return torch.stack(ys, -1)

def scan(u, delta, A, B, C, delta_bias):
    if u.is_cuda:
        if selective_scan_fn is None:
            raise RuntimeError("mamba_ssm not installed; ref_scan is for tests only")
        return selective_scan_fn(u, delta, A, B, C, delta_bias=delta_bias, delta_softplus=True)
    return ref_scan(u.float(), delta.float(), A, B.float(), C, delta_bias).to(u.dtype)

@ARCH_REGISTRY.register()
class KGTS(nn.Module):
    """
    Keyframe-Gated Tap Scan (KGTS) Class

    Consumes TokenBank output: x [P, L, d], valid [P, L]
    where P = (b h w) and L = (n k).

    One instance is called after each ASSG in MambaIRv2.
    Recurrent class - same weights.

    Capacity knobs (all default to the original single-head, d-wide design
    except norm_s / out_gate / out_norm):

    expand: scan width d_inner = expand * d. Each scan channel is its own
        pooling over the L taps (its own dt, so its own admission/decay
        profile) of its own value channel, and the injection is W_c applied to
        the 2*d_inner final states. So d_inner -- not d -- is the number of
        distinct things one injection can deliver to the keyframe. expand > 1
        adds a value projection W_u: d -> d_inner and grows the scan without
        growing the cached token bank (P*L*d), which is the memory term.
    heads: independent key spaces. B is (heads, n) instead of (n): channel
        block h uses key B_h = FiLM_h(W_B,h x). The keyframe-token match that
        decides which taps a channel admits is an n-dim bilinear form
        (C . gamma(s) . W_B x), so one head means every channel ranks taps by
        the same criterion; several heads can specialise (e.g. one on
        occlusion/brightness mismatch, one on sub-pixel phase). This is
        mamba_ssm's grouped-B layout, so it costs no extra kernel launches.
        Needs d_inner % heads == 0.
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
        (Mamba2 norms here too). At init they are ~1e-3 RMS against a keyframe
        state of ~1 (dt starts at 1e-2..1e-1), and Adam moves each W_c weight
        by ~lr per step whatever the gradient scale, so without it the
        injection stays negligible for thousands of steps while the body
        trains as single-image SR. W_c stays zero-init: still an exact
        identity at step 0, but the first steps already inject O(lr * width).

    Both scan directions run as ONE selective_scan call: the backward copy is
    stacked on the channel axis (2*d_inner channels, 2*heads B-groups), as in
    VMamba's cross-scan.
    """
    def __init__(self, ds, d=16, n=8, expand=1, heads=1, norm_s=True, out_gate=True, out_norm=True,
                 dt_min=1e-2, dt_max=1e-1):
        super().__init__()
        di = expand * d
        if di % heads:
            raise ValueError(f'd_inner = expand*d = {di} must be divisible by heads = {heads}')
        self.d, self.di, self.n, self.heads = d, di, n, heads

        self.norm_s = nn.LayerNorm(ds) if norm_s else nn.Identity()

        # value: u = W_u x. Identity at expand=1 (the original design: u = x).
        self.W_u = nn.Linear(d, di, bias=False) if di != d else nn.Identity()

        # admission gate: dt = softplus(W_delta x + U_delta s + b)
        self.W_delta = nn.Linear(d, di, bias=False)     # token side, cached
        self.U_delta = nn.Linear(ds, di, bias=False)    # keyframe side, per ASSG
        dt = torch.exp(torch.rand(di) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.delta_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus -> dt at init

        # input matrix, per head: B = (W_B x) * gamma(s) + beta(s)   (FiLM, identity at init)
        self.W_B = nn.Linear(d, heads * n, bias=False)
        self.gamma, self.beta = nn.Linear(ds, heads * n), nn.Linear(ds, heads * n)
        nn.init.zeros_(self.gamma.weight); nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

        # A = -exp(A_log) stays negative; S4D-real init. Index 0 = fwd, 1 = bwd.
        A = torch.arange(1, n + 1, dtype=torch.float32).repeat(di, 1)      # (di, n)
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
        return u, self.W_delta(x), self.W_B(x), valid.bool()

    def conditioned(self, s, cache):
        """s must already be normalised (norm_s)."""
        u, dx, bx, valid = cache
        P, L = dx.shape[:2]
        delta = dx + self.U_delta(s).unsqueeze(1)                          # (P, L, di)
        delta = delta.masked_fill(~valid.unsqueeze(-1), -30.0)             # mask AFTER the sum
        B = bx * self.gamma(s).unsqueeze(1) + self.beta(s).unsqueeze(1)    # (P, L, heads*n)

        delta = delta.transpose(1, 2)                                      # (P, di, L)
        delta = torch.cat([delta, delta.flip(-1)], 1)                      # (P, 2di, L)
        B = B.view(P, L, self.heads, self.n).permute(0, 2, 3, 1)           # (P, heads, n, L)
        B = torch.cat([B, B.flip(-1)], 1)                                  # (P, 2heads, n, L)
        # mamba_ssm wants delta and a variable B in u's dtype
        return u, delta.to(u.dtype).contiguous(), B.to(u.dtype).contiguous()

    def forward(self, s, cache):
        sn = self.norm_s(s)
        u, delta, B = self.conditioned(sn, cache)
        A = -torch.exp(self.A_log.float()).flatten(0, 1)                   # (2di, n)
        y = scan(u, delta, A, B, self.C.float().flatten(0, 1), self.delta_bias.float().repeat(2))
        y = self.out_norm(y[..., -1])                                      # (P, 2di): [fwd; bwd] end states
        if self.W_z is not None:
            y = y * F.silu(self.W_z(sn))
        d_s = self.W_c(y)                                                  # (P, ds)
        g = torch.sigmoid(self.W_g(torch.cat([sn, d_s], -1)))
        return s + g * d_s
