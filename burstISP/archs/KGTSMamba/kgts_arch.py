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
    pos: [B, N, k*k, 2, h, w]
    valid: [B, N, k*k, h, w]
    """
    B, N, C, h, w = feat.shape
    feat = feat.reshape(B * N, C, h * w)
    flow = flow.reshape(B * N, 2, h, w)

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
    u, delta: (P, d, L)   A: (d, n), negative   B: (P, n, L)   C: (d, n)   delta_bias: (d,)
    Returns y: (P, d, L). Discretization matches mamba_ssm: A_bar = exp(dt*A), B_bar*u = dt*B*u."""
    dt = F.softplus(delta + delta_bias[:, None])                          # (P, d, L)
    h = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])                    # (P, d, n)
    ys = []
    for t in range(u.shape[-1]):
        A_bar = torch.exp(dt[:, :, t, None] * A)                           # (P, d, 1)*(d, n) -> (P, d, n)
        Bu = (dt[:, :, t] * u[:, :, t])[:, :, None] * B[:, None, :, t]     # (P, d, 1)*(P, 1, n) -> (P, d, n)
        h = A_bar * h + Bu
        ys.append((h * C).sum(-1))                                         # (P, d)
    return torch.stack(ys, -1)

def scan(u, delta, A, B, C, delta_bias):
    if u.is_cuda:
        if selective_scan_fn is None:
            raise RuntimeError("mamba_ssm not installed; ref_scan is for tests only")
        return selective_scan_fn(u, delta, A, B, C, delta_bias=delta_bias, delta_softplus=True)
    return ref_scan(u, delta, A, B, C, delta_bias)

@ARCH_REGISTRY.register()
class KGTS(nn.Module):
    """
    Keyframe-Gated Tap Scan (KGTS) Class

    Consumes TokenBank output: x [P, L, d], valid [P, L]
    where P = (b h w) and L = (n k).

    One instance is called after each ASSG in MambaIRv2.
    Recurrent class - same weights.
    """
    def __init__(self, ds, d=16, n=8, dt_min=1e-2, dt_max=1e-1):
        super().__init__()
        # admission gate: dt = softplus(W_delta x + U_delta s + b)
        self.W_delta = nn.Linear(d, d, bias=False)      # token side, cached
        self.U_delta = nn.Linear(ds, d, bias=False)     # keyframe side, per ASSG
        dt = torch.exp(torch.rand(d) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.delta_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus -> dt at init
 
        # input matrix: B = (W_B x) * gamma(s) + beta(s)   (FiLM, identity at init)
        self.W_B = nn.Linear(d, n, bias=False)
        self.gamma, self.beta = nn.Linear(ds, n), nn.Linear(ds, n)
        nn.init.zeros_(self.gamma.weight); nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)
 
        # A = -exp(A_log) stays negative; S4D-real init. Index 0 = fwd, 1 = bwd.
        A = torch.arange(1, n + 1, dtype=torch.float32).repeat(d, 1)       # (d, n)
        self.A_log = nn.Parameter(torch.log(A)[None].repeat(2, 1, 1))      # (2, d, n)
        self.C = nn.Parameter(torch.randn(2, d, n) / math.sqrt(n))         # fixed readout (not s-conditioned)
 
        # injection: s <- s + g * W_c [y_fwd; y_bwd],  g = sigmoid(W_g [s; ds])
        self.W_c = nn.Linear(2 * d, ds)
        nn.init.zeros_(self.W_c.weight); nn.init.zeros_(self.W_c.bias)     # exact identity at init
        self.W_g = nn.Linear(2 * ds, ds)
 
    def precompute(self, x, valid):
        """Token-only terms, once per forward. Invalid taps get dt ~ 0: no decay AND no admission."""
        return x, self.W_delta(x), self.W_B(x), valid.bool()
 
    def conditioned(self, s, cache):
        x, dx, bx, valid = cache
        delta = dx + self.U_delta(s).unsqueeze(1)                          # (P, L, d)
        delta = delta.masked_fill(~valid.unsqueeze(-1), -30.0)             # mask AFTER the sum
        B = bx * self.gamma(s).unsqueeze(1) + self.beta(s).unsqueeze(1)
        return x, delta, B
 
    def forward(self, s, cache):
        x, delta, B = self.conditioned(s, cache)
        u, delta, B = (t.transpose(1, 2).contiguous() for t in (x, delta, B))     # channels-before-length
        ys = []
        for i in range(2):
            if i == 1:
                u, delta, B = u.flip(-1), delta.flip(-1), B.flip(-1)
            y = scan(u, delta, -torch.exp(self.A_log[i]), B, self.C[i], self.delta_bias)
            ys.append(y[..., -1])                                                    # (P, d) at end of scan
        d_s = self.W_c(torch.cat(ys, -1))                                            # (P, ds)
        g = torch.sigmoid(self.W_g(torch.cat([s, d_s], -1)))
        return s + g * d_s