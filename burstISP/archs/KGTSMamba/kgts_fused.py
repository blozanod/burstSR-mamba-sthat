"""Fused KGTS end state (Triton): what KGTS.pooled computes, without running a scan.

KGTS reads only the LAST state of its two scans, over L = N*k*k taps per pixel. Through
mamba_ssm's selective_scan_fn that is P*2*d_inner independent sequences of length ~56
(2.4M per call at M1's batch 4): one CUDA block each, most of it idle, every output step
written although one is read, and a grouped-B backward that reduces with atomics. At M1
it was ~45% of a training step (profile: 34 ms fwd + 80 ms bwd per call, x7 calls).

Here one program owns a block of pixels x one head (its d_inner/heads channels and n
states, all in registers) and loops over the taps. Per call it reads the token-side cache
(W_delta x, W_u x, W_B x, W_k x) and the per-pixel keyframe terms (U_delta s, W_q s,
gamma(s), beta(s)) and builds the scan inputs on the fly -- no (P, L, *) tensor is
materialised per call, and the backward direction reads the same tensors in reverse (no
flip / cat copies). Math, per pixel p, channel c of head h, tap t (the scan's own):

    logit_t = [valid_t,h] (W_delta x + U_delta s + <W_k x, W_q s>/sqrt(n)) + b    (-30 + b if not)
    dt_t    = softplus(logit_t),   w_t = dt_t * u_t,   B_t = (W_B x) * gamma(s) + beta(s)
    fwd:  h <- exp(dt_t A0) h + w_t B_t  over t = 0..L-1,   y_f = C0 . h_L
    bwd:  y_b = sum_t C1 . exp(A1 sum_{r<t} dt_r) w_t B_t    (the reversed scan's end state)

which is KGTS.conditioned + selective_scan's end state exactly (inputs are read in their
dtype and computed in fp32; the old path rounded delta and B to the token dtype first).
The backward pass recomputes everything from the inputs (nothing but the inputs is saved)
in two passes over the taps, and reduces the parameter gradients (A, C, delta bias) per
pixel block into a buffer summed afterwards: deterministic, no atomics.

Checked against KGTS's reference path (conditioned + ref_scan), values and gradients, by
analysis/kgts_sanity.py's scan stage (the config's kgts.fused).
"""
import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                      # CPU-only installs: KGTS falls back to the scan path
    triton = None

# pixels per program and warps. The backward holds ~8 (pixels, d_inner/heads, n) fp32 tiles.
BLOCK_P_FWD, WARPS_FWD = 16, 8
BLOCK_P_BWD, WARPS_BWD = 8, 8


if triton is not None:
    @triton.jit
    def _softplus(x):
        # mamba_ssm's: log1p(exp(x)) up to 20, x above; exp(x) where log1p would round to 0
        return tl.where(x > 20.0, x, tl.where(x < -15.0, tl.exp(x), tl.log(1.0 + tl.exp(x))))

    @triton.jit
    def _tap(t, p64, pm, h, ch, n, pd_m, pn_m, DX, U, BX, KX, VALID, ud, bias, gam, bet, q, L, scale,
             DI: tl.constexpr, H: tl.constexpr, N: tl.constexpr, HAS_AFF: tl.constexpr):
        """Tap t of a (pixel block, head): row index, u, valid, W_B x, W_k x, logit, dt, B."""
        row = p64 * L + t
        dx = tl.load(DX + row[:, None] * DI + ch[None, :], mask=pd_m, other=0.).to(tl.float32)
        u = tl.load(U + row[:, None] * DI + ch[None, :], mask=pd_m, other=0.).to(tl.float32)
        v = tl.load(VALID + row * H + h, mask=pm, other=0) != 0
        hn = row[:, None] * (H * N) + h * N + n[None, :]
        bx = tl.load(BX + hn, mask=pn_m, other=0.).to(tl.float32)
        z = dx + ud
        kx = tl.zeros_like(bx)
        if HAS_AFF:
            kx = tl.load(KX + hn, mask=pn_m, other=0.).to(tl.float32)
            z = z + (tl.sum(kx * q, axis=1) * scale)[:, None]
        logit = tl.where(v[:, None], z, -30.0) + bias[None, :]
        return row, u, v, bx, kx, logit, _softplus(logit), bx * gam + bet

    @triton.jit
    def _consts(pid, h, P, UD, BIAS, GAM, BET, Q, A, C,
                DI: tl.constexpr, H: tl.constexpr, DH: tl.constexpr, N: tl.constexpr, HAS_AFF: tl.constexpr,
                BLOCK_P: tl.constexpr, DH_P: tl.constexpr, N_P: tl.constexpr):
        """Index vectors, masks and the per-program operands."""
        p = pid * BLOCK_P + tl.arange(0, BLOCK_P)
        d = tl.arange(0, DH_P)
        n = tl.arange(0, N_P)
        pm, dm, nm = p < P, d < DH, n < N
        ch = h * DH + d
        p64 = p.to(tl.int64)
        pd_m = pm[:, None] & dm[None, :]
        pn_m = pm[:, None] & nm[None, :]
        dn_m = dm[:, None] & nm[None, :]
        ud = tl.load(UD + p64[:, None] * DI + ch[None, :], mask=pd_m, other=0.).to(tl.float32)
        bias = tl.load(BIAS + ch, mask=dm, other=0.).to(tl.float32)
        hn = p64[:, None] * (H * N) + h * N + n[None, :]
        gam = tl.load(GAM + hn, mask=pn_m, other=0.).to(tl.float32)
        bet = tl.load(BET + hn, mask=pn_m, other=0.).to(tl.float32)
        q = tl.zeros_like(gam)
        if HAS_AFF:
            q = tl.load(Q + hn, mask=pn_m, other=0.).to(tl.float32)
        dn = ch[:, None] * N + n[None, :]
        a0 = tl.load(A + dn, mask=dn_m, other=0.)[None, :, :]
        a1 = tl.load(A + DI * N + dn, mask=dn_m, other=0.)[None, :, :]
        c0 = tl.load(C + dn, mask=dn_m, other=0.)[None, :, :]
        c1 = tl.load(C + DI * N + dn, mask=dn_m, other=0.)[None, :, :]
        return p64, pm, dm, ch, n, pd_m, pn_m, dn_m, dn, hn, ud, bias, gam, bet, q, a0, a1, c0, c1

    @triton.jit
    def _fwd(DX, U, BX, KX, Q, VALID, UD, BIAS, GAM, BET, A, C, Y, P, L, scale,
             DI: tl.constexpr, H: tl.constexpr, DH: tl.constexpr, N: tl.constexpr, HAS_AFF: tl.constexpr,
             BLOCK_P: tl.constexpr, DH_P: tl.constexpr, N_P: tl.constexpr):
        h = tl.program_id(1)
        p64, pm, dm, ch, n, pd_m, pn_m, dn_m, dn, hn, ud, bias, gam, bet, q, a0, a1, c0, c1 = _consts(
            tl.program_id(0), h, P, UD, BIAS, GAM, BET, Q, A, C, DI, H, DH, N, HAS_AFF, BLOCK_P, DH_P, N_P)
        hf = tl.zeros((BLOCK_P, DH_P, N_P), dtype=tl.float32)   # forward scan state
        yb = tl.zeros((BLOCK_P, DH_P), dtype=tl.float32)        # backward scan's end state, closed form
        S = tl.zeros((BLOCK_P, DH_P), dtype=tl.float32)         # sum of dt over the taps before t
        for t in range(0, L):
            row, u, v, bx, kx, logit, dt, B = _tap(t, p64, pm, h, ch, n, pd_m, pn_m, DX, U, BX, KX, VALID,
                                                   ud, bias, gam, bet, q, L, scale, DI, H, N, HAS_AFF)
            wB = (dt * u)[:, :, None] * B[:, None, :]
            yb += tl.sum(c1 * tl.exp(a1 * S[:, :, None]) * wB, axis=2)
            hf = tl.exp(dt[:, :, None] * a0) * hf + wB
            S += dt
        yo = p64[:, None] * (2 * DI) + ch[None, :]
        tl.store(Y + yo, tl.sum(c0 * hf, axis=2), mask=pd_m)
        tl.store(Y + yo + DI, yb, mask=pd_m)

    @triton.jit
    def _bwd(DX, U, BX, KX, Q, VALID, UD, BIAS, GAM, BET, A, C, GY,
             GDX, GU, GBX, GKX, GQ, GUD, GGAM, GBET, PBIAS, PAC, P, L, scale,
             DI: tl.constexpr, H: tl.constexpr, DH: tl.constexpr, N: tl.constexpr, HAS_AFF: tl.constexpr,
             BLOCK_P: tl.constexpr, DH_P: tl.constexpr, N_P: tl.constexpr):
        pid = tl.program_id(0)
        h = tl.program_id(1)
        p64, pm, dm, ch, n, pd_m, pn_m, dn_m, dn, hn, ud, bias, gam, bet, q, a0, a1, c0, c1 = _consts(
            pid, h, P, UD, BIAS, GAM, BET, Q, A, C, DI, H, DH, N, HAS_AFF, BLOCK_P, DH_P, N_P)
        yo = p64[:, None] * (2 * DI) + ch[None, :]
        gf = tl.load(GY + yo, mask=pd_m, other=0.).to(tl.float32)[:, :, None]
        gb = tl.load(GY + yo + DI, mask=pd_m, other=0.).to(tl.float32)[:, :, None]

        # pass 1: T = sum dt, and G_b = sum_s G^b_s, where G^b_s = dy_b/d(sum_{r<s} dt_r) at tap s
        S = tl.zeros((BLOCK_P, DH_P), dtype=tl.float32)
        Gb = tl.zeros((BLOCK_P, DH_P), dtype=tl.float32)
        for t in range(0, L):
            row, u, v, bx, kx, logit, dt, B = _tap(t, p64, pm, h, ch, n, pd_m, pn_m, DX, U, BX, KX, VALID,
                                                   ud, bias, gam, bet, q, L, scale, DI, H, N, HAS_AFF)
            wB = (dt * u)[:, :, None] * B[:, None, :]
            Gb += tl.sum(c1 * gb * tl.exp(a1 * S[:, :, None]) * a1 * wB, axis=2)
            S += dt
        T = S

        # pass 2: every gradient. kf / kb = dy/dh at tap t of the fwd / bwd scan (C g exp(A tau)).
        S = tl.zeros((BLOCK_P, DH_P), dtype=tl.float32)
        cumGb = tl.zeros((BLOCK_P, DH_P), dtype=tl.float32)
        hf = tl.zeros((BLOCK_P, DH_P, N_P), dtype=tl.float32)
        gud = tl.zeros((BLOCK_P, DH_P), dtype=tl.float32)
        ggam = tl.zeros((BLOCK_P, N_P), dtype=tl.float32)
        gbet = tl.zeros((BLOCK_P, N_P), dtype=tl.float32)
        gq = tl.zeros((BLOCK_P, N_P), dtype=tl.float32)
        gbias = tl.zeros((DH_P,), dtype=tl.float32)
        ga0 = tl.zeros((DH_P, N_P), dtype=tl.float32)
        ga1 = tl.zeros((DH_P, N_P), dtype=tl.float32)
        gc0 = tl.zeros((DH_P, N_P), dtype=tl.float32)
        gc1 = tl.zeros((DH_P, N_P), dtype=tl.float32)
        for t in range(0, L):
            row, u, v, bx, kx, logit, dt, B = _tap(t, p64, pm, h, ch, n, pd_m, pn_m, DX, U, BX, KX, VALID,
                                                   ud, bias, gam, bet, q, L, scale, DI, H, N, HAS_AFF)
            w = dt * u
            wB = w[:, :, None] * B[:, None, :]
            tauf = tl.maximum(T - S - dt, 0.0)                    # sum of dt after t (fwd decay of tap t)
            ef = tl.exp(a0 * tauf[:, :, None])
            eb = tl.exp(a1 * S[:, :, None])                       # sum of dt before t (bwd decay of tap t)
            kf = c0 * gf * ef
            kb = c1 * gb * eb
            k = kf + kb
            kB = tl.sum(k * B[:, None, :], axis=2)
            af = tl.exp(dt[:, :, None] * a0)
            # dt_t: as a weight (w = dt u); fwd: it decays every earlier tap (via the state before t);
            # bwd: it decays every later tap (G_b minus the running sum up to t)
            gdt = u * kB + tl.sum(kf * a0 * af * hf, axis=2)
            hf = af * hf + wB
            cumGb += tl.sum(kb * a1 * wB, axis=2)
            gdt += Gb - cumGb
            glogit = gdt / (1.0 + tl.exp(-logit))                 # softplus' = sigmoid
            glv = tl.where(v[:, None], glogit, 0.0)               # invalid taps: logit = -30 + b
            off = row[:, None] * DI + ch[None, :]
            tl.store(GDX + off, glv, mask=pd_m)
            tl.store(GU + off, dt * kB, mask=pd_m)
            gud += glv
            gbias += tl.sum(glogit, axis=0)                      # masked pixels contribute exactly 0
            gB = tl.sum(k * w[:, :, None], axis=1)               # (pixels, n): summed over the head's channels
            offn = row[:, None] * (H * N) + h * N + n[None, :]
            tl.store(GBX + offn, gB * gam, mask=pn_m)
            ggam += gB * bx
            gbet += gB
            if HAS_AFF:
                ga = tl.sum(glv, axis=1) * scale
                tl.store(GKX + offn, ga[:, None] * q, mask=pn_m)
                gq += ga[:, None] * kx
            gc0 += tl.sum(gf * ef * wB, axis=0)
            gc1 += tl.sum(gb * eb * wB, axis=0)
            ga0 += tl.sum(kf * tauf[:, :, None] * wB, axis=0)
            ga1 += tl.sum(kb * S[:, :, None] * wB, axis=0)
            S += dt

        tl.store(GUD + p64[:, None] * DI + ch[None, :], gud, mask=pd_m)
        tl.store(GGAM + hn, ggam, mask=pn_m)
        tl.store(GBET + hn, gbet, mask=pn_m)
        if HAS_AFF:
            tl.store(GQ + hn, gq, mask=pn_m)
        pb = pid.to(tl.int64)
        tl.store(PBIAS + pb * DI + ch, gbias, mask=dm)
        tl.store(PAC + (pb * 4 + 0) * DI * N + dn, ga0, mask=dn_m)
        tl.store(PAC + (pb * 4 + 1) * DI * N + dn, ga1, mask=dn_m)
        tl.store(PAC + (pb * 4 + 2) * DI * N + dn, gc0, mask=dn_m)
        tl.store(PAC + (pb * 4 + 3) * DI * N + dn, gc1, mask=dn_m)


def _meta(dx, valid, A, kx):
    P, L, DI = dx.shape
    H, N = valid.shape[-1], A.shape[-1]
    DH = DI // H
    return P, L, dict(DI=DI, H=H, DH=DH, N=N, HAS_AFF=kx is not None,
                      DH_P=triton.next_power_of_2(DH), N_P=triton.next_power_of_2(N))


class _EndState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dx, u, bx, kx, q, valid, ud, bias, gam, bet, A, C, scale):
        P, L, meta = _meta(dx, valid, A, kx)
        y = torch.empty(P, 2 * meta['DI'], device=dx.device, dtype=torch.float32)
        kx_, q_ = (kx, q) if kx is not None else (bx, gam)          # never read without affinity
        _fwd[(triton.cdiv(P, BLOCK_P_FWD), meta['H'])](
            dx, u, bx, kx_, q_, valid, ud, bias, gam, bet, A, C, y, P, L, scale,
            BLOCK_P=BLOCK_P_FWD, num_warps=WARPS_FWD, **meta)
        ctx.save_for_backward(dx, u, bx, kx, q, valid, ud, bias, gam, bet, A, C)
        ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, gy):
        dx, u, bx, kx, q, valid, ud, bias, gam, bet, A, C = ctx.saved_tensors
        P, L, meta = _meta(dx, valid, A, kx)
        DI, N = meta['DI'], meta['N']
        f32 = dict(device=dx.device, dtype=torch.float32)
        gdx, gu, gbx = torch.empty_like(dx), torch.empty_like(u), torch.empty_like(bx)
        gud, ggam, gbet = torch.empty(ud.shape, **f32), torch.empty(gam.shape, **f32), torch.empty(bet.shape, **f32)
        has = kx is not None
        gkx, gq = (torch.empty_like(kx), torch.empty(q.shape, **f32)) if has else (None, None)
        nb = triton.cdiv(P, BLOCK_P_BWD)
        pbias, pac = torch.empty(nb, DI, **f32), torch.empty(nb, 4, DI, N, **f32)
        kx_, q_ = (kx, q) if has else (bx, gam)
        _bwd[(nb, meta['H'])](
            dx, u, bx, kx_, q_, valid, ud, bias, gam, bet, A, C, gy.contiguous(),
            gdx, gu, gbx, gkx if has else gbx, gq if has else ggam, gud, ggam, gbet, pbias, pac,
            P, L, ctx.scale, BLOCK_P=BLOCK_P_BWD, num_warps=WARPS_BWD, **meta)
        pac = pac.sum(0)
        return (gdx, gu, gbx, gkx, gq.to(q.dtype) if has else None, None, gud.to(ud.dtype),
                pbias.sum(0).to(bias.dtype), ggam.to(gam.dtype), gbet.to(bet.dtype), pac[:2], pac[2:], None)


def end_state(dx, u, bx, kx, q, valid, ud, bias, gam, bet, A, C):
    """Both scans' end states, (P, 2*d_inner) fp32, [fwd; bwd] -- KGTS.pooled's output.

    dx, u: (P, L, d_inner) W_delta x, W_u x;  bx, kx: (P, L, heads*n) W_B x, W_k x (kx None
    without affinity);  valid: (P, L, heads) int8;  ud: (P, d_inner) U_delta s;  q, gam, bet:
    (P, heads*n) W_q s, gamma(s), beta(s);  bias: (d_inner,) delta bias;  A, C: (2, d_inner, n)
    fp32, A negative. Taps in scan order (KGTS.precompute applies the roles' orders)."""
    if triton is None:
        raise RuntimeError('kgts.fused needs triton')
    c = lambda t: None if t is None else t.contiguous()
    return _EndState.apply(c(dx), c(u), c(bx), c(kx), c(q), valid.to(torch.int8).contiguous(), c(ud),
                           bias.float().contiguous(), c(gam), c(bet), A.float().contiguous(),
                           C.float().contiguous(), 1.0 / math.sqrt(A.shape[-1]))
