#!/usr/bin/env python3
"""(2) Per-image colour fit: how many dB are global colour / tone errors?

    python analysis/diagnostics/colour_fit.py --config main/configs/M1_KGTSMamba.yml [--ckpt latest]

For every official SyntheticBurstVal burst, fit GT ~ f(output) on the metric's crop by least
squares and recompute the official PSNR (clamp, 14-bit truncation) of f(output). Nested fits:
  offset   per-channel offset                          3 params   (black level / exposure bias)
  gain     per-channel gain                            3          (exposure, white balance)
  diag     per-channel gain + offset                   6
  affine   3x3 matrix + offset                         12         (the requested fit: channel mixing)
  tone     per output channel, a sum of one 1D curve  144        (16 knots on a sqrt scale: a nonlinear
           per input channel (includes affine)                     tone / clipping bias)
Each is reported in-sample and CROSS-FIT: fit on one half of the image (16 px checkerboard), score
on the other, and vice versa -- what the map is worth on pixels it never saw. A 12- or 144-parameter
fit to ~277k values cannot overfit much, but the cross-fit says so instead of assuming it.
Also: one affine map for the whole set (a fixed, systematic colour error), the mean fitted
matrix / offset, the bias per GT intensity bin, and the gain by noise level.

What it can and cannot show. On SyntheticBurst the GT is the camera-space linear image the RAW
was mosaicked from (rgb2rawburst applies the inverse CCM and gains before generating BOTH), so
there is no colour matrix for the network to invert here; global colour error can only come from
noise-clipping bias in the darks, exposure drift, channel cross-talk from demosaicking, or the
L1 loss's median-vs-mean bias. A global map cannot touch local colour errors (chroma fringes at
edges: error_bands.py's chroma column) or anything spatial.

Reading it: a large jump (>= ~0.2 dB, cross-fit) means global colour / tone errors cost real dB
that no sub-pixel change will recover; ~0 means colour is not where the missing dB are. If the
single global map recovers most of it, the error is systematic (a calibration / bias problem,
cheap to fix); if only per-image maps do, it depends on the image (noise level, content).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

VARIANTS = ('offset', 'gain', 'diag', 'affine', 'tone')
KNOTS = 16
BIAS_EDGES = np.linspace(0, 1, 11) ** 2          # GT intensity bins (sqrt-spaced: dense in the darks)


def hats(v, knots):
    """(n,) values -> (n, K) piecewise-linear hat basis on `knots` (non-uniform, ascending)."""
    k = knots
    v = v.clamp(k[0], k[-1])
    i = torch.bucketize(v, k[1:-1])                                   # segment index 0..K-2
    t = (v - k[i]) / (k[i + 1] - k[i])
    out = torch.zeros(v.shape[0], len(k), device=v.device, dtype=v.dtype)
    out.scatter_(1, i[:, None], (1 - t)[:, None])
    out.scatter_add_(1, (i + 1)[:, None], t[:, None])
    return out


def features(P, variant, knots):
    """P: (n, 3) output pixels -> per output channel c, the design matrix (n, m)."""
    one = torch.ones_like(P[:, :1])
    if variant == 'offset':
        return [one] * 3
    if variant == 'gain':
        return [P[:, c:c + 1] for c in range(3)]
    if variant == 'diag':
        return [torch.cat([P[:, c:c + 1], one], 1) for c in range(3)]
    if variant == 'affine':
        return [torch.cat([P, one], 1)] * 3
    X = torch.cat([hats(P[:, c], knots) for c in range(3)], 1)      # additive per-channel curves
    return [X] * 3


def fit_apply(P, G, variant, knots, fit_mask, apply_mask=None):
    """LS fit of G[:, c] on features(P) over fit_mask rows, applied to every row. 'offset' fits
    G - P. Returns the fitted prediction (n, 3) and the per-channel weights."""
    Xs = features(P, variant, knots)
    pred, Ws = torch.empty_like(P), []
    for c in range(3):
        X = Xs[c].double()
        y = (G[:, c] - P[:, c] if variant == 'offset' else G[:, c]).double()
        Xf, yf = X[fit_mask], y[fit_mask]
        A = Xf.T @ Xf
        A = A + 1e-9 * A.diagonal().mean() * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        w = torch.linalg.solve(A, Xf.T @ yf)
        p = X @ w
        pred[:, c] = (p + P[:, c].double() if variant == 'offset' else p).float()
        Ws.append(w)
    return pred, Ws


def mse_official(pred, G):
    """pred, G: (n, 3) -> MSE of the clamped, 14-bit-truncated prediction."""
    return (C.quantise(pred) - G).pow(2).mean().item()


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument('--val_root', default=None, help='SyntheticBurstVal root (default: config datasets.val)')
    args = ap.parse_args()
    ctx = C.Context(args, 'colour_fit')
    model, dev = ctx.model, ctx.device
    ds = C.ValBursts(ctx.val_root(args.val_root))
    knots = torch.linspace(0, 1, KNOTS, device=dev, dtype=torch.float64) ** 2
    rows, glob_stats, mats = [], [], []
    bias_sum = np.zeros((3, len(BIAS_EDGES) - 1))
    bias_cnt = np.zeros((3, len(BIAS_EDGES) - 1))

    for data in C.loader(ds, args):
        burst, gt = data['burst'], data['gt'].to(dev)
        B, Nf = burst.shape[:2]
        lq, _ = C.arrange(burst, range(1, Nf), C.ref_slot(model, Nf))
        out = ctx.run(lq)
        for i in range(B):
            Pi, Gi = C.crop(C.quantise(out[i])), C.crop(gt[i])
            h, w = Pi.shape[-2:]
            P, G = Pi.reshape(3, -1).T.contiguous(), Gi.reshape(3, -1).T.contiguous()
            yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing='ij')
            A = (((yy // 16) + (xx // 16)) % 2 == 0).flatten()
            allm = torch.ones_like(A)
            r = {'name': data['name'][i], 'shot': float(data['shot'][i]), 'mse': (P - G).pow(2).mean().item()}
            r['psnr'] = C.psnr_from_mse(r['mse'])
            for v in VARIANTS:
                pred, Ws = fit_apply(P.double(), G.double(), v, knots, allm)
                r[f'psnr_{v}'] = C.psnr_from_mse(mse_official(pred.float(), G))
                pa, _ = fit_apply(P.double(), G.double(), v, knots, A)          # fit on A, score on B
                pb, _ = fit_apply(P.double(), G.double(), v, knots, ~A)         # fit on B, score on A
                cross = torch.where(A[:, None], pb, pa).float()
                r[f'psnr_{v}_cv'] = C.psnr_from_mse(mse_official(cross, G))
                if v == 'affine':
                    M = torch.stack([Ws[c] for c in range(3)])                  # (3 out, 4 = R G B 1)
                    mats.append(M.cpu().numpy())
                    r['affine_dev'] = float((M[:, :3] - torch.eye(3, device=dev, dtype=M.dtype)).norm())
                    r['off_r'], r['off_g'], r['off_b'] = M[:, 3].tolist()
                if v == 'diag':
                    r['gain_r'], r['gain_g'], r['gain_b'] = [float(Ws[c][0]) for c in range(3)]
            # for the one-map-for-the-whole-set fit: this image's normal equations
            X = torch.cat([P, torch.ones_like(P[:, :1])], 1).double()
            glob_stats.append((X.T @ X, X.T @ G.double(), (G.double() ** 2).sum(0), P.shape[0]))
            # bias by GT intensity
            e, g = (P - G).cpu().numpy(), G.cpu().numpy()
            for c in range(3):
                idx = np.clip(np.searchsorted(BIAS_EDGES, g[:, c], side='right') - 1, 0, len(BIAS_EDGES) - 2)
                bias_sum[c] += np.bincount(idx, e[:, c], len(BIAS_EDGES) - 1)
                bias_cnt[c] += np.bincount(idx, minlength=len(BIAS_EDGES) - 1)
            rows.append(r)
        print(f'  {len(rows)}/{args.limit or len(ds)}', end='\r', flush=True)
    print()

    # one affine map for the whole set
    XtX = sum(s[0] for s in glob_stats)
    XtG = sum(s[1] for s in glob_stats)
    Wg = torch.linalg.solve(XtX, XtG)                                         # (4, 3)
    for r, (xx_, xg, gg, n) in zip(rows, glob_stats):
        sse = gg - 2 * (Wg * xg).sum(0) + torch.einsum('ic,ij,jc->c', Wg, xx_, Wg)
        r['psnr_global_affine'] = C.psnr_from_mse((sse.sum() / (3 * n)).item())
    report(ctx, rows, Wg.T.cpu().numpy(), np.stack(mats), bias_sum, bias_cnt)


def report(ctx, rows, Wg, mats, bias_sum, bias_cnt):
    L = ctx.log
    col = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    base = col('psnr')
    L(f'{len(rows)} bursts.  PSNR {base.mean():.3f} dB')
    L()
    L('== PSNR gain from a per-image colour map (mean ± 95% CI, dB; cross-fit = fit on half, score on the other half)')
    L(f'   {"fit":<8} {"params":>6} {"in-sample":>18} {"cross-fit":>18} {"images > +0.1 dB":>18}')
    nparam = {'offset': 3, 'gain': 3, 'diag': 6, 'affine': 12, 'tone': 3 * 3 * KNOTS}
    gains = {}
    for v in VARIANTS:
        d_in, d_cv = col(f'psnr_{v}') - base, col(f'psnr_{v}_cv') - base
        gains[v] = {'in': C.mci(d_in), 'cv': C.mci(d_cv), 'frac_gt_0.1': float((d_cv > 0.1).mean())}
        L(f'   {v:<8} {nparam[v]:6d} {C.fci(d_in):>18} {C.fci(d_cv):>18} {gains[v]["frac_gt_0.1"]:18.0%}')
    dg = col('psnr_global_affine') - base
    L(f'   {"global":<8} {12:6d} {C.fci(dg):>18} {"(one map, all images; no clamp/quantise)":>40}')
    L()
    L('== fitted maps')
    L(f'   mean per-image affine (rows = out R G B; cols = in R G B, offset x1e4):')
    M = mats.mean(0)
    for c, nm in enumerate('RGB'):
        L(f'     {nm}: {M[c, 0]:+.4f} {M[c, 1]:+.4f} {M[c, 2]:+.4f}   {1e4 * M[c, 3]:+7.2f}')
    L(f'   spread (std over images) of the diagonal: {mats[:, [0, 1, 2], [0, 1, 2]].std(0).round(4).tolist()}, '
      f'of the offsets x1e4: {(1e4 * mats[:, :, 3].std(0)).round(2).tolist()}')
    L(f'   global affine: diag {np.diag(Wg[:, :3]).round(4).tolist()}  offsets x1e4 {(1e4 * Wg[:, 3]).round(2).tolist()}')
    L(f'   per-channel gains (diag fit), mean: R {col("gain_r").mean():.4f}  G {col("gain_g").mean():.4f}  B {col("gain_b").mean():.4f}')
    L()
    L('== bias (output - GT) by GT intensity, x1e4, pooled over pixels')
    L(f'   {"GT range":>15} ' + ''.join(f'{c:>9}' for c in 'RGB') + f'{"pixels":>10}')
    for k in range(bias_sum.shape[1]):
        if bias_cnt[:, k].sum() == 0:
            continue
        L(f'   {BIAS_EDGES[k]:.3f} .. {BIAS_EDGES[k + 1]:.3f} '
          + ''.join(f'{1e4 * bias_sum[c, k] / max(bias_cnt[c, k], 1):+9.2f}' for c in range(3))
          + f'{int(bias_cnt[0, k]):10d}')
    rmse = np.sqrt(np.mean(col('mse')))
    L(f'   (for scale: RMS error {1e4 * rmse:.1f} x1e4)')
    L()
    terc = []
    groups = C.terciles(col('shot').tolist())
    if groups:
        L('== affine cross-fit gain by noise level (tercile of shot noise)')
        for g in groups:
            d = col('psnr_affine_cv')[g] - base[g]
            sh = col('shot')[g]
            terc.append({'shot_lo': sh.min(), 'shot_hi': sh.max(), 'psnr': base[g].mean(), 'gain': d.mean()})
            L(f'   {sh.min():.2e} .. {sh.max():.2e}  PSNR {base[g].mean():6.2f}  gain {C.fci(d)} dB')

    a = gains['affine']['cv'][0]
    t = gains['tone']['cv'][0]
    ro = []
    if a >= 0.2:
        ro.append(f'A per-image 3x3 + offset recovers {a:+.2f} dB (cross-fit): global colour errors cost real dB that no '
                  f'sub-pixel change will recover.')
    elif a >= 0.05:
        ro.append(f'A per-image 3x3 + offset recovers {a:+.2f} dB (cross-fit): small but not nothing.')
    else:
        ro.append(f'A per-image 3x3 + offset recovers {a:+.3f} dB (cross-fit): global colour is not where the missing dB are.')
    off, gn = gains['offset']['cv'][0], gains['gain']['cv'][0]
    if a >= 0.05:
        what = max((('offset (bias)', off), ('gain (exposure / WB)', gn), ('channel mixing', a - gains['diag']['cv'][0])),
                   key=lambda kv: kv[1])
        ro.append(f'Most of it is {what[0]} ({what[1]:+.2f} dB). One global map recovers {dg.mean():+.2f} dB of the '
                  f'{a:+.2f}: ' + ('systematic -- a calibration / bias problem.' if dg.mean() >= 0.6 * a else
                                   'image-dependent.'))
    if t - a >= 0.05:
        ro.append(f'A nonlinear tone curve adds {t - a:+.2f} dB over the affine fit: intensity-dependent bias '
                  f'(see the bias table; dark-region noise-clipping bias is the usual suspect).')
    ro.append('Note: SyntheticBurst\'s GT shares the RAW\'s camera colour space, so this measures colour/tone bias, '
              'not a colour matrix the network failed to invert.')
    headline = {'psnr': float(base.mean()), 'gain_affine_cv': a, 'gain_tone_cv': t, 'gain_offset_cv': off,
                'gain_gain_cv': gn, 'gain_global_affine': float(dg.mean())}
    ctx.save(headline, ro, rows, extra={'gains': gains, 'mean_affine': M, 'global_affine': Wg,
                                        'bias_x1e4': (1e4 * bias_sum / np.maximum(bias_cnt, 1)), 'bias_edges': BIAS_EDGES,
                                        'noise_terciles': terc})
    plot(ctx, rows, bias_sum, bias_cnt)


def plot(ctx, rows, bias_sum, bias_cnt):
    plt = C.plt()
    col = lambda k: np.array([r[k] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for v in VARIANTS:
        ax[0].hist(col(f'psnr_{v}_cv') - col('psnr'), bins=40, histtype='step', label=v)
    ax[0].set_xlabel('PSNR gain, cross-fit (dB)')
    ax[0].set_ylabel('images')
    ax[0].legend(fontsize=8)
    ax[0].set_title('per-image colour-map gain')
    mid = np.sqrt(BIAS_EDGES[1:] * np.maximum(BIAS_EDGES[:-1], 1e-4))
    for c, colr in enumerate(('C3', 'C2', 'C0')):
        ok = bias_cnt[c] > 0
        ax[1].semilogx(mid[ok], 1e4 * bias_sum[c][ok] / bias_cnt[c][ok], 'o-', color=colr, label='RGB'[c])
    ax[1].axhline(0, color='0.6', lw=0.8)
    ax[1].set_xlabel('GT intensity')
    ax[1].set_ylabel('mean output - GT (x1e4)')
    ax[1].legend(fontsize=8)
    ax[1].set_title('bias by intensity')
    fig.suptitle(f'colour_fit  {ctx.opt["name"]}  {os.path.basename(ctx.ckpt)}')
    fig.tight_layout()
    fig.savefig(ctx.path('.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
