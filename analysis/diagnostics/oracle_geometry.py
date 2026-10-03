#!/usr/bin/env python3
"""(3) Oracle geometry: does LK's tap-position error (and its heavy tail) cost dB?

    python analysis/diagnostics/oracle_geometry.py --config main/configs/M1_KGTSMamba.yml [--ckpt latest]
        [--split test] [--n 300] [--tau 0.25] [--jitter 0.05,0.1,0.2]

SyntheticBurstVal ships no flow, so this generates validation-like bursts WITH the generator's
flow (common.GeneratedBursts: official DBSR protocol, Zurich RAW-to-RGB 'test' split -- the one
SyntheticBurstVal was made from -- deterministic per index). Each burst runs once per arm, all arms
in one batch with the same routing noise, differing only in the flow the taps are gathered with
(KGTSAlign's output is swapped; features, body and everything else are untouched):

  lk          the model as trained: align.global_motion lk (robust photometric affine registration)
  oracle      the generator's own geometry (backward field, packed px; common.oracle_flow)
  tailfix     LK, but every frame whose LK error exceeds --tau packed px gets the oracle instead:
              what fixing ONLY the tail is worth
  fit         the affine fit of FlowAlign's dense flow (global_motion 'affine', the previous geometry)
  jitter_s    oracle + a random per-frame translation error, N(0, s^2) per axis: a dose-response
              curve, PSNR against a known, uniform geometry error
  taildrop    LK with the tail frames REMOVED (true-length burst): rejecting instead of fixing
              (a separate forward; only bursts that have a tail frame)

LK error is measured per frame against the oracle (mean end-point error over the frame's interior,
packed px; 1 packed px = 8 HR px). Reported: its distribution (the tail), PSNR per arm with paired
95% CIs, where LK sits on the jitter dose-response curve, how much of the oracle's gain the tail
alone accounts for, and the gain against the burst's own LK error and noise level.

Caveat: the model was trained on LK positions, so its position codes are calibrated to LK's error.
Feeding it better geometry at test time measures what better geometry is worth to THIS model -- a
lower bound on what training with it would give (it might learn to trust positions more).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def frame_epe(a, b, m=4):
    """(..., N, 2, h, w) x2 -> (..., N) mean end-point error over the interior."""
    return (a - b)[..., m:-m, m:-m].float().norm(dim=-3).mean((-2, -1))


class FlowSwap:
    """Wraps model.align.forward: runs it, then hands its output flow to self.fn (if set), which
    returns the flow the taps are gathered with. Everything else KGTSAlign returns is kept."""

    def __init__(self, align):
        self.orig, self.fn = align.forward, None
        align.forward = self

    def __call__(self, burst, ref_idx, size=None):
        feats, flow, flows = self.orig(burst, ref_idx, size=size)
        if self.fn is not None:
            flow = self.fn(flow, flows, size)
        return feats, flow, flows


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument('--zurich_root', default=None, help='Zurich RAW-to-RGB root (default: config datasets.train)')
    ap.add_argument('--split', default='test', help="Zurich split to generate from ('test'; 'train' if absent)")
    ap.add_argument('--n', type=int, default=300, help='bursts to generate')
    ap.add_argument('--tau', type=float, default=0.25, help='LK tail threshold, packed px (0.25 = 2 HR px)')
    ap.add_argument('--jitter', default='0.05,0.1,0.2', help='per-axis translation error std, packed px')
    args = ap.parse_args()
    ctx = C.Context(args, 'oracle_geometry')
    model, dev = ctx.model, ctx.device
    from burstISP.archs.KGTSMamba.flow_align_arch import affine_flow_fit
    root = ctx.zurich_root(args.zurich_root)
    split = args.split
    if not os.path.isdir(os.path.join(root, split, 'canon')):
        ctx.log(f'WARNING: no {os.path.join(root, split, "canon")}; generating from the TRAIN split instead. '
                f'The model has seen these images (other crops / motions / noise): absolute PSNR is optimistic, '
                f'the paired arm deltas are still valid.')
        split = 'train'
    ds = C.GeneratedBursts(root, split, n=args.n, seed=args.seed)
    jit = [float(s) for s in args.jitter.split(',') if s]
    arms = ['lk', 'oracle', 'tailfix', 'fit'] + [f'jitter_{s:g}' for s in jit]
    swap = FlowSwap(model.align)
    gm = getattr(model.align, 'gm_margin', 4)
    if getattr(model.align, 'global_motion', None) != 'lk':
        ctx.log(f'NOTE: this model uses align.global_motion={model.align.global_motion!r}, not lk: '
                f'the "lk" arm is whatever the model computes.')
    rows, frames_epe = [], []

    for data in C.loader(ds, args):
        for j in range(data['burst'].shape[0]):
            burst, fv, gt = data['burst'][j:j + 1], data['flow'][j:j + 1], data['gt'][j:j + 1].to(dev)
            Nf = burst.shape[1]
            slot = C.ref_slot(model, Nf)
            lq, order = C.arrange(burst, range(1, Nf), slot)
            fo = C.oracle_flow(fv)[:, order].to(dev)                         # (1, N, 2, h, w)
            nonkey = torch.tensor([o != 0 for o in order], device=dev)
            g = torch.Generator(device='cpu').manual_seed(args.seed * 7919 + len(rows))
            noise = {s: torch.randn(1, Nf, 2, 1, 1, generator=g) * s * nonkey.cpu().view(1, Nf, 1, 1, 1) for s in jit}
            st = {}

            def make(flow, flows, size):
                lk = flow[:1].float()
                e = frame_epe(lk, fo)[0]                                      # (N,)
                tail = (e > args.tau) & nonkey
                dense = flows['lv1'][:1].float()
                if dense.shape[-2:] != lk.shape[-2:]:                         # bayer align: fold like PostAlign
                    dense = torch.nn.functional.avg_pool2d(dense.flatten(0, 1), 2).view_as(lk) * 0.5
                assert fo.shape == lk.shape, f'oracle flow {tuple(fo.shape)} vs model flow {tuple(lk.shape)} (padding?)'
                fit = affine_flow_fit(dense.flatten(0, 1), gm, size=size).view_as(dense)
                fit[:, slot] = 0
                fl = {'lk': lk, 'oracle': fo, 'tailfix': torch.where(tail.view(1, -1, 1, 1, 1), fo, lk), 'fit': fit}
                for s in jit:
                    fl[f'jitter_{s:g}'] = fo + noise[s].to(dev)
                st.update(epe=e, tail=tail, arm_epe={a: frame_epe(fl[a], fo)[0][nonkey].mean().item() for a in arms})
                return torch.cat([fl[a] for a in arms]).to(flow.dtype)

            swap.fn = make
            out = ctx.run(lq.expand(len(arms), *lq.shape[1:]).contiguous())
            swap.fn = None
            r = {'name': data['name'][j], 'shot': float(data['shot'][j])}
            for a, o in zip(arms, out):
                r[f'psnr_{a}'] = C.psnr_from_mse(C.error(o, gt[0]).pow(2).mean().item())
                r[f'epe_{a}'] = st['arm_epe'][a]
            e = st['epe'][nonkey].cpu().numpy()
            frames_epe += [(len(rows), float(v)) for v in e]
            r.update(lk_mean=float(e.mean()), lk_max=float(e.max()), lk_tail=int(st['tail'].sum()))
            # rejecting the tail instead: the burst without its tail frames, the model's own flow
            if r['lk_tail']:
                keep = [o for o, t in zip(order, st['tail'].tolist()) if o != 0 and not t]
                lq2, _ = C.arrange(burst, keep, C.ref_slot(model, len(keep) + 1))
                r['psnr_taildrop'] = C.psnr_from_mse(C.error(ctx.run(lq2)[0], gt[0]).pow(2).mean().item())
            else:
                r['psnr_taildrop'] = r['psnr_lk']
            rows.append(r)
            print(f'  {len(rows)}/{len(ds) if not args.limit else min(args.limit, len(ds))}', end='\r', flush=True)

    print()
    report(ctx, rows, arms, jit, np.array([v for _, v in frames_epe]), args.tau, split)


def report(ctx, rows, arms, jit, fe, tau, split):
    L = ctx.log
    col = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    n = len(rows)
    L(f'{n} generated bursts (Zurich {split}), {len(fe)} non-key frames.')
    L()
    L('== LK tap-position error per frame vs the generator (packed px; x8 = HR px)')
    q = np.percentile(fe, [50, 90, 99])
    L(f'   mean {fe.mean():.3f}   median {q[0]:.3f}   p90 {q[1]:.3f}   p99 {q[2]:.3f}   max {fe.max():.3f}')
    L('   frames over: ' + '   '.join(f'{t} px {np.mean(fe > t):.1%}' for t in (0.1, 0.25, 0.5, 1.0)))
    tail_bursts = col('lk_tail') > 0
    L(f'   bursts with >= 1 frame over tau={tau}: {tail_bursts.mean():.1%}   '
      f'(share of the mean error from those frames: {fe[fe > tau].sum() / max(fe.sum(), 1e-12):.0%})')
    L()
    lk = col('psnr_lk')
    L('== PSNR per arm (paired with lk: same bursts, same routing noise)')
    L(f'   {"arm":<14} {"mean EPE":>9} {"PSNR":>8} {"vs lk (dB)":>18} {"vs oracle":>10}')
    arm_rows = {}
    for a in arms + ['taildrop']:
        p = col(f'psnr_{a}')
        ep = col(f'epe_{a}').mean() if f'epe_{a}' in rows[0] else float('nan')
        arm_rows[a] = {'epe': ep, 'psnr': p.mean(), 'vs_lk': C.mci(p - lk), 'vs_oracle': float((p - col('psnr_oracle')).mean())}
        L(f'   {a:<14} {ep:9.3f} {p.mean():8.3f} {C.fci(p - lk):>18} {arm_rows[a]["vs_oracle"]:+10.3f}')
    L('   (taildrop: bursts without a tail frame count with their lk PSNR)')
    L()
    g_or = col('psnr_oracle') - lk
    g_tf = col('psnr_tailfix') - lk
    m, m_ci = C.mci(g_or)
    real = m >= 0.02 and m > 2 * (m_ci if m_ci == m_ci else 0)          # the oracle gain is above the noise
    share = g_tf.sum() / g_or.sum() if real else float('nan')
    L('== does the tail matter?')
    L(f'   oracle gain {C.fci(g_or)} dB; fixing only frames over {tau} px gives {C.fci(g_tf)} dB'
      + (f' = {share:.0%} of it' if real else ' (oracle gain within noise: no share)'))
    if tail_bursts.any() and (~tail_bursts).any():
        L(f'   bursts WITH a tail frame ({tail_bursts.sum()}): oracle gain {C.fci(g_or[tail_bursts])}   '
          f'taildrop vs lk {C.fci((col("psnr_taildrop") - lk)[tail_bursts])}')
        L(f'   bursts without ({(~tail_bursts).sum()}): oracle gain {C.fci(g_or[~tail_bursts])}')
    order = np.argsort(col('lk_max'))
    dec = max(1, n // 10)
    L(f'   worst decile by LK max-frame error (>= {col("lk_max")[order[-dec]]:.2f} px): oracle gain {C.fci(g_or[order[-dec:]])}; '
      f'other 90%: {C.fci(g_or[order[:-dec]])}')
    rho = spearman(col('lk_mean'), g_or)
    L(f'   Spearman(burst mean LK error, oracle gain) = {rho:+.2f}')
    L()
    L('== dose-response: PSNR below oracle vs mean frame error (jitter arms = known uniform error)')
    pts = [(arm_rows[f'jitter_{s:g}']['epe'], -arm_rows[f'jitter_{s:g}']['vs_oracle']) for s in jit]
    for (x, y), s in zip(pts, jit):
        L(f'   jitter {s:<5g} EPE {x:.3f}  ->  {y:+.3f} dB below oracle')
    lk_x, lk_y = arm_rows['lk']['epe'], -arm_rows['lk']['vs_oracle']
    curve = sorted([(0.0, 0.0)] + [(y, x) for x, y in pts])                  # (dB drop, EPE), by drop
    ys = [y for _, y in sorted(pts)]
    readable = bool(pts) and max(ys) >= 0.02 and all(b >= a - 0.005 for a, b in zip(ys, ys[1:]))
    equiv = float(np.interp(lk_y, [c[0] for c in curve], [c[1] for c in curve])) if readable else float('nan')
    slope = np.polyfit([0] + [x for x, _ in pts], [0] + [y for _, y in pts], 1)[0] if pts else float('nan')
    if readable:
        L(f'   lk          EPE {lk_x:.3f}  ->  {lk_y:+.3f} dB below oracle; a uniform error of {equiv:.3f} px would cost '
          f'the same. ' + ('LK costs MORE than its mean error predicts: the tail is doing the damage.'
                           if equiv > 1.5 * lk_x else 'LK costs about what its mean error predicts.' if equiv > 0.67 * lk_x
                           else 'LK costs LESS than its mean error predicts (its error is mostly where it does not matter).'))
    else:
        L(f'   lk          EPE {lk_x:.3f}  ->  {lk_y:+.3f} dB below oracle.  The jitter costs are below the noise floor '
          f'(< 0.02 dB) or not monotone: no dose-response reading.')
    L(f'   linear slope through the jitter points: {slope / 10:.3f} dB per 0.1 packed px')
    L()
    terc = []
    groups = C.terciles(col('shot').tolist())
    if groups:
        L('== by noise level (tercile of shot noise)')
        for gidx in groups:
            sh = col('shot')[gidx]
            d = {'shot_lo': sh.min(), 'shot_hi': sh.max(), 'lk_mean': col('lk_mean')[gidx].mean(),
                 'psnr_lk': lk[gidx].mean(), 'oracle_gain': g_or[gidx].mean()}
            terc.append(d)
            L(f'   {d["shot_lo"]:.2e} .. {d["shot_hi"]:.2e}  LK error {d["lk_mean"]:.3f} px  PSNR {d["psnr_lk"]:6.2f}  '
              f'oracle gain {C.fci(g_or[gidx])}')

    ro = []
    if m < 0.05:
        ro.append(f'Perfect geometry is worth {m:+.3f} dB to this model: tap positions are not the bottleneck '
                  f'(candidate 5 is off the table on SyntheticBurst).')
    else:
        ro.append(f'Perfect geometry is worth {m:+.2f} dB.' + ('' if not real else (
            f' Fixing only the frames over {tau} px recovers {share:.0%} of it: the tail is the problem -> candidate 5 '
            f'(e.g. more LK starts / a per-frame residual check / re-registration of the worst frames).'
            if share >= 0.6 else
            f' The tail accounts for only {share:.0%}: the bulk sub-pixel precision (median {q[0]:.3f} px) matters -- a '
            f'better registration overall, not outlier handling.')))
    if readable:
        ro.append(f'Geometry sensitivity: {slope / 10:.2f} dB per 0.1 packed px of uniform error; LK behaves like a '
                  f'{equiv:.3f} px uniform error (its mean is {lk_x:.3f}).')
    if tail_bursts.any():
        td = (col('psnr_taildrop') - lk)[tail_bursts].mean()
        ro.append(f'Dropping tail frames instead of fixing them: {td:+.3f} dB on the affected bursts '
                  + ('(rejection already helps: a learned/explicit frame-rejection would too).' if td > 0.02 else
                     '(the model gets more from a slightly misplaced frame than from none).'))
    headline = {'lk_epe_mean': float(fe.mean()), 'lk_epe_median': float(q[0]), 'lk_epe_p99': float(q[2]),
                'psnr_lk': float(lk.mean()), 'oracle_gain_db': float(m), 'tailfix_gain_db': float(g_tf.mean()),
                'tail_share_of_gain': float(share), 'db_per_0.1px': float(slope / 10), 'lk_equiv_uniform_px': float(equiv)}
    ctx.save(headline, ro, rows, extra={'arms': arm_rows, 'noise_terciles': terc, 'frame_epe_hist':
                                        np.histogram(fe, bins=np.geomspace(1e-3, 10, 41))})
    plot(ctx, rows, arms, arm_rows, fe, pts, (lk_x, lk_y), tau)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float('nan')


def plot(ctx, rows, arms, arm_rows, fe, pts, lkpt, tau):
    plt = C.plt()
    col = lambda k: np.array([r[k] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].hist(fe, bins=np.geomspace(max(fe.min(), 1e-3), max(fe.max(), 1e-2), 40))
    ax[0].set_xscale('log')
    ax[0].axvline(tau, color='C3', ls='--', label=f'tau {tau}')
    ax[0].set_xlabel('LK frame error (packed px)')
    ax[0].set_title('LK error per frame')
    ax[0].legend(fontsize=8)
    ax[1].scatter(col('lk_max'), col('psnr_oracle') - col('psnr_lk'), s=8, alpha=0.6)
    ax[1].set_xscale('log')
    ax[1].axhline(0, color='0.6', lw=0.8)
    ax[1].set_xlabel('burst max-frame LK error (packed px)')
    ax[1].set_ylabel('oracle - lk (dB)')
    ax[1].set_title('gain from perfect geometry')
    if pts:
        ax[2].plot([0] + [x for x, _ in pts], [0] + [y for _, y in pts], 'o-', label='oracle + uniform jitter')
    ax[2].plot(*lkpt, 'C3*', ms=14, label='lk')
    if 'fit' in arm_rows:
        ax[2].plot(arm_rows['fit']['epe'], -arm_rows['fit']['vs_oracle'], 'C2s', label='fit (dense-flow affine)')
    ax[2].set_xlabel('mean frame error (packed px)')
    ax[2].set_ylabel('dB below oracle')
    ax[2].set_title('dose-response')
    ax[2].legend(fontsize=8)
    fig.suptitle(f'oracle_geometry  {ctx.opt["name"]}  {os.path.basename(ctx.ckpt)}')
    fig.tight_layout()
    fig.savefig(ctx.path('.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
