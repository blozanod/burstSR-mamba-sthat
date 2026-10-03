#!/usr/bin/env python3
"""(4) True-length burst curve: PSNR (and error per band) for N = 1 ... 14 frames.

    python analysis/diagnostics/burst_length.py --config main/configs/M1_KGTSMamba.yml [--ckpt latest]
        [--ns 1-14] [--draws 1] [--source val|generated]

TRUE length: an N-frame burst is fed as N frames (the keyframe plus N-1 others, keyframe in the
slot the network reads for N), not padded with keyframe copies as burst_ablation.py's frame_drop
mode did -- KGTSMamba is frame-count agnostic (L = N k^2 taps per pixel). Draw 0 takes the other
frames in generator order (the generator's frames are i.i.d. random motions, so that is a random
subset); --draws > 1 adds seeded random subsets, nested across N within a draw. All N of a burst
share the routing noise, so the curve is paired. Also one all-ref pass (14 copies of the
keyframe: full length, no new information).

Per N: PSNR, and the error energy in each band (common.py: 5 radial bands in packed-Nyquist
units + the 2-band split at one LR pixel) with the transfer (MTF) of the top bands. That says
WHICH error the extra frames remove: below one LR pixel they are mostly denoising; above 2 F_N
only sub-pixel information can help.

Reading it. The curve's slope is reported per doubling of N. Still climbing at 8 -> 14 (and in the
top bands) means the burst branch keeps extracting from every frame it gets -- more samples per
frame (k = 3 taps) and every frame's geometry matter. Flat from ~8 means extra frames carry
information the branch does not use: under-extraction (the pooled state, the injection into the
packed-grid body, or the x8 tail is the bottleneck, not the number of samples).

CAVEAT: M1 trains with burst_aug (prob 0.2, min_frames 8): it never saw N < 8. The left half of
the curve is out of distribution, so a steep 4 -> 8 rise can be the model recovering from
unfamiliar inputs rather than information; judge the end of the curve on N >= min_frames.
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def parse_ns(s, nmax):
    out = set()
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(n for n in out if 1 <= n <= nmax)


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument('--ns', default='1-14', help="burst lengths, e.g. '1-14' or '1,2,4,8,14'")
    ap.add_argument('--draws', type=int, default=1, help='frame subsets per burst (0 = generator order, then random)')
    ap.add_argument('--source', choices=['val', 'generated'], default='val')
    ap.add_argument('--val_root', default=None)
    ap.add_argument('--zurich_root', default=None)
    ap.add_argument('--split', default='test')
    ap.add_argument('--n', type=int, default=300, help='generated bursts (--source generated)')
    args = ap.parse_args()
    ctx = C.Context(args, 'burst_length')
    model, dev = ctx.model, ctx.device
    ds = C.ValBursts(ctx.val_root(args.val_root)) if args.source == 'val' else \
        C.GeneratedBursts(ctx.zurich_root(args.zurich_root), args.split, n=args.n, seed=args.seed)
    S = None
    rows, seen = [], 0
    for data in C.loader(ds, args):
        burst, gt = data['burst'], data['gt'].to(dev)
        B, Nf = burst.shape[:2]
        seen += B
        ns = parse_ns(args.ns, Nf)
        for d in range(args.draws):
            perms = []
            for b in range(B):
                others = list(range(1, Nf))
                if d:
                    g = np.random.default_rng([args.seed, d, seen - B + b])     # per burst, nested across N
                    others = [others[k] for k in g.permutation(len(others))]
                perms.append(others)
            arms = [(n, d) for n in ns] + ([('allref', d)] if d == 0 else [])
            for n, _ in arms:
                if n == 'allref':
                    lq = burst[:, :1].expand_as(burst).contiguous()
                else:
                    lq = torch.cat([C.arrange(burst[b:b + 1], perms[b][:n - 1], C.ref_slot(model, n))[0]
                                    for b in range(B)])
                out = ctx.run(lq)
                for i in range(B):
                    g_ = C.crop(gt[i])
                    if S is None:
                        S = C.Spectrum(*g_.shape[-2:], dev)
                    e = C.error(out[i], gt[i])
                    X, Xg = S.fft(e), S.fft(g_ - g_.mean((-2, -1), keepdim=True))
                    Eb, cb, Pg = S.bands(X), S.bands(X, Xg), S.bands(Xg)
                    r = {'name': data['name'][i], 'draw': d, 'n': n, 'shot': float(data['shot'][i]),
                         'mse': e.pow(2).mean().item()}
                    r['psnr'] = C.psnr_from_mse(r['mse'])
                    r['lo'], r['hi'] = S.split(X).tolist()
                    for k in range(len(C.BAND_NAMES)):
                        r[f'e_{k}'] = Eb[k].item()
                        r[f'mtf_{k}'] = (1 + cb[k] / Pg[k].clamp_min(1e-20)).item()
                    rows.append(r)
        print(f'  {len({r["name"] for r in rows})}/{args.limit or len(ds)} bursts', end='\r', flush=True)
    print()
    report(ctx, rows, ns)


def report(ctx, rows, ns):
    L = ctx.log
    aug = ctx.opt.get('train', {}).get('burst_aug') or {}
    nmin = int(aug.get('min_frames', max(ns))) if aug.get('prob', 0) > 0 else max(ns)
    names = sorted({r['name'] for r in rows})
    draws = sorted({r['draw'] for r in rows})
    by = {(r['name'], r['draw'], r['n']): r for r in rows}
    nb = len(C.BAND_NAMES)

    def stat(n, key):
        """per-image value averaged over draws (allref: draw 0 only)"""
        dd = [0] if n == 'allref' else draws
        return np.array([np.mean([by[(nm, d, n)][key] for d in dd]) for nm in names])

    nmax = max(ns)
    p_ref = stat(nmax, 'psnr')
    L(f'{len(names)} bursts x {len(draws)} draw(s).  Trained with N in [{nmin}, {nmax}] '
      f'(train.burst_aug); N < {nmin} is out of distribution (marked *).')
    L()
    L(f'== PSNR vs true burst length.  band columns: band error energy relative to N={nmax}, dB (+ = more error)')
    hdr = f'   {"N":>4} {"PSNR":>8} {"vs N-1":>8} {f"vs N={ns[0]}":>8} {"low":>6} {"high":>6} ' + \
          ''.join(f'{b:>7}' for b in C.BAND_NAMES) + f' {"MTF 2-4":>8} {"MTF >4":>7}'
    L(hdr)
    curve = {}
    e_ref = {k: stat(nmax, f'e_{k}') for k in range(nb)}
    lo_ref, hi_ref = stat(nmax, 'lo'), stat(nmax, 'hi')
    rel = lambda a, b: float(np.mean(10 * np.log10(np.maximum(a, 1e-20) / np.maximum(b, 1e-20))))
    prev = None
    for n in list(ns) + ['allref']:
        p = stat(n, 'psnr')
        d = {'psnr': float(p.mean()), 'ci': C.mci(p)[1],
             'vs_prev': float((p - prev).mean()) if prev is not None and n != 'allref' else float('nan'),
             'vs_1': float((p - stat(ns[0], 'psnr')).mean()) if n != ns[0] else 0.0,
             'low_db': rel(stat(n, 'lo'), lo_ref), 'high_db': rel(stat(n, 'hi'), hi_ref),
             'band_db': [rel(stat(n, f'e_{k}'), e_ref[k]) for k in range(nb)],
             'mtf': [float(np.median(stat(n, f'mtf_{k}'))) for k in range(nb)]}
        curve[n] = d
        tag = '*' if n != 'allref' and n < nmin else ' '
        lab = f'{n}{tag}' if n != 'allref' else 'all-ref'
        L(f'   {lab:>4} {d["psnr"]:8.3f} {d["vs_prev"]:+8.3f} {d["vs_1"]:+8.3f} {d["low_db"]:+6.2f} {d["high_db"]:+6.2f} '
          + ''.join(f'{v:+7.2f}' for v in d['band_db']) + f' {d["mtf"][3]:8.3f} {d["mtf"][4]:7.3f}')
        if n != 'allref':
            prev = p
    L(f'   (all-ref = {nmax} copies of the keyframe; vs N=1 there is the cost of the N=1 path itself)')
    L()

    def slope(lo_n, hi_n, key='psnr'):
        """least-squares dB per doubling of N over the ns in [lo_n, hi_n]"""
        xs = [n for n in ns if lo_n <= n <= hi_n]
        if len(xs) < 2:
            return float('nan')
        return float(np.polyfit(np.log2(xs), [curve[n][key] for n in xs], 1)[0])

    def band_slope(lo_n, hi_n, k):
        xs = [n for n in ns if lo_n <= n <= hi_n]
        if len(xs) < 2:
            return float('nan')
        return float(-np.polyfit(np.log2(xs), [curve[n]['band_db'][k] for n in xs], 1)[0])

    L('== slope, dB per doubling of N (least squares on log2 N)')
    segs = [(1, 2), (2, 4), (4, 8), (nmin, nmax)]
    sl = {}
    for a, b in segs:
        sl[f'{a}-{b}'] = slope(a, b)
        bands = [band_slope(a, b, k) for k in range(nb)]
        ood = ' (out of distribution)' if a < nmin else ''
        L(f'   N {a:>2}..{b:<2}  PSNR {sl[f"{a}-{b}"]:+.3f}   error drop per band: '
          + '  '.join(f'{C.BAND_NAMES[k]} {bands[k]:+.2f}' for k in range(nb)) + ood)
    end = sl[f'{nmin}-{nmax}']
    early = sl['2-4']
    L()
    terc = []
    groups = C.terciles(stat(nmax, 'shot').tolist())
    if groups:
        L(f'== end slope (N {nmin}..{nmax}, dB/doubling) by noise level')
        sh = stat(nmax, 'shot')
        for g in groups:
            xs = [n for n in ns if nmin <= n <= nmax]
            ys = [stat(n, 'psnr')[g].mean() for n in xs]
            s = float(np.polyfit(np.log2(xs), ys, 1)[0]) if len(xs) > 1 else float('nan')
            terc.append({'shot_lo': sh[g].min(), 'shot_hi': sh[g].max(), 'end_slope': s,
                         'psnr': float(stat(nmax, 'psnr')[g].mean())})
            L(f'   {sh[g].min():.2e} .. {sh[g].max():.2e}  PSNR@{nmax} {terc[-1]["psnr"]:6.2f}  end slope {s:+.3f}')

    ro = []
    top = [band_slope(nmin, nmax, k) for k in (3, 4)]
    low = [band_slope(nmin, nmax, k) for k in (0, 1)]
    total = curve[nmax]['psnr'] - curve['allref']['psnr']
    if total < 0.1:
        ro.append(f'The full burst is worth only {total:+.2f} dB over {nmax} keyframe copies: the burst branch contributes '
                  f'~nothing at any length, so the slope says nothing about extraction.')
    elif end >= 0.5:
        ro.append(f'Still climbing at the end: {end:+.2f} dB per doubling over N {nmin}..{nmax} (2..4: {early:+.2f}). '
                  f'The branch extracts from every frame it gets: more samples per frame (k = 3, candidate 4) and per-frame '
                  f'geometry are worth pursuing.')
    elif end >= 0.2:
        ro.append(f'Slowing: {end:+.2f} dB per doubling over N {nmin}..{nmax} vs {early:+.2f} over 2..4 (N<{nmin} is out '
                  f'of distribution, so the early slope is only indicative).')
    else:
        ro.append(f'Flat at the end: {end:+.2f} dB per doubling over N {nmin}..{nmax}. Extra frames carry information the '
                  f'branch does not use -> under-extraction; more taps (candidate 4) will not help by itself -- the '
                  f'pooling / injection / x8 tail is the bottleneck (candidates 1-3).')
    ro.append(f'Over N {nmin}..{nmax} the extra frames cut error by {low[0]:+.2f} / {low[1]:+.2f} dB per doubling below one '
              f'LR pixel and {top[0]:+.2f} / {top[1]:+.2f} dB in the 2-4 / >4 F_N bands: '
              + ('they still add sub-pixel detail.' if max(top) >= 0.3 else
                 'neither moves much.' if max(low) < 0.1 else
                 'they mostly denoise -- the top bands barely move.'))
    headline = {f'psnr_N{n}': curve[n]['psnr'] for n in ns}
    headline.update({'psnr_allref': curve['allref']['psnr'], 'burst_vs_allref_db': float(total), 'slope_end_db_per_doubling': end,
                     'slope_2_4': early, 'slope_4_8': sl['4-8'], 'min_trained_N': nmin,
                     'top_band_slope_end': float(np.nanmean(top))})
    ctx.save(headline, ro, rows, extra={'curve': {str(k): v for k, v in curve.items()}, 'slopes': sl,
                                        'noise_terciles': terc})
    plot(ctx, curve, ns, nmin)


def plot(ctx, curve, ns, nmin):
    plt = C.plt()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    x = np.array(ns)
    y = np.array([curve[n]['psnr'] for n in ns])
    ci = np.array([curve[n]['ci'] for n in ns])
    ax[0].plot(x, y, 'C0o-')
    ax[0].fill_between(x, y - ci, y + ci, alpha=0.2)
    ax[0].axhline(curve['allref']['psnr'], color='C1', ls='--', label=f'all-ref ({max(ns)} keyframe copies)')
    ax[0].axvspan(0.8, nmin - 0.5, color='0.9', label='not seen in training')
    ax[0].set_xscale('log', base=2)
    ax[0].set_xticks(ns, [str(n) for n in ns])
    ax[0].set_xlabel('burst length N (true length)')
    ax[0].set_ylabel('PSNR (dB)')
    ax[0].legend(fontsize=8)
    for k, b in enumerate(C.BAND_NAMES):
        ax[1].plot(x, [curve[n]['band_db'][k] for n in ns], 'o-', ms=3, label=b)
    ax[1].set_xscale('log', base=2)
    ax[1].set_xticks(ns, [str(n) for n in ns])
    ax[1].set_ylabel(f'band error vs N={max(ns)} (dB)')
    ax[1].set_title('which error the frames remove')
    ax[1].legend(fontsize=8, title='band (F_N)')
    for k in (2, 3, 4):
        ax[2].plot(x, [curve[n]['mtf'][k] for n in ns], 'o-', ms=3, label=C.BAND_NAMES[k])
    ax[2].set_xscale('log', base=2)
    ax[2].set_xticks(ns, [str(n) for n in ns])
    ax[2].set_ylabel('MTF (median)')
    ax[2].set_title('transfer in the upper bands')
    ax[2].legend(fontsize=8, title='band (F_N)')
    fig.suptitle(f'burst_length  {ctx.opt["name"]}  {os.path.basename(ctx.ckpt)}')
    fig.tight_layout()
    fig.savefig(ctx.path('.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
