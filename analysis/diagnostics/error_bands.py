#!/usr/bin/env python3
"""(1) Where does the error live? Frequency band, luma vs chroma, edges vs flat, smoothing vs noise.

    python analysis/diagnostics/error_bands.py --config main/configs/M1_KGTSMamba.yml [--ckpt latest]

Every official SyntheticBurstVal burst goes through the model three ways, paired (same routing
noise, analysis/diagnostics/common.py):
  burst   the real 14-frame burst (what validation scores)
  all-ref every frame replaced by the keyframe: the same pipeline with no new information, so
          burst vs all-ref per band says what the burst branch contributes there
  aux     (if the model has aux_head) the burst-only linear reconstruction from the first KGTS
          call's pooled states (KGTSMamba.aux)

Per image, on the official metric's crop and quantisation, the error e = output - GT is split:
  * 2 bands at about one LR pixel: energy weighted by a Gaussian low-pass's amplitude response
    (sigma 3 HR px, half amplitude at the packed Nyquist F_N = 1/16 cycles / HR px) and the rest.
    An exact partition of the MSE.
  * 5 radial bands (F_N units: <0.5, 0.5-1, 1-2, 2-4, >4; see common.py for what each means).
  * luma vs chroma (orthonormal opponent basis) x low vs high.
  * transfer vs additive, per band: E(f) = (MTF - 1) GT(f) + residual. The part of a band's error
    explained by one real gain on GT's own spectrum is attenuation (the output is too smooth:
    detail missing); the rest is additive (residual noise, demosaicking / aliasing artifacts,
    misplaced detail). MTF is that gain + 1.
  * where the high band sits: GT pixels split by luma gradient into flat (lower 50%), texture
    (50-85%) and edges (top 15%).
  * aux mixing: per band, the error left after adding alpha_b x (aux - main) to the output, with one
    alpha per band per image (an oracle; GT picks alpha) and one per band for the whole set. If the
    burst-only path improves the output in a high band, it carries detail the main path drops.
  * all of it by noise level (tercile of the burst's shot noise, from meta_info.pkl).

Reading it -- the question is "colour / denoising (the body)" vs "sub-pixel extraction and
reconstruction (the burst branch and the x8 tail)":
  - low-band or chroma error dominant                 -> colour / tone / large-scale denoising;
                                                         run colour_fit.py next
  - high-band error, ADDITIVE, spread into flat areas -> residual noise and artifacts (denoising)
  - high-band error, ATTENUATION, concentrated on edges-> detail not recovered: sub-pixel
                                                         extraction / reconstruction
  - burst MTF > all-ref MTF in the 2-4 / >4 bands     -> the burst is delivering sub-pixel detail;
    if the burst gain is all additive (noise) and MTF barely moves, it is used as a denoiser
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

FINE = np.geomspace(0.05, 11.4, 49)    # fine radial bins (F_N units) for the profile plot


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument('--val_root', default=None, help='SyntheticBurstVal root (default: config datasets.val)')
    args = ap.parse_args()
    ctx = C.Context(args, 'error_bands')
    model, dev = ctx.model, ctx.device
    has_aux = getattr(model, 'aux', None) is not None
    ds = C.ValBursts(ctx.val_root(args.val_root))
    S, fine_idx = None, None
    nb = len(C.BAND_NAMES)
    acc = {k: np.zeros(len(FINE) + 1) for k in ('main', 'allref', 'aux', 'gt', 'count')}
    rows = []

    for data in C.loader(ds, args):
        burst, gt = data['burst'], data['gt'].to(dev)
        B, Nf = burst.shape[:2]
        lq, _ = C.arrange(burst, range(1, Nf), C.ref_slot(model, Nf))
        allref = burst[:, :1].expand_as(burst).contiguous()
        out = ctx.run(torch.cat([lq, allref]), return_aux=has_aux)
        out, aux = (out[0], out[1].get('burst')) if has_aux else (out, None)
        for i in range(B):
            g = C.crop(gt[i])
            if S is None:
                S = C.Spectrum(*g.shape[-2:], dev)
                f = torch.sqrt(torch.fft.fftfreq(2 * g.shape[-2], device=dev)[:, None] ** 2
                               + torch.fft.fftfreq(2 * g.shape[-1], device=dev)[None, :] ** 2) / C.F_N
                fine_idx = torch.bucketize(f.flatten(), torch.tensor(FINE, device=dev, dtype=f.dtype))
            Xg = S.fft(g - g.mean((-2, -1), keepdim=True))
            em, ea = C.error(out[i], gt[i]), C.error(out[B + i], gt[i])
            Xm, Xa = S.fft(em), S.fft(ea)
            Pg, Em, Ea = S.bands(Xg), S.bands(Xm), S.bands(Xa)
            cm, ca = S.bands(Xm, Xg), S.bands(Xa, Xg)                       # cross terms with GT
            safe = Pg.clamp_min(1e-20)
            r = {'name': data['name'][i], 'shot': float(data['shot'][i]), 'read': float(data['read'][i]),
                 'mse': em.pow(2).mean().item(), 'mse_allref': ea.pow(2).mean().item()}
            r['psnr'], r['psnr_allref'] = C.psnr_from_mse(r['mse']), C.psnr_from_mse(r['mse_allref'])
            lo, hi = S.split(Xm).tolist()
            r['lo'], r['hi'] = lo, hi
            r['lo_allref'], r['hi_allref'] = S.split(Xa).tolist()
            Xo = S.fft(C.opponent(em))
            r['luma_lo'], r['luma_hi'] = (S.split(Xo[:1]) / 3).tolist()
            r['chroma_lo'], r['chroma_hi'] = (S.split(Xo[1:]) * 2 / 3).tolist()
            for b in range(nb):
                r[f'gt_{b}'], r[f'e_{b}'], r[f'ea_{b}'] = Pg[b].item(), Em[b].item(), Ea[b].item()
                r[f'atten_{b}'] = (cm[b] ** 2 / safe[b]).item()             # energy explained by MTF-1
                r[f'mtf_{b}'] = (1 + cm[b] / safe[b]).item()
                r[f'mtf_allref_{b}'] = (1 + ca[b] / safe[b]).item()
                r[f'chroma_{b}'] = (S.bands(Xo[1:])[b] * 2 / 3).item()
            # where the error sits in the image: spatial split with the same Gaussian
            lab = C.strata(g)
            em_lo = C.gblur(em[None], C.SPLIT_SIGMA)[0]
            ea_lo = C.gblur(ea[None], C.SPLIT_SIGMA)[0]
            hi2, lo2 = (em - em_lo).pow(2).sum(0), em_lo.pow(2).sum(0)
            hia2 = (ea - ea_lo).pow(2).sum(0)
            for s, nm in enumerate(('flat', 'tex', 'edge')):
                m = lab == s
                r[f'area_{nm}'] = m.float().mean().item()
                r[f'hi_{nm}'] = (hi2[m].sum() / hi2.sum().clamp_min(1e-20)).item()
                r[f'lo_{nm}'] = (lo2[m].sum() / lo2.sum().clamp_min(1e-20)).item()
                r[f'gain_hi_{nm}'] = 10 * math.log10(max(hia2[m].sum().item(), 1e-20) / max(hi2[m].sum().item(), 1e-20))
            if aux is not None:
                ex = C.error(aux[i], gt[i])
                Xx = S.fft(ex)
                D = Xx - Xm                                                  # spectrum of aux - main
                Ex, num, den = S.bands(Xx), S.bands(Xm, D), S.bands(D)
                r['mse_aux'] = ex.pow(2).mean().item()
                r['psnr_aux'] = C.psnr_from_mse(r['mse_aux'])
                for b in range(nb):
                    r[f'ex_{b}'], r[f'num_{b}'], r[f'den_{b}'] = Ex[b].item(), num[b].item(), den[b].item()
            rows.append(r)
            # fine radial profiles, pooled
            for k, X in (('main', Xm), ('allref', Xa), ('gt', Xg)) + ((('aux', Xx),) if aux is not None else ()):
                acc[k] += torch.bincount(fine_idx, (X.abs() ** 2).sum(0).flatten(), len(FINE) + 1).cpu().numpy()
            acc['count'] += torch.bincount(fine_idx, minlength=len(FINE) + 1).cpu().numpy()
        print(f'  {len(rows)}/{args.limit or len(ds)}', end='\r', flush=True)
    print()
    report(ctx, rows, has_aux, acc)


def report(ctx, rows, has_aux, acc):
    n, nb = len(rows), len(C.BAND_NAMES)
    col = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
    mse, mse_a = col('mse'), col('mse_allref')
    db_fix = lambda part, tot: float(np.mean(-10 * np.log10(np.clip(1 - part / tot, 1e-12, None))))
    gain = col('psnr') - col('psnr_allref')
    L = ctx.log
    L(f'{n} bursts.  PSNR {np.mean(col("psnr")):.3f} dB   all-ref {np.mean(col("psnr_allref")):.3f}   '
      f'burst gain {C.fci(gain)} dB' + (f'   aux (burst-only head) {np.mean(col("psnr_aux")):.3f}' if has_aux else ''))
    L()
    L('== 2-band split at one LR pixel (Gaussian sigma 3 HR px; pooled share of MSE | dB if that band were error-free)')
    lo, hi = col('lo'), col('hi')
    L(f'   burst   : low {lo.sum() / mse.sum():6.1%} | {db_fix(lo, mse):5.2f} dB     high {hi.sum() / mse.sum():6.1%} | {db_fix(hi, mse):5.2f} dB')
    la, ha = col('lo_allref'), col('hi_allref')
    L(f'   all-ref : low {la.sum() / mse_a.sum():6.1%} | {db_fix(la, mse_a):5.2f} dB     high {ha.sum() / mse_a.sum():6.1%} | {db_fix(ha, mse_a):5.2f} dB')
    L(f'   burst gain: low band {np.mean(10 * np.log10(la / lo)):+.2f} dB, high band {np.mean(10 * np.log10(ha / hi)):+.2f} dB')
    L()
    L('== luma / chroma x low / high (pooled share of MSE | dB if fixed)')
    for nm in ('luma', 'chroma'):
        a, b = col(f'{nm}_lo'), col(f'{nm}_hi')
        L(f'   {nm:6s}: low {a.sum() / mse.sum():6.1%} | {db_fix(a, mse):5.2f} dB     high {b.sum() / mse.sum():6.1%} | {db_fix(b, mse):5.2f} dB')
    L()
    L('== 5 radial bands (F_N = packed-LR Nyquist).  err%: pooled share of MSE.  dBfix: PSNR if the band were')
    L('   error-free.  SNR: GT power / error power, dB.  MTF: output/GT gain in the band (burst | all-ref).')
    L('   atten%: share of the band\'s error that is attenuation (missing detail), rest additive (noise,')
    L('   artifacts).  burst: all-ref / burst error energy, dB (what the burst removes there).')
    hdr = f'   {"band":>6} {"what":<28} {"GT%":>6} {"err%":>6} {"dBfix":>6} {"SNR":>6} {"MTF":>11} {"atten%":>7} {"chroma%":>8} {"burst":>6}'
    if has_aux:
        hdr += f' {"aux":>6}'
    L(hdr)
    bands = []
    gt_all = sum(col(f'gt_{b}') for b in range(nb))
    for b in range(nb):
        g, e, ea, at = col(f'gt_{b}'), col(f'e_{b}'), col(f'ea_{b}'), col(f'atten_{b}')
        d = {'band': C.BAND_NAMES[b], 'gt_share': g.sum() / gt_all.sum(), 'err_share': e.sum() / mse.sum(),
             'db_fix': db_fix(e, mse), 'snr_db': float(np.mean(10 * np.log10(g / np.maximum(e, 1e-20)))),
             'mtf': float(np.median(col(f'mtf_{b}'))), 'mtf_allref': float(np.median(col(f'mtf_allref_{b}'))),
             'atten_share': at.sum() / e.sum(), 'chroma_share': col(f'chroma_{b}').sum() / e.sum(),
             'burst_gain_db': float(np.mean(10 * np.log10(ea / np.maximum(e, 1e-20))))}
        line = (f'   {d["band"]:>6} {C.BAND_NOTES[b]:<28} {d["gt_share"]:6.1%} {d["err_share"]:6.1%} {d["db_fix"]:6.2f} '
                f'{d["snr_db"]:6.1f} {d["mtf"]:5.2f}|{d["mtf_allref"]:<5.2f} {d["atten_share"]:7.1%} {d["chroma_share"]:8.1%} '
                f'{d["burst_gain_db"]:+6.2f}')
        if has_aux:
            d['aux_vs_main_db'] = float(np.mean(10 * np.log10(col(f'ex_{b}') / np.maximum(e, 1e-20))))
            line += f' {d["aux_vs_main_db"]:+6.1f}'
        bands.append(d)
        L(line)
    if has_aux:
        L('   aux: aux-head error energy relative to the main output\'s in the band, dB (> 0: aux is worse)')
    L()
    L('== where the error sits (GT luma-gradient strata).  share of the image\'s high- / low-band error energy;')
    L('   conc = share / area (1 = spread evenly).  burst gain: all-ref / burst high-band energy in the stratum, dB')
    L(f'   {"stratum":<8} {"area":>6} {"high%":>7} {"conc":>5} {"low%":>7} {"conc":>5} {"burst gain hi":>14}')
    strata = {}
    for nm in ('flat', 'tex', 'edge'):
        a, h_, l_ = col(f'area_{nm}').mean(), col(f'hi_{nm}').mean(), col(f'lo_{nm}').mean()
        strata[nm] = {'area': a, 'hi_share': h_, 'lo_share': l_, 'hi_conc': h_ / a, 'gain_hi_db': col(f'gain_hi_{nm}').mean()}
        L(f'   {nm:<8} {a:6.1%} {h_:7.1%} {h_ / a:5.2f} {l_:7.1%} {l_ / a:5.2f} {strata[nm]["gain_hi_db"]:+14.2f}')
    L()

    mix = None
    if has_aux:
        L('== aux mixing: output + alpha_b (aux - main) per band.  dB gained with one alpha per band per image')
        L('   (oracle: GT picks it) and with one alpha per band for the whole set (alpha shown)')
        num = np.stack([col(f'num_{b}') for b in range(nb)], 1)
        den = np.stack([col(f'den_{b}') for b in range(nb)], 1)
        a_glob = -num.sum(0) / np.maximum(den.sum(0), 1e-20)
        red_img = np.where(den > 0, num ** 2 / np.maximum(den, 1e-20), 0.0)            # per-image alpha
        red_glob = -(2 * a_glob * num + a_glob ** 2 * den)                              # global alpha
        mix = []
        for b in range(nb):
            d = {'band': C.BAND_NAMES[b], 'alpha_global': float(a_glob[b]),
                 'db_oracle': db_fix(red_img[:, b], mse), 'db_global': db_fix(red_glob[:, b], mse)}
            mix.append(d)
            L(f'   {d["band"]:>6}  oracle {d["db_oracle"]:+.3f} dB   global {d["db_global"]:+.3f} dB  (alpha {d["alpha_global"]:+.3f})')
        L(f'   all bands: oracle {db_fix(red_img.sum(1), mse):+.3f} dB   global {db_fix(red_glob.sum(1), mse):+.3f} dB')
        L()

    terc = []
    shot = col('shot').tolist()
    groups = C.terciles(shot)
    if groups:
        L('== by noise level (tercile of shot noise)')
        L(f'   {"shot noise":>22} {"n":>4} {"PSNR":>7} {"burst gain":>11} {"high%":>6} {"chroma%":>8}')
        for g in groups:
            sh = [shot[i] for i in g]
            d = {'shot_lo': min(sh), 'shot_hi': max(sh), 'n': len(g), 'psnr': float(col('psnr')[g].mean()),
                 'gain': float(gain[g].mean()), 'hi_share': float(col('hi')[g].sum() / mse[g].sum()),
                 'chroma_share': float((col('chroma_lo')[g] + col('chroma_hi')[g]).sum() / mse[g].sum())}
            terc.append(d)
            L(f'   {d["shot_lo"]:.2e} .. {d["shot_hi"]:.2e} {d["n"]:4d} {d["psnr"]:7.2f} {d["gain"]:+11.2f} '
              f'{d["hi_share"]:6.1%} {d["chroma_share"]:8.1%}')
    else:
        L('   (no noise levels in meta_info.pkl: noise breakdown skipped)')

    # ---- readout (heuristics) ----
    ro = []
    hs = hi.sum() / mse.sum()
    chroma = (col('chroma_lo') + col('chroma_hi')).sum() / mse.sum()
    ro.append(f'{hs:.0%} of the MSE is above ~one LR pixel (fixing it all: +{db_fix(hi, mse):.2f} dB); '
              f'below: +{db_fix(lo, mse):.2f} dB.  Chroma is {chroma:.0%} of the MSE.')
    sub = [d for d in bands if d['band'] in ('2-4', '>4')]
    att_sub = sum(d['atten_share'] * d['err_share'] for d in sub) / max(sum(d['err_share'] for d in sub), 1e-12)
    edge = strata['edge']
    if lo.sum() / mse.sum() > 0.5 or chroma > 0.4:
        ro.append('Low-frequency / colour error dominates: the body\'s colour and large-scale denoising are the gap '
                  '(see colour_fit) -- none of the burst-side candidates targets this.')
    if hs > 0.5:
        if edge['hi_conc'] >= 2 and att_sub >= 0.5:
            ro.append(f'High-band error is concentrated on edges ({edge["hi_share"]:.0%} on {edge["area"]:.0%} of the pixels) '
                      f'and is mostly attenuation ({att_sub:.0%} above 2 F_N): detail is missing -> sub-pixel extraction / '
                      f'reconstruction (candidates 1-4).')
        elif strata['flat']['hi_share'] > strata['flat']['area'] * 0.8 and att_sub < 0.5:
            ro.append(f'High-band error is mostly additive ({1 - att_sub:.0%} above 2 F_N) and spread into flat areas '
                      f'({strata["flat"]["hi_share"]:.0%} on {strata["flat"]["area"]:.0%}): residual noise / artifacts, '
                      f'i.e. denoising, not missing detail.')
        else:
            ro.append(f'High-band error is mixed: {att_sub:.0%} attenuation above 2 F_N, edge concentration '
                      f'{edge["hi_conc"]:.1f}x. Both detail and residual noise matter.')
    dm = [d['mtf'] - d['mtf_allref'] for d in sub]
    ro.append(f'Burst vs all-ref MTF in the 2-4 / >4 F_N bands: {sub[0]["mtf_allref"]:.2f} -> {sub[0]["mtf"]:.2f}, '
              f'{sub[1]["mtf_allref"]:.2f} -> {sub[1]["mtf"]:.2f}. '
              + ('The burst adds ~nothing overall (gain < 0.1 dB): no reading.' if gain.mean() < 0.1 else
                 'The burst delivers sub-pixel detail.' if max(dm) >= 0.05 else
                 'The burst barely raises the transfer above one frame\'s Nyquist: it acts mostly as a denoiser '
                 '(under-extraction of sub-pixel content).'))
    if mix:
        best = max((d for d in mix if d['band'] in ('1-2', '2-4', '>4')), key=lambda d: d['db_global'])
        if best['db_global'] >= 0.05:
            ro.append(f'Mixing the burst-only aux reconstruction into the output gains {best["db_global"]:+.2f} dB in band '
                      f'{best["band"]} with one global alpha: the burst branch carries detail the body + x8 tail drop '
                      f'-> HR-side burst injection (candidate 2) has something to deliver.')
        else:
            ro.append('The aux reconstruction adds nothing the output lacks (global-alpha mixing < 0.05 dB in every band >= 1 F_N); '
                      'inconclusive for candidate 2 (aux is a linear decode of call 0, before the body).')

    headline = {'psnr': float(np.mean(col('psnr'))), 'psnr_allref': float(np.mean(col('psnr_allref'))),
                'burst_gain_db': float(gain.mean()), 'high_share': float(hs), 'low_share': float(lo.sum() / mse.sum()),
                'db_fix_high': db_fix(hi, mse), 'db_fix_low': db_fix(lo, mse), 'chroma_share': float(chroma),
                'atten_share_above_2FN': float(att_sub), 'edge_hi_concentration': float(edge['hi_conc'])}
    if has_aux:
        headline['psnr_aux'] = float(np.mean(col('psnr_aux')))
    ctx.save(headline, ro, rows, extra={'bands': bands, 'strata': strata, 'aux_mix': mix, 'noise_terciles': terc,
                                        'fine_profile': {k: v.tolist() for k, v in acc.items()}, 'fine_edges': FINE.tolist()})
    plot(ctx, bands, acc, strata, has_aux)


def plot(ctx, bands, acc, strata, has_aux):
    plt = C.plt()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    centers = np.sqrt(FINE[1:] * FINE[:-1])
    cnt = np.where(acc['count'][1:-1] > 0, acc['count'][1:-1], np.nan)   # empty bins: not plotted
    for k, sty in (('gt', 'k-'), ('allref', 'C1-'), ('main', 'C0-'), ('aux', 'C2--')):
        if k == 'aux' and not has_aux:
            continue
        ax[0].loglog(centers, acc[k][1:-1] / cnt, sty, label={'gt': 'GT (signal)', 'main': 'error: burst',
                                                             'allref': 'error: all-ref', 'aux': 'error: aux head'}[k])
    for e in C.BAND_EDGES:
        ax[0].axvline(e, color='0.8', lw=0.8)
    ax[0].set_xlabel('radial frequency (packed-LR Nyquist units)')
    ax[0].set_ylabel('power per frequency sample')
    ax[0].set_title('radial power spectra, pooled')
    ax[0].legend(fontsize=8)
    x = np.arange(len(bands))
    ax[1].plot(x, [d['mtf'] for d in bands], 'C0o-', label='burst')
    ax[1].plot(x, [d['mtf_allref'] for d in bands], 'C1o-', label='all-ref')
    ax[1].set_xticks(x, [d['band'] for d in bands])
    ax[1].axhline(1, color='0.7', lw=0.8)
    ax[1].set_xlabel('band (F_N units)')
    ax[1].set_title('transfer (MTF) per band, median over images')
    ax[1].legend(fontsize=8)
    w = 0.38
    ax[2].bar(x - w / 2, [d['err_share'] * (1 - d['atten_share']) for d in bands], w, label='additive', color='C3')
    ax[2].bar(x - w / 2, [d['err_share'] * d['atten_share'] for d in bands], w,
              bottom=[d['err_share'] * (1 - d['atten_share']) for d in bands], label='attenuation', color='C0')
    ax[2].bar(x + w / 2, [d['err_share'] * d['chroma_share'] for d in bands], w, label='of which chroma', color='C4', alpha=0.6)
    ax[2].set_xticks(x, [d['band'] for d in bands])
    ax[2].set_title('share of MSE per band')
    ax[2].legend(fontsize=8)
    fig.suptitle(f'error_bands  {ctx.opt["name"]}  {os.path.basename(ctx.ckpt)}  '
                 f'(edges: {strata["edge"]["hi_share"]:.0%} of high-band error on {strata["edge"]["area"]:.0%} of pixels)')
    fig.tight_layout()
    fig.savefig(ctx.path('.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
