#!/usr/bin/env python3
"""Run the four KGTSMamba diagnostics on one checkpoint and write SUMMARY.md.

    python analysis/diagnostics/run_all.py --config main/configs/M1_KGTSMamba.yml [--ckpt latest]
        [--only error_bands,burst_length] [--limit 20] [--draws 1] [--n 300]

Each script runs in its own process (a failure in one does not stop the others), all writing to
analysis/outputs/diagnostics/<name>/<checkpoint>_<weights>/. SUMMARY.md collects the headline
numbers and readout lines, plus the evidence each result gives for or against the architecture
candidates (rules of thumb with their thresholds written next to them -- read the tables).
On the cluster: main/jobs/diagnostics_job.sh. Rough cost on one A10 at M1's size: error_bands and
colour_fit a few minutes each, oracle_geometry ~10 min, burst_length ~15 min per draw.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SCRIPTS = ('error_bands', 'colour_fit', 'oracle_geometry', 'burst_length')
TITLES = {'error_bands': '1. Error by frequency band', 'colour_fit': '2. Per-image colour fit',
          'oracle_geometry': '3. Oracle geometry', 'burst_length': '4. True-length burst curve'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', default='latest')
    ap.add_argument('--weights', choices=['ema', 'raw'], default='ema')
    ap.add_argument('--exp_root', default=os.path.join(C.REPO, 'experiments'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--only', default=','.join(SCRIPTS))
    ap.add_argument('--summary_only', action='store_true', help='rewrite SUMMARY.md from existing outputs')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--routing', default='paired')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--val_root', default=None)
    ap.add_argument('--zurich_root', default=None)
    ap.add_argument('--split', default='test')
    ap.add_argument('--n', type=int, default=300, help='generated bursts for oracle_geometry')
    ap.add_argument('--draws', type=int, default=1, help='frame subsets per burst for burst_length')
    ap.add_argument('--ns', default='1-14', help='burst lengths for burst_length')
    args = ap.parse_args()

    opt = C.load_opt(args.config)
    ckpt = C.resolve_ckpt(opt, args.ckpt, args.exp_root)
    out = args.out or os.path.join(C.REPO, 'analysis', 'outputs', 'diagnostics', opt['name'],
                                   f'{os.path.splitext(os.path.basename(ckpt))[0]}_{args.weights}')
    os.makedirs(out, exist_ok=True)
    print(f'run_all: {opt["name"]}  {ckpt}  -> {out}', flush=True)

    status = {}
    if not args.summary_only:
        common = ['--config', args.config, '--ckpt', ckpt, '--weights', args.weights, '--out', out,
                  '--batch', str(args.batch), '--workers', str(args.workers), '--routing', args.routing,
                  '--seed', str(args.seed)] + (['--amp'] if args.amp else []) + (['--cpu'] if args.cpu else []) \
            + (['--limit', str(args.limit)] if args.limit else [])
        val = ['--val_root', args.val_root] if args.val_root else []
        zur = (['--zurich_root', args.zurich_root] if args.zurich_root else []) + ['--split', args.split, '--n', str(args.n)]
        extra = {'error_bands': val, 'colour_fit': val, 'oracle_geometry': zur,
                 'burst_length': val + ['--draws', str(args.draws), '--ns', args.ns]}
        for s in [s for s in args.only.split(',') if s]:
            t = time.time()
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{s}.py')] + common + extra[s]
            print(f'\n=== {s}: {" ".join(cmd)}', flush=True)
            rc = subprocess.run(cmd).returncode
            status[s] = (rc, (time.time() - t) / 60)
            print(f'=== {s}: exit {rc} after {status[s][1]:.1f} min', flush=True)
    write_summary(out, opt, ckpt, args, status)
    sys.exit(0 if all(rc == 0 for rc, _ in status.values()) else 1)


def load(out, s):
    p = os.path.join(out, f'{s}.json')
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def evidence(res):
    """Per candidate: (for, against) lists of strings. Thresholds are rules of thumb."""
    eb, cf, og, bl = (res.get(s) for s in SCRIPTS)
    ev = {k: ([], []) for k in ('c1', 'c2', 'c3', 'c4', 'c5')}
    f = lambda k, s: ev[k][0].append(s)
    a = lambda k, s: ev[k][1].append(s)
    if eb:
        h = eb['headline']
        bands = {d['band']: d for d in eb.get('bands', [])}
        detail = h['high_share'] >= 0.5 and h['atten_share_above_2FN'] >= 0.5 and h['edge_hi_concentration'] >= 2
        noise = h['high_share'] >= 0.5 and h['atten_share_above_2FN'] < 0.5
        mtf_gain = max((bands[b]['mtf'] - bands[b]['mtf_allref']) for b in ('2-4', '>4')) if bands else float('nan')
        tag = (f'[bands: high {h["high_share"]:.0%} of MSE, {h["atten_share_above_2FN"]:.0%} attenuation >2F_N, '
               f'edge conc {h["edge_hi_concentration"]:.1f}x]')
        if detail:
            for k in ('c1', 'c2', 'c3', 'c4'):
                f(k, f'error is missing detail at edges {tag}')
        if noise:
            for k in ('c1', 'c2', 'c3', 'c4'):
                a(k, f'high-band error is mostly additive (noise / artifacts), not missing detail {tag}')
        if h['low_share'] > 0.5 or h['chroma_share'] > 0.4:
            for k in ('c1', 'c2', 'c3', 'c4', 'c5'):
                a(k, f'low-band / chroma error dominates (low {h["low_share"]:.0%}, chroma {h["chroma_share"]:.0%}): '
                     f'a body colour / denoising problem')
        if mtf_gain == mtf_gain and h['burst_gain_db'] >= 0.1:
            if mtf_gain < 0.05:
                f('c3', f'the burst barely lifts the transfer above 2 F_N (MTF +{mtf_gain:.2f} vs all-ref): '
                        f'used as a denoiser, sub-pixel content not separated')
                f('c2', f'burst detail does not reach the output above 2 F_N (MTF +{mtf_gain:.2f} vs all-ref)')
            else:
                f('c1', f'the burst lifts the transfer above 2 F_N (MTF +{mtf_gain:.2f}): the detail is extracted, '
                        f'so rendering it at x8 is where the rest goes')
        mix = eb.get('aux_mix') or []
        best = max((d['db_global'] for d in mix if d['band'] in ('1-2', '2-4', '>4')), default=None)
        if best is not None:
            (f if best >= 0.05 else a)('c2', f'mixing the burst-only aux output into the result: {best:+.3f} dB '
                                             f'best band >= 1 F_N (threshold +0.05)')
    if cf:
        g = cf['headline']['gain_affine_cv']
        if g >= 0.2:
            for k in ('c1', 'c2', 'c3', 'c4', 'c5'):
                a(k, f'{g:+.2f} dB is plain global colour/tone error (colour_fit), which none of these fixes')
    if og:
        h = og['headline']
        g, share = h['oracle_gain_db'], h['tail_share_of_gain']
        if g >= 0.1:
            f('c5', f'perfect geometry is worth {g:+.2f} dB' + (f', {share:.0%} of it from the tail' if share == share else ''))
            (f if share >= 0.6 else a)('c5', f'tail share of the oracle gain {share:.0%} (>= 60%: fix the tail; '
                                             f'else the bulk precision)')
        elif g < 0.05:
            a('c5', f'perfect geometry is worth only {g:+.3f} dB')
            f('c4', f'tap positions are already good enough ({g:+.3f} dB from the oracle): more taps would read '
                    f'correctly placed samples')
    if bl:
        h = bl['headline']
        s, top = h['slope_end_db_per_doubling'], h['top_band_slope_end']
        n0 = h.get('min_trained_N', 8)
        if h.get('burst_vs_allref_db', 1.0) < 0.1:
            s = top = float('nan')                      # the burst adds nothing: no slope evidence
        if s >= 0.5:
            f('c4', f'PSNR still climbs {s:+.2f} dB per doubling at N {n0}..14: more samples pay')
            f('c5', f'every frame counts ({s:+.2f} dB/doubling at the end), so each misplaced one costs')
        elif s < 0.2:
            a('c4', f'PSNR is flat at N {n0}..14 ({s:+.2f} dB/doubling): more samples are not the bottleneck')
            for k in ('c1', 'c2', 'c3'):
                f(k, f'flat burst curve ({s:+.2f} dB/doubling at N {n0}..14): information is there but not extracted')
        if top == top and top < 0.3:
            f('c3', f'extra frames barely reduce the >2 F_N error ({top:+.2f} dB/doubling): they only denoise')
    return ev


CANDIDATES = {'c1': 'QMambaBSR AdaUp upsampler (reconstruction tail)',
              'c2': 'projected burst features injected at HR once at the end',
              'c3': 'wavelet-conditioned scan (BurstMamba)',
              'c4': 'k = 3 taps',
              'c5': 'fix the geometry tail'}


def write_summary(out, opt, ckpt, args, status):
    res = {s: load(out, s) for s in SCRIPTS}
    L = [f'# KGTS diagnostics: {opt["name"]} @ {os.path.basename(ckpt)} ({args.weights})', '']
    meta = next((r['meta'] for r in res.values() if r), {})
    L.append(f'{meta.get("date", "")} - commit {meta.get("commit", "?")} - {meta.get("device", "?")} - '
             f'routing {meta.get("routing", "?")} - {"bf16" if meta.get("amp") else "fp32"}')
    L.append('')
    for s in SCRIPTS:
        L.append(f'## {TITLES[s]}')
        r = res[s]
        if s in status and status[s][0] != 0:
            L.append(f'**FAILED** (exit {status[s][0]}); see the job log.')
        if not r:
            L += ['(no result)', '']
            continue
        L.append('')
        L.append('| | |')
        L.append('|---|---|')
        for k, v in r['headline'].items():
            L.append(f'| {k} | {v:.4g} |' if isinstance(v, (int, float)) else f'| {k} | {v} |')
        L.append('')
        for line in r['readout']:
            L.append(f'- {line}')
        L.append('')
        L.append(f'Full report: `{s}.txt`, per image: `{s}_per_image.csv`, plot: `{s}.png`')
        L.append('')
    L.append('## Candidates: evidence for / against')
    L.append('')
    L.append('Rules of thumb (thresholds in the text); a candidate with no lines has no evidence either way here.')
    L.append('')
    ev = evidence(res)
    for k, name in CANDIDATES.items():
        fo, ag = ev[k]
        L.append(f'**{k[1]}. {name}**')
        L += [f'- for: {x}' for x in fo] + [f'- against: {x}' for x in ag]
        if not fo and not ag:
            L.append('- (nothing either way)')
        L.append('')
    path = os.path.join(out, 'SUMMARY.md')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nSUMMARY -> {path}')


if __name__ == '__main__':
    main()
