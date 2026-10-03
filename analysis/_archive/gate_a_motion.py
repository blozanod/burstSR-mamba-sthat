#!/usr/bin/env python3
"""
Gate-A: Phase-correlation inter-frame camera motion measurement.

Measures real inter-frame camera motion across the full RealBSR dataset
(train + test splits) using phase cross-correlation in the packed (half-res)
RAW domain. Compares measured motion to the model's predicted offset scale
and prints an automated verdict.

Usage:
    python analysis/gate_a_motion.py [options]

Options:
    --config       Path to YAML config (default: main/config.yml)
    --train-root   Train split root dir (overrides config)
    --test-root    Test split root dir  (overrides config)
    --limit N      Process only first N bursts (smoke test)
    --num-workers  Parallel workers (default: cpu_count())
    --window / --no-window  2D Hann window before correlation (default: on)
    --out-dir      Output directory for CSV, PNG, log (default: analysis/outputs/gate_a)
"""

import os
import sys
import glob
import argparse
import csv
import logging
import multiprocessing as mp
from datetime import datetime

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.registration import phase_cross_correlation


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, 'main', 'config.yml')

# Global flag set per-worker by pool initializer — avoids repeated arg passing
_USE_WINDOW = True


def _init_worker(use_window):
    global _USE_WINDOW
    _USE_WINDOW = use_window
    cv2.setNumThreads(0)


def luma4(f):
    """Equal-weight luma from packed RGGB frame. f: [4, H, W] float32."""
    return 0.25 * f[0] + 0.25 * f[1] + 0.25 * f[2] + 0.25 * f[3]


def hann2d(h, w):
    """2D separable Hann window of shape [h, w]."""
    return np.hanning(h).reshape(-1, 1) * np.hanning(w).reshape(1, -1)


def process_burst(burst_dir):
    """
    Measure motion between reference (index 0) and all other LQ frames.

    Returns:
        (burst_dir, list_of_(slot, mag, max_mag), None)  — success
        (burst_dir, None, error_str)                     — skipped
    """
    paths = sorted(glob.glob(os.path.join(burst_dir, '*_x1_*.png')))
    if len(paths) < 2:
        return burst_dir, None, f'Skipped {burst_dir}: only {len(paths)} LQ frame(s)'

    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            return burst_dir, None, f'Skipped {burst_dir}: unreadable file {os.path.basename(p)}'
        # [H/2, W/2, 4] -> [4, H/2, W/2] float32; no black-level or normalisation needed
        frames.append(img.astype(np.float32).transpose(2, 0, 1))

    ref = frames[0]
    _, h, w = ref.shape
    max_mag = 0.45 * min(h, w)

    ref_luma = luma4(ref)
    win = None
    if _USE_WINDOW:
        win = hann2d(h, w)
        ref_luma = ref_luma * win

    results = []
    for slot, other in enumerate(frames[1:], start=1):
        other_luma = luma4(other)
        if _USE_WINDOW:
            other_luma = other_luma * win
        try:
            shift, _, _ = phase_cross_correlation(ref_luma, other_luma, upsample_factor=16)
            mag = float(np.hypot(*shift))
        except Exception:
            mag = float('nan')
        results.append((slot, mag, max_mag))

    return burst_dir, results, None


def main():
    parser = argparse.ArgumentParser(
        description='Gate-A: inter-frame camera motion via phase correlation.'
    )
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help='YAML config path (default: main/config.yml)')
    parser.add_argument('--train-root', default=None,
                        help='Train split root (overrides config)')
    parser.add_argument('--test-root', default=None,
                        help='Test split root (overrides config)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N bursts (smoke test)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Worker processes (default: cpu_count())')
    parser.add_argument('--window', dest='window', action='store_true', default=True,
                        help='Apply 2D Hann window (default: on)')
    parser.add_argument('--no-window', dest='window', action='store_false',
                        help='Disable Hann window')
    parser.add_argument('--out-dir', default=None,
                        help='Output dir for CSV, PNG, log (default: analysis/)')
    args = parser.parse_args()

    # --- Resolve config ---
    config_path = args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_root = args.train_root or cfg['datasets']['train']['dataroot']
    test_root  = args.test_root  or cfg['datasets']['val']['dataroot']
    out_dir    = args.out_dir    or os.path.join(REPO_ROOT, 'analysis', 'outputs', 'gate_a')
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = os.path.join(out_dir, f'gate_a_motion_{timestamp}.log')
    csv_path  = os.path.join(out_dir, 'gate_a_magnitudes.csv')
    hist_path = os.path.join(out_dir, 'gate_a_hist.png')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger()

    n_workers = args.num_workers or mp.cpu_count()
    log.info(f'Train root  : {train_root}')
    log.info(f'Test root   : {test_root}')
    log.info(f'Hann window : {args.window}')
    log.info(f'Workers     : {n_workers}')

    # --- Collect burst directories ---
    train_bursts = sorted(glob.glob(os.path.join(train_root, '*')))
    test_bursts  = sorted(glob.glob(os.path.join(test_root,  '*')))
    all_bursts   = train_bursts + test_bursts

    if args.limit:
        all_bursts = all_bursts[:args.limit]
        log.info(f'[smoke test] Limited to {len(all_bursts)} bursts')
    else:
        log.info(f'Total bursts: {len(all_bursts)}  ({len(train_bursts)} train + {len(test_bursts)} test)')

    # --- Parallel processing ---
    rows       = []   # (burst_name, slot, mag)
    n_skipped  = 0
    n_discarded = 0

    with mp.Pool(n_workers, initializer=_init_worker, initargs=(args.window,)) as pool:
        for burst_dir, results, err in tqdm(
            pool.imap_unordered(process_burst, all_bursts, chunksize=8),
            total=len(all_bursts),
            desc='Bursts',
            unit='burst',
        ):
            burst_name = os.path.basename(burst_dir)
            if results is None:
                log.warning(err)
                n_skipped += 1
                continue
            for slot, mag, max_mag in results:
                if np.isnan(mag) or mag > max_mag:
                    n_discarded += 1
                else:
                    rows.append((burst_name, slot, mag))

    if not rows:
        log.error('No valid measurements — check data paths and file patterns.')
        return

    # --- Save raw data ---
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['burst', 'frame_slot', 'magnitude_packed_px'])
        writer.writerows(rows)
    log.info(f'Raw magnitudes saved → {csv_path}')

    # --- Statistics ---
    mags  = np.array([r[2] for r in rows], dtype=np.float64)
    slots = np.array([r[1] for r in rows], dtype=int)
    pcts  = np.percentile(mags, [10, 25, 50, 75, 90, 95])
    median = pcts[2]

    SEP = '=' * 72
    log.info(SEP)
    log.info('  GATE-A: INTER-FRAME CAMERA MOTION  (packed-RAW domain)')
    log.info(SEP)
    log.info(f'  Valid pairs         : {len(rows)}')
    log.info(f'  Discarded (NaN/OOB) : {n_discarded}')
    log.info(f'  Skipped bursts      : {n_skipped}')
    log.info('')
    log.info(f'  {"Mean":<32} {float(np.mean(mags)):>10.4f} px')
    log.info(f'  {"Std":<32} {float(np.std(mags)):>10.4f} px')
    labels = ['p10', 'p25', 'p50 (MEDIAN)  <-- headline', 'p75', 'p90', 'p95']
    for label, val in zip(labels, pcts):
        log.info(f'  {label:<32} {val:>10.4f} px')
    frac_noise = float(np.mean(mags < 0.1))
    log.info(f'  Fraction below 0.1 px (noise floor): {frac_noise:.2%}')
    log.info(SEP)

    # --- Per-frame-slot breakdown ---
    log.info('  PER-FRAME-SLOT MEDIAN (slot 1 = first non-reference frame):')
    log.info(f'  {"Slot":<8} {"N":>8} {"Median (px)":>14}')
    log.info('  ' + '-' * 34)
    for s in range(1, 14):
        mask = slots == s
        if mask.sum() == 0:
            continue
        log.info(f'  {s:<8} {int(mask.sum()):>8} {float(np.median(mags[mask])):>14.4f}')
    log.info(SEP)

    # --- Automated verdict ---
    REF_OFFSET = 0.17   # model's measured/predicted offset in packed pixels
    log.info(f'  VERDICT  (model reference offset = {REF_OFFSET:.2f} px):')
    if median >= 1.0:
        log.info(f'  Median = {median:.4f} px  —  SUBSTANTIAL REAL MOTION (>= 1 px).')
        log.info('  Model offsets do not capture it — alignment is the suspect (proceed to P0).')
    elif median <= 0.35:
        log.info(f'  Median = {median:.4f} px  —  matches reference offset (~{REF_OFFSET:.2f} px).')
        log.info('  Little real motion; alignment exonerated — pivot to fusion / capacity / loss.')
    else:
        log.info(f'  Median = {median:.4f} px  —  AMBIGUOUS (between {REF_OFFSET:.2f} px and 1 px).')
        log.info('  Recommend block-wise phase correlation as a follow-up.')
    log.info(SEP)

    # --- Histogram ---
    fig, ax = plt.subplots(figsize=(10, 5))
    clip = np.percentile(mags, 99)
    ax.hist(mags[mags <= clip], bins=200, color='steelblue', edgecolor='none', alpha=0.85)
    ax.axvline(median, color='crimson', lw=1.8, label=f'Median = {median:.3f} px')
    ax.axvline(REF_OFFSET, color='orange', lw=1.8, ls='--',
               label=f'Model offset = {REF_OFFSET:.2f} px')
    ax.set_xlabel('Magnitude (packed pixels)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Gate-A: Inter-Frame Motion Distribution (packed-RAW domain)', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()
    log.info(f'Histogram saved → {hist_path}')
    log.info(f'Log saved       → {log_path}')

    print(f'\n*** HEADLINE: Median inter-frame motion = {median:.4f} packed pixels ***\n')


if __name__ == '__main__':
    main()