#!/usr/bin/env python3
"""
Non-ML baseline PSNR on the SyntheticBurst benchmark: bilinear and bicubic
interpolation of the reference frame, no fusion, no learning.

Pipeline mirrors MambaFusionNet's own GlobalSkipConnection
(burstISP/archs/mambafusion_arch.py) exactly: unpack the reference frame's
packed RGGB via pixel_shuffle, demosaic with kornia's Bayer demosaic
(non-learnable), then interpolate the remaining scale factor. Only the
interpolation mode changes between the two variants.

Runs entirely on CPU (no GPU op is used anywhere; CUDA is hidden from the
process below, not merely unused).

Usage:
    python3 synburst_interp_baseline.py
    python3 synburst_interp_baseline.py --limit 30   # quick smoke test
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # hide GPUs before torch/kornia import

import argparse
import sys
import time

import torch
import torch.nn.functional as F
import kornia


def demosaic_and_upsample(packed_rggb, mode):
    """packed_rggb: [4, 48, 48] float in [0, 1] -> [3, 384, 384] linear RGB.

    Same steps as GlobalSkipConnection: pixel_shuffle(2) to unpack RGGB into
    a single-channel Bayer mosaic, kornia raw_to_rgb (CFA.BG) to demosaic,
    then interpolate the remaining scale (2 * 4 = 8, matching this repo's
    scale=8 packed convention).
    """
    x = packed_rggb.unsqueeze(0)                              # [1, 4, 48, 48]
    bayer = F.pixel_shuffle(x, 2)                              # [1, 1, 96, 96]
    rgb = kornia.color.raw_to_rgb(bayer, kornia.color.CFA.BG)  # [1, 3, 96, 96]
    out = F.interpolate(rgb, scale_factor=4, mode=mode, align_corners=False)
    return out.squeeze(0).clamp(0.0, 1.0)                      # [3, 384, 384]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_root', default='/groups/rls/blozanod/MambaFusion')
    parser.add_argument('--val_root', default='/groups/rls/blozanod/MambaFusion/dataset/SyntheticBurstVal')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process the first N bursts (smoke test)')
    args = parser.parse_args()

    sys.path.insert(0, args.repo_root)
    from burstISP.data.dbsr.synthetic_burst_val_set import SyntheticBurstVal
    from burstISP.metrics.synburst_psnr import calculate_psnr_synburst

    assert not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') == '', \
        'CUDA is visible -- refusing to continue on a supposedly CPU-only run'

    val_set = SyntheticBurstVal(root=args.val_root)
    n = len(val_set) if args.limit is None else min(args.limit, len(val_set))
    print(f'Evaluating {n} bursts from {args.val_root} on CPU\n')

    methods = ['bilinear', 'bicubic']
    scores = {m: [] for m in methods}

    t0 = time.time()
    for i in range(n):
        burst, gt, _ = val_set[i]        # burst: [14, 4, 48, 48], gt: [3, 384, 384]
        ref_frame = burst[0]             # official reference frame, no fusion

        for mode in methods:
            pred = demosaic_and_upsample(ref_frame, mode)
            psnr = calculate_psnr_synburst(pred, gt, boundary_ignore=40)
            scores[mode].append(psnr)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            print(f'  {i + 1}/{n} bursts done ({elapsed:.1f}s elapsed)')

    print(f'\nTotal time: {time.time() - t0:.1f}s\n')
    print(f'{"Method":<10} {"Mean PSNR":>10} {"Std":>8} {"Min":>8} {"Max":>8}')
    print('-' * 46)
    for mode in methods:
        vals = torch.tensor(scores[mode])
        print(f'{mode:<10} {vals.mean():>10.4f} {vals.std():>8.4f} '
              f'{vals.min():>8.4f} {vals.max():>8.4f}')


if __name__ == '__main__':
    main()
