#!/usr/bin/env python3
"""
Alignment Displacement Analysis Across Training Checkpoints

For every net_g_*.pth checkpoint in a given experiment models/ directory,
this script runs inference on the full test set and measures the mean
absolute displacement BurstAlign estimates at each pyramid level:

  lv3  — coarsest, resolution H/4 x W/4, correlation flow head
  lv2  — mid,      resolution H/2 x W/2, flow2  = up(flow3) + residual
  lv1  — finest,   resolution H   x W,   flow1b = flow1 + cascade residual

These come from the model's aux return (`MambaFusionNet.forward(...,
return_aux=True)`), not from a forward hook on a projection layer, and they
are in *pixels at that level's resolution* by construction.

That matters. The pre-L6 version of this script hooked the three
offset_proj layers and pulled out (Δx, Δy) on the assumption that DCNv4
packs its offset tensor as (Δx, Δy, mask) per kernel point, at channels
k*3 and k*3+1. The CUDA kernel actually blocks per group as
[K*2 offsets | K masks] (see DCNv4_op/src/cuda/dcnv4_im2col_cuda.cuh:44 and
113-124), so those readings averaged a mix of offset and mask channels and
are not a measurement of displacement. Any number produced by this script
before L6 should be treated as void, including the "44 % of mean motion
compensated" figure in the L5 post-run analysis.

The levels *compose*: lv3's estimate is upsampled (x2, and doubled in
magnitude) into lv2 and again into lv1, so lv1 already contains the whole
chain. Read lv1 as the total applied displacement — do not add the rows.

Outputs:
  - Log file  : analysis/outputs/ablation_logs/offset_analysis_<timestamp>.log
  - Plot (PNG): analysis/outputs/ablation_logs/offset_analysis_<timestamp>.png

Run with:
    torchrun --nproc_per_node=4 analysis/offset_analysis.py \\
        --models_dir /path/to/experiments/STHAT_GW/models \\
        [--config      main/config_newarch.yml] \\
        [--dataset     realbsr|synburst] \\
        [--data_root   /path/to/test/set] \\
        [--seed        42] \\
        [--num_frames  N]   # defaults to config network_g.num_frames \\
        [--log_dir     analysis/outputs/ablation_logs]
"""

import os
import sys
import glob
import pickle
import random
import logging
import argparse
import re
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling: burst_data

from burstISP.archs.mambafusion_arch import MambaFusionNet
from burst_data import build_lq_source, gt_px_per_align_px, DEFAULT_ROOTS


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path, rank):
    logger = logging.getLogger('offset_analysis')
    logger.setLevel(logging.INFO)
    if rank == 0:
        fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Displacement readout
# ---------------------------------------------------------------------------

LEVELS = ('lv3', 'lv2', 'lv1')

# How many level-pixels one pixel at that level spans, relative to lv1.
LEVEL_STRIDE = {'lv1': 1, 'lv2': 2, 'lv3': 4}


def flow_mean_abs(flow):
    """Mean |displacement| over both axes, frames, batch and space.

    flow : [B, N, 2, h, w] in pixels at that level's own resolution.
    """
    return flow.detach().abs().float().mean().item()


# ---------------------------------------------------------------------------
# Inference pass for one checkpoint
# ---------------------------------------------------------------------------

def run_checkpoint(model, source, args, device, rank, world_size):
    """Run inference on this rank's subset of bursts.

    Returns list of (name, mean_lv3, mean_lv2, mean_lv1) — one entry per burst.
    """
    results = []

    for idx in range(rank, len(source), world_size):
        lq = source.load(idx)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, aux = model(lq.unsqueeze(0).to(device), return_aux=True)

        flows = aux['flows']
        results.append((source.name(idx), *(flow_mean_abs(flows[lv]) for lv in LEVELS)))

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def save_plot(iters, means, stds, output_path, gt_per_px):
    """
    iters : list[int]
    means : list[(m_lv3, m_lv2, m_lv1)]  — mean |flow| in each level's own px
    stds  : list[(s_lv3, s_lv2, s_lv1)]

    Plotted in GT pixels so the three levels are on one comparable axis and
    can be read against the protocol's actual motion (mean 12, max 24 GT px).
    """
    iters   = np.array(iters)
    fig, ax = plt.subplots(figsize=(11, 5))
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']

    labels = {
        'lv3': 'lv3  (coarsest, H/4) — correlation flow head',
        'lv2': 'lv2  (mid, H/2)',
        'lv1': 'lv1  (finest, H) — total applied displacement',
    }

    for i, lvl in enumerate(LEVELS):
        # One level pixel spans LEVEL_STRIDE[lvl] alignment px, each gt_per_px GT px.
        k = gt_per_px * LEVEL_STRIDE[lvl]
        vals = np.array([m[i] for m in means]) * k
        errs = np.array([s[i] for s in stds]) * k
        ax.plot(iters, vals, marker='o', color=colours[i], label=labels[lvl])
        ax.fill_between(iters, vals - errs, vals + errs, alpha=0.15, color=colours[i])

    ax.axhline(12.0, ls='--', lw=1, color='#666666')
    ax.axhline(24.0, ls=':',  lw=1, color='#666666')
    ax.text(iters[-1], 12.0, ' mean motion present', va='bottom', ha='right', fontsize=9, color='#666666')
    ax.text(iters[-1], 24.0, ' max motion present',  va='bottom', ha='right', fontsize=9, color='#666666')

    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Mean |displacement|  (ground-truth pixels)', fontsize=12)
    ax.set_title('BurstAlign estimated displacement vs. Training Iteration', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', required=True,
                        help='Folder containing net_g_<iter>.pth checkpoints')
    parser.add_argument('--config', default='main/config.yml',
                        help='YAML config with network_g architecture params')
    parser.add_argument('--dataset', choices=['realbsr', 'synburst'], default='realbsr',
                        help='which test set to measure on')
    parser.add_argument('--data_root', default=None,
                        help='Root directory of test burst folders')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed for frame selection')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='burst size; defaults to the config network_g.num_frames')
    parser.add_argument('--log_dir', default='analysis/outputs/ablation_logs')
    args = parser.parse_args()

    # --- Distributed ---
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    device     = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    log_dir   = (args.log_dir if os.path.isabs(args.log_dir)
                 else os.path.join(repo_root, args.log_dir))
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = os.path.join(log_dir, f'offset_analysis_{timestamp}.log')
    logger    = setup_logger(log_path, rank)

    # --- Architecture config ---
    config_path = (args.config if os.path.isabs(args.config)
                   else os.path.join(repo_root, args.config))
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    net_opt            = opt['network_g']
    net_opt['is_train'] = False

    # --- Checkpoints (sorted by iteration) ---
    ckpt_paths = sorted(
        glob.glob(os.path.join(args.models_dir, 'net_g_[0-9]*.pth')),
        key=lambda p: int(re.search(r'net_g_(\d+)\.pth', os.path.basename(p)).group(1))
    )
    if not ckpt_paths:
        if rank == 0:
            print(f'ERROR: No net_g_<iter>.pth files found in {args.models_dir}')
        dist.destroy_process_group()
        return

    # --- Dataset ---
    if args.num_frames is None:
        args.num_frames = net_opt['num_frames']
    if args.num_frames != net_opt['num_frames']:
        raise ValueError(f'--num_frames {args.num_frames} does not match the config '
                         f'network_g.num_frames {net_opt["num_frames"]}; the model '
                         f'has fixed-size temporal parameters and would misread the burst.')

    source   = build_lq_source(args.dataset, args.data_root, args.num_frames, args.seed)
    n_total  = len(source)

    # One alignment-grid (lv1) pixel is this many GT pixels. Displacements are
    # reported in both units because a packed-domain checkpoint and a
    # Bayer-domain one are not comparable in raw feature-map pixels.
    gt_per_px = gt_px_per_align_px(net_opt)

    if rank == 0:
        logger.info(f'Models dir    : {args.models_dir}')
        logger.info(f'Checkpoints   : {len(ckpt_paths)}')
        logger.info(f'Dataset       : {args.dataset}')
        logger.info(f'Data root     : {args.data_root or DEFAULT_ROOTS[args.dataset]}')
        logger.info(f'Align grid    : {"Bayer" if net_opt.get("pre_align", False) else "packed RGGB"}'
                    f'  (lv1 px = {gt_per_px} GT px, lv2 px = {gt_per_px * 2}, lv3 px = {gt_per_px * 4})')
        logger.info(f'Test bursts   : {n_total}  ({n_total // world_size}–{-(-n_total // world_size)} per GPU)')
        logger.info(f'Cost-vol r    : {net_opt.get("offset_r", 2)} lv3 px '
                    f'(+/-{net_opt.get("offset_r", 2) * gt_per_px * 4} GT px search window)')
        logger.info(f'Seed          : {args.seed}\n')

    # Build model once; reload state dict per checkpoint
    model       = MambaFusionNet(**net_opt).to(device)
    model.eval()

    iters_list  = []
    means_list  = []
    stds_list   = []

    for ckpt_path in ckpt_paths:
        basename = os.path.basename(ckpt_path)
        iter_num = int(re.search(r'net_g_(\d+)\.pth', basename).group(1))

        ckpt  = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('params_ema', ckpt.get('params', ckpt.get('state_dict', ckpt)))
        model.load_state_dict(state, strict=True)

        local_results = run_checkpoint(model, source, args, device, rank, world_size)

        # Gather from all ranks
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_results)

        if rank == 0:
            flat = [r for rank_res in gathered for r in rank_res]
            n    = len(flat)

            arrs = [np.array([r[i + 1] for r in flat]) for i in range(len(LEVELS))]

            m = tuple(a.mean() for a in arrs)
            s = tuple(a.std() for a in arrs)

            logger.info(
                f'iter {iter_num:>7d} | '
                + '  '.join(f'{lv}={m[i]:.5f}±{s[i]:.5f}' for i, lv in enumerate(LEVELS))
                + f'  ({n} bursts)'
            )

            iters_list.append(iter_num)
            means_list.append(m)
            stds_list.append(s)

        torch.cuda.empty_cache()

    if rank == 0:
        # Summary table, in each level's own pixel units
        SEP = '=' * 78
        logger.info('\n' + SEP)
        logger.info('  DISPLACEMENT SUMMARY  (mean ± std of mean|flow|, in that level\'s own px)')
        logger.info('  Each value is averaged over all bursts × all frames × all spatial positions')
        logger.info(SEP)
        logger.info('  ' + f'{"Iter":>8}' + ''.join(f'  {lv + " mean":>10}  {lv + " std":>9}' for lv in LEVELS))
        logger.info('  ' + '-' * 72)
        for it, m, sd in zip(iters_list, means_list, stds_list):
            logger.info('  ' + f'{it:>8d}' + ''.join(f'  {m[i]:>10.5f}  {sd[i]:>9.5f}'
                                                     for i in range(len(LEVELS))))
        logger.info(SEP)

        # The same numbers in ground-truth pixels, so a packed-domain checkpoint
        # and a Bayer-domain one can be compared directly, and so the rows can be
        # read against the motion the protocol actually generates.
        logger.info('')
        logger.info(SEP)
        logger.info('  THE SAME DISPLACEMENTS IN GROUND-TRUTH PIXELS')
        logger.info(f'  1 lv1 px = {gt_per_px} GT px; 1 lv2 px = {gt_per_px * 2}; 1 lv3 px = {gt_per_px * 4}')
        logger.info(SEP)
        logger.info('  ' + f'{"Iter":>8}' + ''.join(f'  {lv + " (GT px)":>13}' for lv in LEVELS))
        logger.info('  ' + '-' * 52)
        for it, m in zip(iters_list, means_list):
            logger.info('  ' + f'{it:>8d}'
                        + ''.join(f'  {m[i] * gt_per_px * LEVEL_STRIDE[lv]:>13.4f}'
                                  for i, lv in enumerate(LEVELS)))
        logger.info('  ' + '-' * 52)
        logger.info('  SyntheticBurst inter-frame translation is 12 GT px mean, up to 24 GT px '
                    '(plus <=1 deg rotation).')
        logger.info('  The levels COMPOSE — lv3 is upsampled into lv2 and again into lv1 — so')
        logger.info('  lv1 is the total applied displacement. Do not sum the rows.')
        logger.info('  lv1 far below 12 GT px means alignment is not compensating the motion')
        logger.info('  that exists; the coarse rows say at which level the estimate stalls.')
        logger.info(SEP)

        # Plot
        plot_path = os.path.join(log_dir, f'offset_analysis_{timestamp}.png')
        save_plot(iters_list, means_list, stds_list, plot_path, gt_per_px)
        logger.info(f'\nPlot → {plot_path}')
        logger.info(f'Log  → {log_path}')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
