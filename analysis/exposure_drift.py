#!/usr/bin/env python3
"""
Exposure-Drift Check Across Training Checkpoints (PLAN.md L5)

Hypothesis under test: the PSNR-linear decline over training is a *global
brightness drift* of the raw linear outputs that the auto-exposure step of the
visualization ISP (scale by 0.2/mean) masks in the sRGB metrics.

For every net_g_*.pth checkpoint in a given experiment models/ directory, this
script measures across the val set:

  mean_out   — mean intensity of the raw linear model output
  mean_gt    — mean intensity of the linear ground truth (checkpoint-independent)
  ratio      — mean_out / mean_gt per burst (1.0 = no drift)

computed inside the same border crop used by the metrics (default 40 px), per
channel and overall. Mirrors the per-checkpoint structure of
offset_analysis.py.

Outputs:
  - Log file  : analysis/outputs/ablation_logs/exposure_drift_<timestamp>.log
  - Plot (PNG): analysis/outputs/ablation_logs/exposure_drift_<timestamp>.png

Run with:
    torchrun --nproc_per_node=4 analysis/exposure_drift.py \\
        --models_dir /path/to/experiments/MF_STHAT_P0.x/models \\
        [--config      main/config.yml] \\
        [--data_root   /path/to/RealBSR_RAW_testpatch] \\
        [--seed        42] \\
        [--num_frames  5] \\
        [--crop_border 40] \\
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from burstISP.archs.mambafusion_arch import MambaFusionNet


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path, rank):
    logger = logging.getLogger('exposure_drift')
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
# Data loading
# ---------------------------------------------------------------------------

def load_burst(burst_dir, lq_indices):
    """Load GT image and the requested LQ frames for one RealBSR burst."""
    pkl_file = glob.glob(os.path.join(burst_dir, '*.pkl'))[0]
    with open(pkl_file, 'rb') as f:
        meta = pickle.load(f)

    subtract_bl = not meta.get('black_level_subtracted', False)

    gt_file = glob.glob(os.path.join(burst_dir, '*_x4_rgb.png'))[0]
    gt_img = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)
    gt = torch.from_numpy(gt_img.astype(np.float32)).permute(2, 0, 1)
    if subtract_bl:
        gt = gt - 512.0
    gt = gt / 16383.0

    lq_paths = sorted(glob.glob(os.path.join(burst_dir, '*_x1_*.png')))
    frames = []
    for idx in lq_indices:
        img = cv2.imread(lq_paths[idx], cv2.IMREAD_UNCHANGED)
        frame = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        if subtract_bl:
            frame = frame - 512.0
        frame = frame / 16383.0
        frames.append(frame)

    return torch.stack(frames, dim=0), gt


def normal_indices(count=5, total_lq=14, seed=None):
    """Same frame-selection logic as BurstImageDataset._generate_lq_indices."""
    rng    = random.Random(seed)
    others = rng.sample(range(1, total_lq), count - 1)
    others.insert(count // 2, 0)
    return others


# ---------------------------------------------------------------------------
# Per-checkpoint pass
# ---------------------------------------------------------------------------

def run_checkpoint(model, all_burst_dirs, args, device, rank, world_size):
    """Returns list of (name, mean_out[3], mean_gt[3]) — one entry per burst.

    Means are per-channel over the crop-border interior of the raw linear
    output / ground truth (no clamping, no ISP)."""
    local_dirs = all_burst_dirs[rank::world_size]
    cb = args.crop_border
    results = []

    for i, burst_dir in enumerate(local_dirs):
        global_idx = rank + i * world_size
        indices = normal_indices(count=args.num_frames, total_lq=14,
                                 seed=args.seed + global_idx)
        lq, gt = load_burst(burst_dir, indices)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(lq.unsqueeze(0).to(device))
        out = out.squeeze(0).float().cpu()

        if cb > 0:
            out_i = out[:, cb:-cb, cb:-cb]
            gt_i = gt[:, cb:-cb, cb:-cb]
        else:
            out_i, gt_i = out, gt

        mean_out = out_i.mean(dim=(1, 2)).numpy()
        mean_gt = gt_i.mean(dim=(1, 2)).numpy()
        results.append((os.path.basename(burst_dir), mean_out, mean_gt))

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def save_plot(iters, out_means, gt_mean, ratio_means, ratio_stds, output_path):
    iters = np.array(iters)
    out_m = np.array(out_means)
    r_m = np.array(ratio_means)
    r_s = np.array(ratio_stds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.plot(iters, out_m, marker='o', color='#1f77b4', label='model output (linear)')
    ax1.axhline(gt_mean, color='#2ca02c', linestyle='--',
                label=f'ground truth ({gt_mean:.4f})')
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Mean intensity (raw linear)')
    ax1.set_title('Mean output intensity vs. GT')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.35)

    ax2.plot(iters, r_m, marker='o', color='#ff7f0e', label='mean(out)/mean(gt)')
    ax2.fill_between(iters, r_m - r_s, r_m + r_s, alpha=0.15, color='#ff7f0e')
    ax2.axhline(1.0, color='#2ca02c', linestyle='--', label='no drift (1.0)')
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Exposure ratio')
    ax2.set_title('Per-burst exposure ratio')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)

    fig.suptitle('Exposure drift of raw linear outputs across training', fontsize=13)
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
    parser.add_argument('--data_root',
                        default='/groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_testpatch',
                        help='Root directory of test burst folders')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed for frame selection')
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--crop_border', type=int, default=40,
                        help='Interior region used for the intensity means '
                             '(matches the metric crop)')
    parser.add_argument('--log_dir', default='analysis/outputs/ablation_logs')
    args = parser.parse_args()

    # --- Distributed ---
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    device     = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir   = (args.log_dir if os.path.isabs(args.log_dir)
                 else os.path.join(repo_root, args.log_dir))
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = os.path.join(log_dir, f'exposure_drift_{timestamp}.log')
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
    all_dirs = sorted(glob.glob(os.path.join(args.data_root, '*')))
    n_total  = len(all_dirs)

    if rank == 0:
        logger.info(f'Models dir    : {args.models_dir}')
        logger.info(f'Checkpoints   : {len(ckpt_paths)}')
        logger.info(f'Data root     : {args.data_root}')
        logger.info(f'Test bursts   : {n_total}')
        logger.info(f'Crop border   : {args.crop_border}  |  Frames: {args.num_frames}')
        logger.info(f'Seed          : {args.seed}\n')

    # Build model once; reload state dict per checkpoint
    model = MambaFusionNet(**net_opt).to(device)
    model.eval()

    iters_list  = []
    out_list    = []   # overall mean_out per checkpoint
    ratio_means = []
    ratio_stds  = []
    gt_overall  = None

    for ckpt_path in ckpt_paths:
        basename = os.path.basename(ckpt_path)
        iter_num = int(re.search(r'net_g_(\d+)\.pth', basename).group(1))

        ckpt  = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('params_ema', ckpt.get('params', ckpt.get('state_dict', ckpt)))
        model.load_state_dict(state, strict=True)

        local_results = run_checkpoint(model, all_dirs, args, device, rank, world_size)

        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_results)

        if rank == 0:
            flat = [r for rank_res in gathered for r in rank_res]
            mean_out = np.stack([r[1] for r in flat], axis=0)   # [n, 3]
            mean_gt = np.stack([r[2] for r in flat], axis=0)    # [n, 3]

            ratios = mean_out.mean(axis=1) / np.maximum(mean_gt.mean(axis=1), 1e-8)
            out_all = float(mean_out.mean())
            gt_all = float(mean_gt.mean())
            gt_overall = gt_all  # identical across checkpoints
            ch_out = mean_out.mean(axis=0)

            logger.info(
                f'iter {iter_num:>7d} | mean_out={out_all:.5f} (R={ch_out[0]:.5f} '
                f'G={ch_out[1]:.5f} B={ch_out[2]:.5f}) | mean_gt={gt_all:.5f} | '
                f'ratio={ratios.mean():.5f}±{ratios.std():.5f} ({len(flat)} bursts)'
            )

            iters_list.append(iter_num)
            out_list.append(out_all)
            ratio_means.append(float(ratios.mean()))
            ratio_stds.append(float(ratios.std()))

        torch.cuda.empty_cache()

    if rank == 0:
        SEP = '=' * 72
        logger.info('\n' + SEP)
        logger.info('  EXPOSURE DRIFT SUMMARY  (raw linear domain, crop-border interior)')
        logger.info(f'  Ground-truth mean intensity: {gt_overall:.5f} (checkpoint-independent)')
        logger.info(SEP)
        logger.info(f'  {"Iter":>8}  {"mean_out":>10}  {"ratio":>8}  {"ratio std":>10}')
        logger.info('  ' + '-' * 42)
        for it, o, rm, rs in zip(iters_list, out_list, ratio_means, ratio_stds):
            logger.info(f'  {it:>8d}  {o:>10.5f}  {rm:>8.5f}  {rs:>10.5f}')
        logger.info(SEP)
        logger.info('  ratio drifting away from 1.0 with training → global brightness')
        logger.info('  drift; the auto-exposure ISP (0.2/mean) would mask this in sRGB.')
        logger.info(SEP)

        plot_path = os.path.join(log_dir, f'exposure_drift_{timestamp}.png')
        save_plot(iters_list, out_list, gt_overall, ratio_means, ratio_stds, plot_path)
        logger.info(f'\nPlot → {plot_path}')
        logger.info(f'Log  → {log_path}')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
