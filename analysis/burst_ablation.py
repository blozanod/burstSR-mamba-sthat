#!/usr/bin/env python3
"""
Burst vs. Single-Image SR Ablation

Mode 'two_pass' (default) — two-pass evaluation over the full test set:
  Pass 1 (Normal):  Reference frame at center, N-1 frames randomly sampled
                    (identical to validation during training)
  Pass 2 (All-Ref): All N input frames are the reference frame

If the metrics don't differ, the model is ignoring burst frames and
effectively doing single-image SR.

Mode 'frame_drop' (PLAN.md L5) — frame-drop curves: one pass per N in
--drop_counts (default 1,2,5,9,14), where N counts the *distinct* real frames
fed to the model: the reference plus N-1 others; the remaining input slots are
filled with reference copies (so N=1 is the all-ref pass and N=num_frames is
the normal pass). The frame draw is shared across N values per burst, so the
selections are nested and the curve is not confounded by resampling. Values of
N above --num_frames are skipped (e.g. 9 and 14 for the 5-slot RealBSR models).

Datasets: --dataset realbsr (default; RealBSR_RAW folders, sRGB/linear/SSIM
metrics via the camera pkl) or --dataset synburst (official SyntheticBurstVal
set; official psnr_synburst + linear metrics, frames taken in official order).

Run with:
    torchrun --nproc_per_node=4 analysis/burst_ablation.py \\
        --model_path /path/to/net_g_35000.pth \\
        [--mode two_pass|frame_drop] \\
        [--drop_counts 1,2,5,9,14] \\
        [--dataset realbsr|synburst] \\
        [--config main/config_newarch.yml] \\
        [--data_root /path/to/RealBSR_RAW_testpatch] \\
        [--seed 42] \\
        [--crop_border 40] \\
        [--num_frames 5] \\
        [--log_dir analysis/outputs/ablation_logs]
"""

import os
import sys
import glob
import pickle
import random
import logging
import argparse
import copy
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
from burstISP.data.dbsr.synthetic_burst_val_set import SyntheticBurstVal
from burstISP.metrics.psnr_ssim import (
    calculate_psnr_srgb,
    calculate_psnr_linear,
    calculate_ssim,
    calculate_ssim_srgb,
)
from burstISP.metrics.synburst_psnr import calculate_psnr_synburst


METRIC_LABELS = {
    'realbsr': ('PSNR-sRGB  (dB)', 'PSNR-Linear (dB)', 'SSIM'),
    'synburst': ('PSNR-SynBurst(dB)', 'PSNR-Linear (dB)', 'SSIM-Linear'),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logger(log_path, rank):
    logger = logging.getLogger('burst_ablation')
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


def load_burst(burst_dir, lq_indices):
    """Load GT image, metadata, and the requested LQ frames for one RealBSR burst.

    Returns:
        lq  : FloatTensor [N, 4, H, W] in [0, 1]
        gt  : FloatTensor [3, H_gt, W_gt] in [0, 1]
        meta: dict loaded from .pkl
    """
    pkl_file = glob.glob(os.path.join(burst_dir, '*.pkl'))[0]
    with open(pkl_file, 'rb') as f:
        meta = pickle.load(f)

    subtract_bl = not meta.get('black_level_subtracted', False)

    # GT
    gt_file = glob.glob(os.path.join(burst_dir, '*_x4_rgb.png'))[0]
    gt_img = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)
    gt = torch.from_numpy(gt_img.astype(np.float32)).permute(2, 0, 1)
    if subtract_bl:
        gt = gt - 512.0
    gt = gt / 16383.0

    # LQ frames
    lq_paths = sorted(glob.glob(os.path.join(burst_dir, '*_x1_*.png')))
    frames = []
    for idx in lq_indices:
        img = cv2.imread(lq_paths[idx], cv2.IMREAD_UNCHANGED)
        frame = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        if subtract_bl:
            frame = frame - 512.0
        frame = frame / 16383.0
        frames.append(frame)

    lq = torch.stack(frames, dim=0)
    return lq, gt, meta


def normal_indices(count=5, total_lq=14, seed=None):
    """Same logic as BurstImageDataset._generate_lq_indices.

    Reference frame (index 0) is placed at the center position; the remaining
    count-1 frames are randomly drawn from [1, total_lq).
    """
    rng = random.Random(seed)
    ref_idx = 0
    others = rng.sample(range(1, total_lq), count - 1)
    others.insert(count // 2, ref_idx)
    return others


def frame_drop_indices(n_distinct, num_frames, total_lq=14, seed=None, ordered=False):
    """Frame-drop slot assignment: n_distinct real frames (the reference plus
    n_distinct - 1 others), remaining slots filled with reference copies, and
    the reference at the center slot as always.

    With ordered=False (RealBSR convention) the others are drawn with the same
    RNG stream as normal_indices, so for a fixed seed the real frames are
    nested across n_distinct values and n_distinct == num_frames reproduces
    the normal pass exactly. With ordered=True (SyntheticBurst convention,
    matching the official burst-size slicing burst[:, :n]) the others are the
    first n_distinct - 1 frames in official order.
    """
    if ordered:
        kept = list(range(1, n_distinct))
    else:
        rng = random.Random(seed)
        others = rng.sample(range(1, total_lq), num_frames - 1)
        kept = others[:max(n_distinct - 1, 0)]
    slots = kept + [0] * (num_frames - 1 - len(kept))
    slots.insert(num_frames // 2, 0)
    return slots


def run_inference(model, lq, device):
    """Forward pass. Returns output on CPU as FloatTensor [3, H_out, W_out]."""
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(lq.unsqueeze(0).to(device))
    return out.squeeze(0).float().cpu()


def compute_metrics(output, gt, meta, crop_border, dataset):
    """Three metric columns per dataset flavor (see METRIC_LABELS)."""
    if dataset == 'realbsr':
        p_srgb = calculate_psnr_srgb(output, gt, copy.deepcopy(meta), crop_border)
        p_lin = calculate_psnr_linear(output, gt, crop_border)
        ssim = calculate_ssim_srgb(output, gt, copy.deepcopy(meta), crop_border)
        return p_srgb, p_lin, ssim
    # synburst: official headline metric + linear-domain metrics (no camera pkl)
    p_syn = calculate_psnr_synburst(output, gt, boundary_ignore=crop_border)
    p_lin = calculate_psnr_linear(output, gt, crop_border)
    ssim = calculate_ssim(output, gt, crop_border, input_order='CHW')
    return p_syn, p_lin, ssim


# ---------------------------------------------------------------------------
# Dataset abstraction
# ---------------------------------------------------------------------------

class RealBSRItems:
    total_lq = 14
    ordered = False

    def __init__(self, data_root):
        self.dirs = sorted(glob.glob(os.path.join(data_root, '*')))

    def __len__(self):
        return len(self.dirs)

    def name(self, i):
        return os.path.basename(self.dirs[i])

    def load(self, i, lq_indices):
        return load_burst(self.dirs[i], lq_indices)


class SynBurstItems:
    total_lq = 14
    ordered = True

    def __init__(self, data_root):
        self.val_set = SyntheticBurstVal(root=data_root)

    def __len__(self):
        return len(self.val_set)

    def name(self, i):
        return f'{i:04d}'

    def load(self, i, lq_indices):
        burst, gt, _ = self.val_set[i]
        return burst[lq_indices], gt, None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_two_pass(logger, all_results, labels, log_path):
    n = len(all_results)

    def col_mean(key, col):
        return float(np.mean([r[key][col] for r in all_results]))

    mn = (col_mean('normal', 0), col_mean('normal', 1), col_mean('normal', 2))
    ma = (col_mean('all_ref', 0), col_mean('all_ref', 1), col_mean('all_ref', 2))

    SEP = '=' * 72
    logger.info(SEP)
    logger.info(f'  BURST vs. SINGLE-IMAGE SR ABLATION  ({n} test bursts)')
    logger.info(SEP)
    logger.info(f'  {"Metric":<22} {"Pass 1  Normal":>16} {"Pass 2  All-Ref":>16} {"Delta (N-A)":>12}')
    logger.info('  ' + '-' * 68)

    for i_col, label in enumerate(labels):
        d = mn[i_col] - ma[i_col]
        sign = '+' if d >= 0 else ''
        logger.info(f'  {label:<22} {mn[i_col]:>16.4f} {ma[i_col]:>16.4f} {sign + f"{d:.4f}":>12}')

    logger.info(SEP)
    logger.info('  Delta > 0  →  burst helps (model uses extra frames)')
    logger.info('  Delta ≈ 0  →  model defaults to single-image SR')
    logger.info(SEP + '\n')

    # Per-sample table
    logger.info(f'  {"Burst Dir":<24} {"N:m1":>8} {"A:m1":>8} {"N:m2":>8} {"A:m2":>8} {"N:m3":>7} {"A:m3":>7}')
    logger.info('  ' + '-' * 72)
    for r in sorted(all_results, key=lambda x: x['name']):
        nm, am = r['normal'], r['all_ref']
        logger.info(
            f'  {r["name"]:<24}'
            f' {nm[0]:>8.4f} {am[0]:>8.4f}'
            f' {nm[1]:>8.4f} {am[1]:>8.4f}'
            f' {nm[2]:>7.4f} {am[2]:>7.4f}'
        )

    logger.info(f'\nFull log saved to: {log_path}')


def report_frame_drop(logger, all_results, counts, labels, log_dir, timestamp, log_path):
    n = len(all_results)

    means = {}
    stds = {}
    for c in counts:
        cols = np.array([r['passes'][c] for r in all_results])  # [n, 3]
        means[c] = cols.mean(axis=0)
        stds[c] = cols.std(axis=0)

    SEP = '=' * 72
    logger.info(SEP)
    logger.info(f'  FRAME-DROP CURVES  ({n} test bursts; N = distinct real frames, '
                f'remaining slots ref-filled)')
    logger.info(SEP)
    logger.info(f'  {"N":>4}  ' + ''.join(f'{lab:>20}' for lab in labels))
    logger.info('  ' + '-' * 68)
    for c in counts:
        row = ''.join(f'{means[c][i]:>13.4f}±{stds[c][i]:<6.4f}' for i in range(3))
        logger.info(f'  {c:>4}  {row}')
    logger.info('  ' + '-' * 68)
    base = counts[0]
    for c in counts[1:]:
        deltas = means[c] - means[base]
        row = ''.join(f'{d:>+20.4f}' for d in deltas)
        logger.info(f'  Δ(N={c} − N={base}){row}')
    logger.info(SEP)

    # Curve plot: one panel per metric
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    xs = np.array(counts)
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ys = np.array([means[c][i] for c in counts])
        es = np.array([stds[c][i] for c in counts])
        ax.plot(xs, ys, marker='o', color='#1f77b4')
        ax.fill_between(xs, ys - es, ys + es, alpha=0.15, color='#1f77b4')
        ax.set_xlabel('Distinct real frames N')
        ax.set_ylabel(lab)
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.35)
    fig.suptitle('Frame-drop curves (ref-filled slots)', fontsize=13)
    fig.tight_layout()
    plot_path = os.path.join(log_dir, f'frame_drop_{timestamp}.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    logger.info(f'\nPlot → {plot_path}')
    logger.info(f'Log  → {log_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True,
                        help='Path to checkpoint (.pth)')
    parser.add_argument('--mode', choices=['two_pass', 'frame_drop'], default='two_pass',
                        help='two_pass: normal vs all-ref; frame_drop: curve over N')
    parser.add_argument('--drop_counts', default='1,2,5,9,14',
                        help='Comma-separated N values for frame_drop mode')
    parser.add_argument('--dataset', choices=['realbsr', 'synburst'], default='realbsr',
                        help='realbsr: RealBSR_RAW folders; synburst: official SyntheticBurstVal root')
    parser.add_argument('--config', default='main/config.yml',
                        help='YAML config for network architecture')
    parser.add_argument('--data_root',
                        default='/groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_testpatch',
                        help='Root directory containing burst sub-folders (or the SyntheticBurstVal root)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed for Pass 1 frame selection (per-burst seeding)')
    parser.add_argument('--crop_border', type=int, default=40,
                        help='Border pixels excluded from metric computation')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of LQ frames per burst')
    parser.add_argument('--log_dir', default='analysis/outputs/ablation_logs',
                        help='Directory for log file output')
    args = parser.parse_args()

    # --- Distributed setup ---
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    device     = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # --- Logger ---
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir   = os.path.join(repo_root, args.log_dir) if not os.path.isabs(args.log_dir) else args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = os.path.join(log_dir, f'burst_ablation_{timestamp}.log')
    logger    = setup_logger(log_path, rank)

    labels = METRIC_LABELS[args.dataset]

    if rank == 0:
        logger.info(f'Checkpoint : {args.model_path}')
        logger.info(f'Mode       : {args.mode}  |  Dataset: {args.dataset}')
        logger.info(f'Data root  : {args.data_root}')
        logger.info(f'Seed       : {args.seed}  |  Crop border: {args.crop_border}  |  Frames: {args.num_frames}')
        logger.info(f'World size : {world_size} GPU(s)')

    # --- Load architecture config ---
    config_path = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    net_opt = opt['network_g']
    net_opt['is_train'] = False

    # --- Build and load model ---
    model = MambaFusionNet(**net_opt).to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    state = ckpt.get('params_ema', ckpt.get('params', ckpt.get('state_dict', ckpt)))
    model.load_state_dict(state, strict=True)
    model.eval()

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f'Parameters : {n_params / 1e6:.3f}M')

    # --- Dataset items, split across GPUs ---
    items = RealBSRItems(args.data_root) if args.dataset == 'realbsr' else SynBurstItems(args.data_root)
    n_total = len(items)
    local_ids = list(range(n_total))[rank::world_size]

    # --- Frame-drop counts (only N <= num_frames are runnable) ---
    counts = sorted({int(c) for c in args.drop_counts.split(',')})
    runnable = [c for c in counts if 1 <= c <= args.num_frames]
    if rank == 0:
        logger.info(f'Test bursts : {n_total} total, ~{len(local_ids)} per GPU')
        if args.mode == 'frame_drop':
            skipped = [c for c in counts if c not in runnable]
            if skipped:
                logger.info(f'Frame-drop  : N={runnable} (skipping N={skipped} > num_frames={args.num_frames})')
            else:
                logger.info(f'Frame-drop  : N={runnable}')
        logger.info('Running inference...\n')

    # --- Inference ---
    local_results = []

    for i, item_id in enumerate(local_ids):
        name = items.name(item_id)
        # Deterministic per-burst seed: consistent regardless of which GPU handles it
        burst_seed = args.seed + item_id

        if args.mode == 'two_pass':
            # Pass 1 — Normal: ref at center, others sampled (realbsr) or in
            # official order (synburst)
            if items.ordered:
                idx_normal = frame_drop_indices(args.num_frames, args.num_frames,
                                                items.total_lq, ordered=True)
            else:
                idx_normal = normal_indices(count=args.num_frames, total_lq=items.total_lq,
                                            seed=burst_seed)
            lq_n, gt, meta = items.load(item_id, idx_normal)
            out_n = run_inference(model, lq_n, device)
            m_n   = compute_metrics(out_n, gt, meta, args.crop_border, args.dataset)

            # Pass 2 — All-Ref: every slot is the reference frame
            idx_all_ref = [0] * args.num_frames
            lq_a, _, _ = items.load(item_id, idx_all_ref)
            out_a = run_inference(model, lq_a, device)
            m_a   = compute_metrics(out_a, gt, meta, args.crop_border, args.dataset)

            local_results.append({'name': name, 'normal': m_n, 'all_ref': m_a})
        else:
            passes = {}
            for c in runnable:
                idx = frame_drop_indices(c, args.num_frames, items.total_lq,
                                         seed=burst_seed, ordered=items.ordered)
                lq_c, gt, meta = items.load(item_id, idx)
                out_c = run_inference(model, lq_c, device)
                passes[c] = compute_metrics(out_c, gt, meta, args.crop_border, args.dataset)
            local_results.append({'name': name, 'passes': passes})

        if (i + 1) % 20 == 0:
            print(f'[Rank {rank}] {i + 1}/{len(local_ids)} done', flush=True)

    # --- Gather all results on rank 0 ---
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_results)

    if rank == 0:
        all_results = [r for rank_res in gathered for r in rank_res]
        if args.mode == 'two_pass':
            report_two_pass(logger, all_results, labels, log_path)
        else:
            report_frame_drop(logger, all_results, runnable, labels, log_dir, timestamp, log_path)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
