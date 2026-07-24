#!/usr/bin/env python3
"""
FusionBlock Attention Mass on Non-Reference Frames (PLAN.md L5)

For every net_g_*.pth checkpoint in a given experiment models/ directory,
this script runs inference on the full test set and measures where the
FusionBlock's cross-attention (reference queries, all frames as keys/values)
puts its probability mass:

  non_ref_mass  — total attention mass on keys belonging to non-reference
                  frames (1 - mass on the reference frame's keys)
  per-frame     — mass per source frame index (sums to 1 across frames)

A model that ignores the burst concentrates all mass on the reference frame
(non_ref_mass → 0); uniform attention would give (N-1)/N.

Implementation: a forward hook on the FusionBlock's softmax module captures
the attention matrix [B*windows, heads, P, N*P] (P = window_size^2 tokens);
key columns [f*P, (f+1)*P) belong to source frame f. Mirrors the structure of
offset_analysis.py (per-checkpoint loop over the RealBSR test set).

Outputs:
  - Log file  : analysis/outputs/ablation_logs/fusion_attention_<timestamp>.log
  - Plot (PNG): analysis/outputs/ablation_logs/fusion_attention_<timestamp>.png

Run with:
    torchrun --nproc_per_node=4 analysis/fusion_attention_mass.py \\
        --models_dir /path/to/experiments/MF_STHAT_P0.x/models \\
        [--config      main/config.yml] \\
        [--data_root   /path/to/RealBSR_RAW_testpatch] \\
        [--seed        42] \\
        [--num_frames  5] \\
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
    logger = logging.getLogger('fusion_attention_mass')
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
# Data loading (LQ only — GT not needed for attention measurement)
# ---------------------------------------------------------------------------

def load_lq(burst_dir, lq_indices):
    """Return stacked LQ frames as FloatTensor [N, 4, H, W] in [0, 1]."""
    pkl_file = glob.glob(os.path.join(burst_dir, '*.pkl'))[0]
    with open(pkl_file, 'rb') as f:
        meta = pickle.load(f)

    subtract_bl = not meta.get('black_level_subtracted', False)
    lq_paths    = sorted(glob.glob(os.path.join(burst_dir, '*_x1_*.png')))

    frames = []
    for idx in lq_indices:
        img   = cv2.imread(lq_paths[idx], cv2.IMREAD_UNCHANGED)
        frame = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        if subtract_bl:
            frame = frame - 512.0
        frame = frame / 16383.0
        frames.append(frame)

    return torch.stack(frames, dim=0)


def normal_indices(count=5, total_lq=14, seed=None):
    """Same frame-selection logic as BurstImageDataset._generate_lq_indices."""
    rng    = random.Random(seed)
    others = rng.sample(range(1, total_lq), count - 1)
    others.insert(count // 2, 0)
    return others


# ---------------------------------------------------------------------------
# Attention hook
# ---------------------------------------------------------------------------

class AttentionMassAccumulator:
    """Collects per-call attention mass per source frame from the FusionBlock
    softmax output [B*windows, heads, P, N*P]."""

    def __init__(self, num_frames):
        self.num_frames = num_frames
        self._per_frame = []          # list of np arrays [N]
        self._handle = None

    def register(self, model):
        fusion_block = model.fusion.fusion_block

        def hook(module, inp, output):
            attn = output.detach()                      # [B_, heads, P, N*P]
            n = self.num_frames
            p = attn.shape[-1] // n
            # mass per source frame: sum over that frame's key columns,
            # averaged over batch, heads and query positions
            mass = attn.view(*attn.shape[:-1], n, p).sum(dim=-1)   # [B_, heads, P, N]
            self._per_frame.append(mass.mean(dim=(0, 1, 2)).float().cpu().numpy())

        self._handle = fusion_block.softmax.register_forward_hook(hook)

    def clear(self):
        self._per_frame.clear()

    def burst_mean(self):
        """Mean per-frame mass vector [N] over the calls since last clear()."""
        return np.mean(np.stack(self._per_frame, axis=0), axis=0)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Inference pass for one checkpoint
# ---------------------------------------------------------------------------

def run_checkpoint(model, all_burst_dirs, accumulator, args, device, rank, world_size):
    """Run inference on this rank's subset of bursts.

    Returns list of (name, per_frame_mass[N]) — one entry per burst.
    """
    local_dirs = all_burst_dirs[rank::world_size]
    results    = []

    for i, burst_dir in enumerate(local_dirs):
        global_idx = rank + i * world_size
        indices    = normal_indices(count=args.num_frames, total_lq=14,
                                    seed=args.seed + global_idx)
        lq = load_lq(burst_dir, indices)

        accumulator.clear()

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _ = model(lq.unsqueeze(0).to(device))

        results.append((os.path.basename(burst_dir), accumulator.burst_mean()))

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def save_plot(iters, nonref_means, nonref_stds, num_frames, output_path):
    iters = np.array(iters)
    m = np.array(nonref_means)
    s = np.array(nonref_stds)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(iters, m, marker='o', color='#1f77b4', label='mass on non-reference frames')
    ax.fill_between(iters, m - s, m + s, alpha=0.15, color='#1f77b4')
    uniform = (num_frames - 1) / num_frames
    ax.axhline(uniform, color='#2ca02c', linestyle='--',
               label=f'uniform attention ({uniform:.3f})')
    ax.axhline(0.0, color='#d62728', linestyle=':',
               label='reference-only (single-image SR)')
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('FusionBlock attention mass on non-ref frames', fontsize=12)
    ax.set_title('FusionBlock non-reference attention mass vs. Training Iteration', fontsize=13)
    ax.set_ylim(-0.02, 1.0)
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
    parser.add_argument('--data_root',
                        default='/groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_testpatch',
                        help='Root directory of test burst folders')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed for frame selection')
    parser.add_argument('--num_frames', type=int, default=5)
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
    log_path  = os.path.join(log_dir, f'fusion_attention_{timestamp}.log')
    logger    = setup_logger(log_path, rank)

    # --- Architecture config ---
    config_path = (args.config if os.path.isabs(args.config)
                   else os.path.join(repo_root, args.config))
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    net_opt            = opt['network_g']
    net_opt['is_train'] = False

    ref_idx = args.num_frames // 2

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
        logger.info(f'Frames        : {args.num_frames}  (reference slot {ref_idx})')
        logger.info(f'Uniform mass  : non-ref would be {(args.num_frames - 1) / args.num_frames:.4f}')
        logger.info(f'Seed          : {args.seed}\n')

    # Build model once; reload state dict per checkpoint
    model       = MambaFusionNet(**net_opt).to(device)
    model.eval()
    accumulator = AttentionMassAccumulator(args.num_frames)
    accumulator.register(model)

    iters_list   = []
    nonref_means = []
    nonref_stds  = []

    for ckpt_path in ckpt_paths:
        basename = os.path.basename(ckpt_path)
        iter_num = int(re.search(r'net_g_(\d+)\.pth', basename).group(1))

        ckpt  = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('params_ema', ckpt.get('params', ckpt.get('state_dict', ckpt)))
        model.load_state_dict(state, strict=True)

        local_results = run_checkpoint(model, all_dirs, accumulator, args, device, rank, world_size)

        # Gather from all ranks
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_results)

        if rank == 0:
            flat = [r for rank_res in gathered for r in rank_res]
            per_frame = np.stack([r[1] for r in flat], axis=0)      # [n_bursts, N]
            nonref    = 1.0 - per_frame[:, ref_idx]

            m, s = float(nonref.mean()), float(nonref.std())
            frame_means = per_frame.mean(axis=0)

            frame_str = '  '.join(
                f'f{j}{"*" if j == ref_idx else ""}={frame_means[j]:.4f}'
                for j in range(args.num_frames))
            logger.info(f'iter {iter_num:>7d} | non_ref_mass={m:.5f}±{s:.5f} | {frame_str} '
                        f'({per_frame.shape[0]} bursts)')

            iters_list.append(iter_num)
            nonref_means.append(m)
            nonref_stds.append(s)

        torch.cuda.empty_cache()

    if rank == 0:
        SEP = '=' * 72
        logger.info('\n' + SEP)
        logger.info('  FUSIONBLOCK NON-REFERENCE ATTENTION MASS SUMMARY')
        logger.info('  (mass on keys of non-reference frames; * marks the reference slot)')
        logger.info(SEP)
        logger.info(f'  {"Iter":>8}  {"non-ref mean":>13}  {"std":>9}')
        logger.info('  ' + '-' * 36)
        for it, m, s in zip(iters_list, nonref_means, nonref_stds):
            logger.info(f'  {it:>8d}  {m:>13.5f}  {s:>9.5f}')
        logger.info(SEP)

        plot_path = os.path.join(log_dir, f'fusion_attention_{timestamp}.png')
        save_plot(iters_list, nonref_means, nonref_stds, args.num_frames, plot_path)
        logger.info(f'\nPlot → {plot_path}')
        logger.info(f'Log  → {log_path}')

    accumulator.remove()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
