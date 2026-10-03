#!/usr/bin/env python3
"""Shared LQ-burst sources for the checkpoint-forensics scripts.

offset_analysis.py and fusion_attention_mass.py only need the low-quality
burst — they measure activations, not reconstruction quality — so this module
exposes an LQ-only source with one interface over both datasets:

    source = build_lq_source('synburst', root, num_frames=14, seed=42)
    lq = source.load(i)          # FloatTensor [N, 4, H, W] in [0, 1]

Frame ordering matches what the model was trained on in each case:

  realbsr  - reference frame (index 0) at the centre slot, the remaining
             num_frames - 1 drawn at random, mirroring
             BurstImageDataset._generate_lq_indices. The draw is seeded per
             burst so repeated runs and different checkpoints see identical
             inputs.
  synburst - the official 300-burst val set, first num_frames frames in
             official order, then permuted so frame 0 lands at the centre
             slot, mirroring SyntheticBurstDataset._center_ref_order. There
             is nothing random here: the official set is a fixed benchmark.

Anything comparing measurements across the two datasets, or across the packed
and Bayer domains, should also read gt_px_per_align_px() below — a raw offset
in feature-map pixels means different physical distances in each.
"""

import glob
import os
import pickle
import random

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_ROOTS = {
    'realbsr': '/groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_testpatch',
    'synburst': '/groups/rls/blozanod/MambaFusion/dataset/SyntheticBurstVal',
}


def center_ref_order(num_frames):
    """Permutation putting source frame 0 at slot num_frames // 2.

    Identical to SyntheticBurstDataset._center_ref_order.
    """
    order = list(range(1, num_frames))
    order.insert(num_frames // 2, 0)
    return order


def normal_indices(count=5, total_lq=14, seed=None):
    """RealBSR frame selection: reference at centre, rest drawn at random.

    Identical to BurstImageDataset._generate_lq_indices and to the helper of
    the same name in burst_ablation.py.
    """
    rng = random.Random(seed)
    others = rng.sample(range(1, total_lq), count - 1)
    others.insert(count // 2, 0)
    return others


def gt_px_per_align_px(net_opt):
    """How many ground-truth pixels one alignment-grid pixel spans.

    BurstAlign runs on the packed RGGB grid when pre_align is off and on the
    Bayer grid (2x finer) when it is on, so the same numeric offset means
    different physical displacements between the two. Report offsets in GT
    pixels to compare a packed checkpoint against a Bayer one.
    """
    scale = net_opt['scale']
    return scale // 2 if net_opt.get('pre_align', False) else scale


class RealBSRLQ:
    """LQ bursts from RealBSR_RAW folders (16-bit packed RGGB + camera pkl)."""

    total_lq = 14

    def __init__(self, data_root, num_frames, seed=42):
        self.dirs = sorted(glob.glob(os.path.join(data_root, '*')))
        if not self.dirs:
            raise ValueError(f'No burst folders found under {data_root}')
        self.num_frames = num_frames
        self.seed = seed

    def __len__(self):
        return len(self.dirs)

    def name(self, i):
        return os.path.basename(self.dirs[i])

    def load(self, i):
        burst_dir = self.dirs[i]
        indices = normal_indices(count=self.num_frames, total_lq=self.total_lq,
                                 seed=self.seed + i)

        pkl_file = glob.glob(os.path.join(burst_dir, '*.pkl'))[0]
        with open(pkl_file, 'rb') as f:
            meta = pickle.load(f)
        subtract_bl = not meta.get('black_level_subtracted', False)

        lq_paths = sorted(glob.glob(os.path.join(burst_dir, '*_x1_*.png')))
        frames = []
        for idx in indices:
            img = cv2.imread(lq_paths[idx], cv2.IMREAD_UNCHANGED)
            frame = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
            if subtract_bl:
                frame = frame - 512.0
            frames.append(frame / 16383.0)

        return torch.stack(frames, dim=0)


class SynBurstLQ:
    """LQ bursts from the official pre-generated SyntheticBurstVal set."""

    def __init__(self, data_root, num_frames, seed=42):
        # Imported lazily so RealBSR-only runs do not need the DBSR toolkit.
        from burstISP.data.dbsr.synthetic_burst_val_set import SyntheticBurstVal

        self.val_set = SyntheticBurstVal(root=data_root)
        if num_frames > self.val_set.burst_size:
            raise ValueError(f'num_frames={num_frames} exceeds the official val '
                             f'burst size {self.val_set.burst_size}')
        self.num_frames = num_frames
        self.order = center_ref_order(num_frames)

    def __len__(self):
        return len(self.val_set)

    def name(self, i):
        return f'{i:04d}'

    def load(self, i):
        burst, _gt, _meta = self.val_set[i]
        return burst[:self.num_frames][self.order].float()


def build_lq_source(dataset, data_root=None, num_frames=14, seed=42):
    """Construct the LQ source for `dataset`, defaulting its root."""
    root = data_root or DEFAULT_ROOTS[dataset]
    if dataset == 'realbsr':
        return RealBSRLQ(root, num_frames, seed)
    if dataset == 'synburst':
        return SynBurstLQ(root, num_frames, seed)
    raise ValueError(f'Unknown dataset: {dataset}')
