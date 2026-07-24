import glob
import os
import random

import cv2
import torch
import torch.nn.functional as F
from torch.utils import data as data

from burstISP.data.dbsr import processing_utils as prutils
from burstISP.data.dbsr import synthetic_burst_generation as syn_burst
from burstISP.data.dbsr.data_format_utils import npimage_to_torch
from burstISP.data.dbsr.synthetic_burst_val_set import SyntheticBurstVal
from burstISP.utils.registry import DATASET_REGISTRY

# Official SyntheticBurst protocol constants, taken from the DBSR toolkit
# (train_settings/dbsr/default_synthetic.py). See burstISP/data/dbsr/README.md.
CROP_SZ = (384, 384)
DOWNSAMPLE_FACTOR = 4
BURST_TRANSFORMATION_PARAMS = {'max_translation': 24.0,
                               'max_rotation': 1.0,
                               'max_shear': 0.0,
                               'max_scale': 0.0,
                               'border_crop': 24}
IMAGE_PROCESSING_PARAMS = {'random_ccm': True, 'random_gains': True, 'smoothstep': True,
                           'gamma': True, 'add_noise': True}


@DATASET_REGISTRY.register()
class SyntheticBurstDataset(data.Dataset):
    """Standard SyntheticBurst protocol dataset (PLAN.md L3/L4).

    Emits the same dict contract as BurstImageDataset:
        lq  - [num_frames, 4, 48, 48] packed RGGB RAW burst, float32 in [0, 1]
        gt  - [3, 384, 384] linear RGB ground truth, float32 in [0, 1]
        lq_path - identifier string (used by the val loop for naming)
    plus, in train mode:
        flow_vectors - [num_frames, 2, 96, 96] dense flows describing the
        *generated* burst geometry at LR-RGB resolution (2x the packed
        resolution), in LR-RGB pixel units, relative to the generator's base
        frame (channel 0 = x/columns, channel 1 = y/rows). They are exposed
        regardless of oracle_align for future alignment-supervision work, and
        always describe the pre-warp geometry.

    There is deliberately no 'meta' key: synthetic bursts have no RealBSR
    camera pkl, and the val loop treats 'meta' as optional.

    Frame ordering: the DBSR generator (and the official val set) place the
    reference frame at index 0, while this repo's architecture takes the
    reference at index num_frames // 2. The burst (and flows) are therefore
    reordered so the reference lands at the center slot, mirroring what
    BurstImageDataset does for RealBSR. This only permutes the model's input
    order; the frame content and the ground truth are untouched.

    opt keys:
        phase       - 'train' or 'val'
        dataroot    - train: Zurich RAW-to-RGB root (expects <root>/<split>/canon/*.jpg)
                      val:   official SyntheticBurstVal root (expects bursts/ and gt/)
        num_frames  - burst size (default 14; val must be <= 14)
        split       - Zurich split subfolder for train mode (default 'train')
        oracle_align- if true (train mode only), warp every frame to the
                      reference with the generator's ground-truth flows before
                      returning the burst (PLAN.md L4 oracle arm). Rejected in
                      val mode: the official val set does not ship flows.
        add_noise   - override the official add_noise=True (default true;
                      intended for smoke tests only, not for benchmark runs)
    """

    def __init__(self, opt):
        super(SyntheticBurstDataset, self).__init__()
        self.opt = opt
        self.phase = opt['phase']
        self.data_root = opt['dataroot']
        self.num_frames = opt.get('num_frames', 14)
        self.oracle_align = opt.get('oracle_align', False)

        if self.phase == 'train':
            split = opt.get('split', 'train')
            self.image_paths = sorted(
                glob.glob(os.path.join(self.data_root, split, 'canon', '*.jpg')),
                key=lambda p: self._numeric_key(p))
            if len(self.image_paths) == 0:
                raise ValueError(f'No Zurich RAW-to-RGB images found under '
                                 f'{os.path.join(self.data_root, split, "canon")}')
            self.image_processing_params = dict(IMAGE_PROCESSING_PARAMS)
            if not opt.get('add_noise', True):
                self.image_processing_params['add_noise'] = False
        else:
            if self.oracle_align:
                raise ValueError('oracle_align is not supported in val mode: the official '
                                 'SyntheticBurstVal set does not ship ground-truth flows.')
            self.val_set = SyntheticBurstVal(root=self.data_root)
            if self.num_frames > self.val_set.burst_size:
                raise ValueError(f'num_frames={self.num_frames} exceeds the official val '
                                 f'burst size {self.val_set.burst_size}')

    @staticmethod
    def _numeric_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    def __len__(self):
        if self.phase == 'train':
            return len(self.image_paths)
        return len(self.val_set)

    def __getitem__(self, index):
        if self.phase == 'train':
            return self._get_train_sample(index)
        return self._get_val_sample(index)

    def _get_train_sample(self, index):
        path = self.image_paths[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        frame = npimage_to_torch(img, normalize=True, input_bgr=True)  # [3, H, W] in [0, 1]

        # Official augmentation: random horizontal flip of the source image
        if random.random() < 0.5:
            frame = frame.flip(2)

        # Official processing: pad the crop by 2 * border_crop, random-crop,
        # generate the burst, then border-crop the ground truth
        # (data/processing.py SyntheticBurstProcessing upstream).
        border_crop = BURST_TRANSFORMATION_PARAMS.get('border_crop', 0)
        crop_sz = [c + 2 * border_crop for c in CROP_SZ]
        frame_crop = prutils.random_resized_crop(frame, crop_sz)

        burst, frame_gt, _, flow_vectors, _ = syn_burst.rgb2rawburst(
            frame_crop, self.num_frames, DOWNSAMPLE_FACTOR,
            burst_transformation_params=BURST_TRANSFORMATION_PARAMS,
            image_processing_params=self.image_processing_params,
            interpolation_type='bilinear')

        if border_crop > 0:
            frame_gt = frame_gt[:, border_crop:-border_crop, border_crop:-border_crop]

        if self.oracle_align:
            burst = self.oracle_warp(burst, flow_vectors)

        order = self._center_ref_order(self.num_frames)
        burst = burst[order]
        flow_vectors = flow_vectors[order]

        rel_path = os.path.relpath(path, self.data_root)
        return {'lq': burst.float(),
                'gt': frame_gt.float(),
                'flow_vectors': flow_vectors.float(),
                'lq_path': rel_path}

    def _get_val_sample(self, index):
        burst, gt, meta_info = self.val_set[index]

        burst = burst[:self.num_frames]
        order = self._center_ref_order(self.num_frames)
        burst = burst[order]

        return {'lq': burst.float(),
                'gt': gt.float(),
                'lq_path': meta_info['burst_name']}

    @staticmethod
    def _center_ref_order(num_frames):
        """Move the generator's reference frame (index 0) to the center slot
        num_frames // 2, mirroring BurstImageDataset._generate_lq_indices."""
        order = list(range(1, num_frames))
        order.insert(num_frames // 2, 0)
        return order

    @staticmethod
    def oracle_warp(burst, flow_vectors):
        """Warp every frame onto the reference frame using the generator's
        ground-truth flows (PLAN.md L4 oracle arm).

        args:
            burst        - [N, 4, h, w] packed RGGB burst (reference at index 0)
            flow_vectors - [N, 2, 2h, 2w] flows at LR-RGB resolution, in LR-RGB
                           pixel units (see class docstring)
        returns:
            [N, 4, h, w] burst with every frame resampled onto the reference
            frame's grid (frame 0 has zero flow and passes through unchanged
            up to interpolation).

        Each packed channel is a complete single-color image at packed
        resolution, so all four channels are warped with the same flow field
        converted to packed units: one packed pixel covers a 2x2 LR-RGB block,
        so the flow is average-pooled 2x and halved.
        """
        n, c, h, w = burst.shape
        flow_packed = F.avg_pool2d(flow_vectors, kernel_size=2) * 0.5  # [N, 2, h, w]

        ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                                torch.arange(w, dtype=torch.float32), indexing='ij')
        base_x = xs.unsqueeze(0).expand(n, -1, -1)
        base_y = ys.unsqueeze(0).expand(n, -1, -1)

        # The content seen at reference pixel p sits at p - flow(p) in frame i
        # (flow_vectors = sample_pos_inv_i - sample_pos_inv_ref).
        sample_x = base_x - flow_packed[:, 0]
        sample_y = base_y - flow_packed[:, 1]

        grid_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
        grid_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)  # [N, h, w, 2]

        return F.grid_sample(burst, grid, mode='bilinear', padding_mode='zeros',
                             align_corners=True)
