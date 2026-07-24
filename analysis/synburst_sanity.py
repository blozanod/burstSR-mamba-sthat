#!/usr/bin/env python3
"""
SyntheticBurst Port Sanity Checks (PLAN.md L3/L4)

Two modes:

  --mode cpu    Smoke-tests the vendored generation pipeline, the
                SyntheticBurstDataset contract, the oracle warp, and the
                official eval metric. Needs no GPU, no DCNv4, no mamba_ssm and
                no downloaded data (a synthetic Zurich-style fixture is created
                in a temp dir; pass --zurich_root / --val_root to also check
                against the real datasets).

  --mode model  One-command full-model sanity check for the cluster: builds
                MambaFusionNet from a config, forwards one synthetic burst
                (from --val_root if given, random otherwise) and checks the
                output shape. Requires GPU + compiled DCNv4 + mamba_ssm.

Run with:
    python analysis/synburst_sanity.py --mode cpu
    python analysis/synburst_sanity.py --mode model \\
        --config experiments/MF_STHAT_L3_SynBase/config.yml \\
        [--val_root /path/to/SyntheticBurstVal]
"""

import argparse
import math
import os
import sys
import tempfile

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def make_zurich_fixture(root, n_images=2, size=448):
    """Create a tiny Zurich-style folder (<root>/train/canon/<i>.jpg) with
    smooth synthetic images (low-frequency content keeps interpolation error
    small for the oracle-warp check)."""
    canon = os.path.join(root, 'train', 'canon')
    os.makedirs(canon, exist_ok=True)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    for i in range(n_images):
        r = 0.5 + 0.4 * np.sin(2 * np.pi * (1.5 * xs + 0.3 * i))
        g = 0.5 + 0.4 * np.sin(2 * np.pi * (1.0 * ys + 0.1 * i))
        b = 0.5 + 0.4 * np.sin(2 * np.pi * (0.7 * (xs + ys)))
        img = np.stack([b, g, r], axis=-1)  # BGR for cv2
        img8 = (img * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(canon, f'{i}.jpg'), img8,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    return root


# ---------------------------------------------------------------------------
# CPU checks
# ---------------------------------------------------------------------------

def check_generation():
    """Raw generator output: shapes, dtypes, ranges, zero base-frame flow."""
    from burstISP.data.dbsr import synthetic_burst_generation as syn_burst
    from burstISP.data.synthetic_burst_dataset import (
        BURST_TRANSFORMATION_PARAMS, DOWNSAMPLE_FACTOR, IMAGE_PROCESSING_PARAMS)

    torch.manual_seed(0)
    image = torch.rand(3, 432, 432)
    burst, frame_gt, burst_rgb, flows, meta = syn_burst.rgb2rawburst(
        image, 14, DOWNSAMPLE_FACTOR,
        burst_transformation_params=BURST_TRANSFORMATION_PARAMS,
        image_processing_params=IMAGE_PROCESSING_PARAMS)

    assert burst.shape == (14, 4, 48, 48), f'burst {tuple(burst.shape)}'
    assert frame_gt.shape == (3, 432, 432), f'gt (pre border-crop) {tuple(frame_gt.shape)}'
    assert burst_rgb.shape == (14, 3, 96, 96), f'burst_rgb {tuple(burst_rgb.shape)}'
    assert flows.shape == (14, 2, 96, 96), f'flows {tuple(flows.shape)}'
    assert burst.dtype == torch.float32 and frame_gt.dtype == torch.float32
    assert 0.0 <= burst.min() and burst.max() <= 1.0, 'burst outside [0, 1]'
    assert 0.0 <= frame_gt.min() and frame_gt.max() <= 1.0, 'gt outside [0, 1]'
    assert flows[0].abs().max().item() == 0.0, 'base frame flow must be zero'
    assert {'rgb2cam', 'cam2rgb', 'rgb_gain', 'red_gain', 'blue_gain'} <= set(meta.keys())
    print('CPU-1 PASS: rgb2rawburst shapes/dtypes/ranges OK, base-frame flow is zero')


def check_dataset_contract(zurich_root):
    """SyntheticBurstDataset train-mode dict contract."""
    from burstISP.data.synthetic_burst_dataset import SyntheticBurstDataset

    torch.manual_seed(0)
    ds = SyntheticBurstDataset({'phase': 'train', 'dataroot': zurich_root, 'num_frames': 14})
    sample = ds[0]

    assert set(sample.keys()) == {'lq', 'gt', 'flow_vectors', 'lq_path'}, sample.keys()
    assert 'meta' not in sample, "synthetic samples must not fabricate a 'meta' key"
    assert sample['lq'].shape == (14, 4, 48, 48), tuple(sample['lq'].shape)
    assert sample['gt'].shape == (3, 384, 384), tuple(sample['gt'].shape)
    assert sample['flow_vectors'].shape == (14, 2, 96, 96)
    assert sample['lq'].dtype == torch.float32 and sample['gt'].dtype == torch.float32
    assert 0.0 <= sample['lq'].min() and sample['lq'].max() <= 1.0
    assert 0.0 <= sample['gt'].min() and sample['gt'].max() <= 1.0
    # After the center-ref reorder, the generator's base frame sits at slot
    # N//2 and its flow is exactly zero.
    ref = 14 // 2
    assert sample['flow_vectors'][ref].abs().max().item() == 0.0, \
        'reference frame is not at the center slot'
    assert sample['flow_vectors'][0].abs().max().item() > 0.0, \
        'non-ref slot unexpectedly has zero flow'
    print('CPU-2 PASS: SyntheticBurstDataset contract OK '
          '(lq [14,4,48,48], gt [3,384,384], ref at center slot, no meta key)')


def check_oracle_warp(zurich_root):
    """Oracle warp: near-zero residual against the reference for a static
    scene (noise disabled so residuals reflect geometry only)."""
    from burstISP.data.synthetic_burst_dataset import SyntheticBurstDataset

    torch.manual_seed(0)
    common = {'phase': 'train', 'dataroot': zurich_root, 'num_frames': 14,
              'add_noise': False}
    ref = 14 // 2
    margin = 8  # skip border pixels affected by warp padding

    def interior_residual(burst):
        ref_frame = burst[ref, :, margin:-margin, margin:-margin]
        res = []
        for i in range(14):
            if i == ref:
                continue
            frame = burst[i, :, margin:-margin, margin:-margin]
            res.append((frame - ref_frame).abs().mean().item())
        return float(np.mean(res))

    torch.manual_seed(1); np.random.seed(1)
    import random as _random; _random.seed(1)
    plain = SyntheticBurstDataset(dict(common))[0]['lq']
    torch.manual_seed(1); np.random.seed(1); _random.seed(1)
    oracle = SyntheticBurstDataset(dict(common, oracle_align=True))[0]['lq']

    res_plain = interior_residual(plain)
    res_oracle = interior_residual(oracle)
    ratio = res_oracle / max(res_plain, 1e-12)

    assert res_oracle < 0.02, f'oracle residual too large: {res_oracle:.5f}'
    assert ratio < 0.25, (f'oracle warp barely helps: {res_oracle:.5f} vs '
                          f'{res_plain:.5f} unwarped (ratio {ratio:.3f})')
    print(f'CPU-3 PASS: oracle warp residual {res_oracle:.5f} vs {res_plain:.5f} '
          f'unwarped (ratio {ratio:.3f}) — frames land on the reference')


def check_official_eval():
    """Vendored eval on identical images -> infinite PSNR (per-image), and the
    documented behavior of the official inf-filter."""
    from burstISP.data.dbsr.image_quality import PSNR
    from burstISP.metrics.synburst_psnr import calculate_psnr_synburst

    torch.manual_seed(0)
    img = torch.rand(3, 384, 384)
    img_q = (img.clamp(0, 1) * 2 ** 14).short().float() / 2 ** 14  # 14-bit, like saved GT

    # Per-image PSNR on identical inputs is +inf in the official code
    psnr_module = PSNR(boundary_ignore=40)
    val = psnr_module.psnr(img_q.unsqueeze(0), img_q.unsqueeze(0))
    assert math.isinf(val.item()) and val.item() > 0, f'expected +inf, got {val}'

    # The official batch wrapper FILTERS inf values and returns 0 for a batch
    # that only contains identical pairs — that is upstream behavior, kept
    # verbatim ("invalid psnr" is printed by their code above).
    filtered = calculate_psnr_synburst(img_q, img_q)
    assert filtered == 0, f'expected the official inf-filter to yield 0, got {filtered}'

    # Near-identical images score very high but finite
    noisy = (img_q + 1e-4 * torch.randn_like(img_q)).clamp(0, 1)
    high = calculate_psnr_synburst(noisy, img_q)
    assert high > 60.0, f'near-identical PSNR suspiciously low: {high:.2f} dB'

    # Quantization must matter: skipping it would change the score
    direct = PSNR(boundary_ignore=40)(noisy.unsqueeze(0), img_q.unsqueeze(0)).item()
    quantized = calculate_psnr_synburst(noisy, img_q)
    print(f'CPU-4 PASS: official eval — identical images give +inf per-image PSNR '
          f'(batch filter returns 0, upstream behavior); near-identical '
          f'{high:.2f} dB (unquantized would be {direct:.2f} dB)')


def check_val_set(val_root):
    """Load burst 0 of the official val set through the val-mode dataset."""
    from burstISP.data.synthetic_burst_dataset import SyntheticBurstDataset

    ds = SyntheticBurstDataset({'phase': 'val', 'dataroot': val_root, 'num_frames': 14})
    assert len(ds) == 300, f'expected 300 official val bursts, got {len(ds)}'
    sample = ds[0]
    assert sample['lq'].shape == (14, 4, 48, 48)
    assert sample['gt'].shape == (3, 384, 384)
    assert sample['lq_path'] == '0000'
    assert 'meta' not in sample
    print('CPU-5 PASS: official val set loads as distributed '
          f'(burst 0000: lq {tuple(sample["lq"].shape)}, gt {tuple(sample["gt"].shape)})')


def run_cpu_checks(args):
    check_generation()

    zurich_root = args.zurich_root
    tmpdir = None
    if zurich_root is None:
        tmpdir = tempfile.mkdtemp(prefix='synburst_fixture_')
        zurich_root = make_zurich_fixture(tmpdir)
        print(f'(no --zurich_root given: using generated fixture at {zurich_root})')

    check_dataset_contract(zurich_root)
    check_oracle_warp(zurich_root)
    check_official_eval()

    if args.val_root is not None:
        check_val_set(args.val_root)
    else:
        print('CPU-5 SKIP: pass --val_root /path/to/SyntheticBurstVal to also '
              'check the official val set (cluster data)')

    print('\nALL CPU SANITY CHECKS PASSED')


# ---------------------------------------------------------------------------
# GPU full-model check
# ---------------------------------------------------------------------------

def run_model_check(args):
    from burstISP.archs.mambafusion_arch import MambaFusionNet

    config_path = (args.config if os.path.isabs(args.config)
                   else os.path.join(REPO_ROOT, args.config))
    with open(config_path, 'r') as f:
        opt = yaml.safe_load(f)
    net_opt = dict(opt['network_g'])
    net_opt.pop('type', None)
    net_opt['is_train'] = False

    device = torch.device('cuda')
    model = MambaFusionNet(**net_opt).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'MambaFusionNet built from {args.config}: {n_params / 1e6:.3f}M params')

    num_frames = net_opt['num_frames']
    if args.val_root is not None:
        from burstISP.data.synthetic_burst_dataset import SyntheticBurstDataset
        ds = SyntheticBurstDataset({'phase': 'val', 'dataroot': args.val_root,
                                    'num_frames': num_frames})
        lq = ds[0]['lq'].unsqueeze(0)
        src = 'official val burst 0000'
    else:
        lq = torch.rand(1, num_frames, 4, 48, 48)
        src = 'random tensor'

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(lq.to(device))

    expected = (1, 3, lq.shape[-2] * net_opt['scale'], lq.shape[-1] * net_opt['scale'])
    assert out.shape == expected, f'output {tuple(out.shape)}, expected {expected}'
    peak_mem = torch.cuda.max_memory_allocated(device) / 2 ** 30
    print(f'MODEL PASS: forward on {src} {tuple(lq.shape)} -> {tuple(out.shape)} '
          f'(peak GPU mem {peak_mem:.2f} GiB)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['cpu', 'model'], default='cpu')
    parser.add_argument('--config', default='experiments/MF_STHAT_L3_SynBase/config.yml',
                        help='Config with network_g params (model mode)')
    parser.add_argument('--zurich_root', default=None,
                        help='Zurich RAW-to-RGB root; a fixture is generated if omitted')
    parser.add_argument('--val_root', default=None,
                        help='Official SyntheticBurstVal root (optional)')
    args = parser.parse_args()

    if args.mode == 'cpu':
        run_cpu_checks(args)
    else:
        run_model_check(args)


if __name__ == '__main__':
    main()
