#!/usr/bin/env python3
"""Two correctness checks for BurstAlign's flow plumbing.

Both were scratch code in burstISP/archs/dummy.py; they are here because they
verify claims the whole L6 alignment rewrite rests on, and because they should
exercise the shipped methods rather than copies that can drift from them.

  1. --scatter  (needs CUDA + the compiled DCNv4 extension)

     BurstAlign folds its estimated flow into the DCN's offset channels rather
     than pre-warping the features, so that the features are resampled exactly
     once. That only works if `scatter_flow` writes into the channels the CUDA
     kernel actually reads as (dx, dy).

     DCNv4 blocks its offset tensor per group as [K*2 offsets | K masks]
     (DCNv4_op/src/cuda/dcnv4_im2col_cuda.cuh:44 and 113-124), NOT as
     (dx, dy, mask) interleaved per kernel point. The pre-L6 code assumed the
     latter in two places -- BurstAlign's `offset_scale` buffer and
     offset_analysis's `_xy_mean_abs` -- so this is worth an explicit test
     rather than a reading of the .cuh.

     The check: a DCN whose mask is 1.0 on the centre kernel point and 0
     elsewhere, given a constant flow through `scatter_flow`, must reproduce
     `grid_sample` at that same displacement.

  2. --sign  (CPU; needs the Zurich RAW-to-RGB training set)

     The sign of the flow supervision target is the highest-risk line in the
     L6 build: get it backwards and alignment is trained to apply motion in the
     wrong direction, roughly doubling misregistration. The loss falls anyway,
     so nothing looks broken until PSNR comes in flat.

     The claim under test is that BurstAlign's internal flow and the
     generator's `flow_vectors` share a convention, so `flow_loss` supervises
     against +flow_vectors with no negation (the negation into DCN offset units
     lives in `scatter_flow`). The check: warping each frame by +flow with
     `BurstAlign.warp` must move it TOWARD the reference, and must agree with
     `SyntheticBurstDataset.oracle_warp`, which is the repo's reference
     implementation of the same convention.

Usage:
    python analysis/dcn_scatter_check.py --scatter
    python analysis/dcn_scatter_check.py --sign --dataroot /path/to/Zurich-RAW-to-RGB
    python analysis/dcn_scatter_check.py --scatter --sign --dataroot ...
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from burstISP.archs.arch_util import DCNv4Block
from burstISP.archs.dcn_align_arch import BurstAlign


def _build_align(groups=4, num_feat=64, device='cpu'):
    """A BurstAlign only for its index buffers and its warp/scatter methods.

    Its __init__ already asserts that idx_dx | idx_dy | idx_m partition
    range(G*K*3) exactly, so constructing it is itself part of the check.
    """
    return BurstAlign(in_channels=num_feat, num_feat=num_feat, num_frames=1,
                      offset_groups=groups, r=2).to(device)


def check_scatter(tol=2e-3):
    """scatter_flow + DCN == grid_sample at the same displacement."""
    if not torch.cuda.is_available():
        print('SKIP --scatter: no CUDA device (DCNv4 is a CUDA extension).')
        return None

    torch.manual_seed(0)
    dev = 'cuda'
    B, C, H, W, G = 1, 64, 16, 16, 4
    K = 9

    align = _build_align(groups=G, num_feat=C, device=dev)
    print(f'  index buffers partition range({G * K * 3}) ........ ok (asserted in __init__)')

    # without_pointwise so the block is a pure resampler and the comparison
    # against grid_sample is exact rather than up to two 1x1 projections.
    dcn = DCNv4Block(channels=C, kernel_size=3, pad=1, stride=1,
                     groups=G, without_pointwise=True).to(dev)

    x = torch.randn(B, C, H, W, device=dev)

    worst = 0.0
    for a, b in [(1.3, -0.7), (-2.1, 0.4), (0.0, 0.0), (3.5, 3.5)]:
        flow = torch.zeros(B, 2, H, W, device=dev)
        flow[:, 0] = a
        flow[:, 1] = b

        # Mask 1.0 on the centre kernel point of every group, 0 elsewhere, so
        # the DCN reduces to a single bilinear sample per output position.
        om = torch.zeros(B, align.offset_proj.out_channels, H, W, device=dev)
        centre = align.idx_m.view(G, K)[:, K // 2]
        om[:, centre] = 1.0

        with torch.no_grad():
            out_dcn = dcn(x, align.scatter_flow(om, flow))
            out_warp = align.warp(x, flow)

        # Trim the border: grid_sample and the DCN kernel handle out-of-range
        # samples with different conventions, and only the interior is claimed.
        pad = int(max(abs(a), abs(b))) + 2
        diff = (out_dcn - out_warp)[..., pad:-pad, pad:-pad].abs().max().item()
        worst = max(worst, diff)
        status = 'ok' if diff < tol else 'FAIL'
        print(f'  flow=({a:+.1f}, {b:+.1f})  max|dcn - grid_sample| = {diff:.3e}  {status}')

    ok = worst < tol
    print(f'  --scatter: {"PASS" if ok else "FAIL"} (worst {worst:.3e}, tol {tol:.0e})')
    return ok


def check_sign(dataroot, num_frames=14, samples=4):
    """+flow_vectors moves a frame toward the reference, and matches oracle_warp."""
    import torch.nn.functional as F
    from burstISP.data.synthetic_burst_dataset import SyntheticBurstDataset

    ds = SyntheticBurstDataset({'phase': 'train', 'dataroot': dataroot,
                                'num_frames': num_frames, 'scale': 8})
    # num_feat stays at the default: DCNv4Block asserts channels // groups is a
    # multiple of 16, and only `warp` (which is channel-agnostic) is used here.
    align = _build_align(device='cpu')
    ref = num_frames // 2

    all_ok = True
    for i in range(samples):
        d = ds[i]
        burst, flow = d['lq'], d['flow_vectors']          # [N,4,h,w], [N,2,2h,2w]

        # flow_vectors are at LR-RGB resolution (2x packed); one packed pixel
        # covers a 2x2 LR-RGB block, so pool 2x and halve -- oracle_warp's
        # conversion, and the same ratio flow_loss derives from the shapes.
        flow_packed = F.avg_pool2d(flow, kernel_size=2) * 0.5

        warped = align.warp(burst, flow_packed)
        oracle = SyntheticBurstDataset.oracle_warp(burst, flow)

        keep = [n for n in range(num_frames) if n != ref]
        r = burst[ref:ref + 1]
        # Interior only. Max translation is 24 GT px = 6 packed px, and the two
        # warps use different align_corners conventions -- both map a pixel
        # coordinate to the same sample position, but they diverge in the
        # half-pixel band at the border where padding_mode kicks in.
        c = slice(12, -12)
        before = (burst[keep][..., c, c] - r[..., c, c]).abs().mean().item()
        after = (warped[keep][..., c, c] - r[..., c, c]).abs().mean().item()
        vs_oracle = (warped - oracle)[..., c, c].abs().max().item()

        moved_closer = after < before
        agrees = vs_oracle < 1e-2
        all_ok &= moved_closer and agrees
        print(f'  burst {i}: mean|frame - ref| {before:.5f} -> {after:.5f} '
              f'({"closer" if moved_closer else "WORSE"}), '
              f'max|warp - oracle_warp| = {vs_oracle:.2e} '
              f'({"ok" if agrees else "MISMATCH"})')

    print(f'  --sign: {"PASS" if all_ok else "FAIL"}')
    if not all_ok:
        print('  A "WORSE" row means the target for flow_loss should be -flow_vectors,')
        print('  not +flow_vectors. Fix MambaFusionModel.flow_loss before training.')
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scatter', action='store_true', help='DCN offset-layout check (needs CUDA)')
    ap.add_argument('--sign', action='store_true', help='flow supervision sign check (CPU)')
    ap.add_argument('--dataroot', default=None, help='Zurich RAW-to-RGB root, for --sign')
    ap.add_argument('--num_frames', type=int, default=14)
    args = ap.parse_args()

    if not (args.scatter or args.sign):
        ap.error('pick at least one of --scatter / --sign')

    results = []
    if args.scatter:
        print('scatter_flow / DCNv4 offset layout')
        results.append(check_scatter())
    if args.sign:
        if not args.dataroot:
            ap.error('--sign needs --dataroot')
        print('flow supervision sign')
        results.append(check_sign(args.dataroot, args.num_frames))

    # A skipped check (None) is not a failure, but it is not a pass either.
    sys.exit(1 if any(r is False for r in results) else 0)


if __name__ == '__main__':
    main()
