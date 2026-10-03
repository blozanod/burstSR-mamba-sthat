#!/usr/bin/env python3
"""Measure a config's per-sample MACs with forward hooks, one forward at the
training crop (batch 1).

Counts nn.Conv{1,2,3}d, nn.ConvTranspose{1,2,3}d and nn.Linear -- every
multiply-accumulate in a pure conv net such as RCAN3D, so its number here is
exact. It does NOT count attention matmuls, selective scans, grid_sample or
DCN sampling, so for MambaFusionNet it is a lower bound; use
analysis/_archive/fusion_cost_model.py there.

Only registered submodules are hooked. RCAN3D's frozen RAFT aligner is
deliberately not a registered submodule, so it runs (the forward needs its
flow) but is excluded from the count, as it is from parameters().

Usage:
    python analysis/count_macs.py main/configs/R3D_RCAN3D_RAFT.yml
    python analysis/count_macs.py <config> --budget 80     # exit 1 if over
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from burstISP.utils.registry import ARCH_REGISTRY
import burstISP.archs  # noqa: F401  (populates ARCH_REGISTRY)

CONVS = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--budget', type=float, default=None, help='GMACs; exit 1 if the count exceeds it')
    args = ap.parse_args()

    with open(args.config) as f:
        opt = yaml.safe_load(f)
    net_opt = dict(opt['network_g'])
    net_type = net_opt.pop('type')
    img = net_opt['img_size']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = ARCH_REGISTRY.get(net_type)(**net_opt).to(device).eval()
    if opt.get('model_type') == 'MambaIRv2Model':
        lq = torch.rand(1, net_opt.get('in_chans', 3), img, img, device=device)
    else:
        lq = torch.rand(1, net_opt['num_frames'], 4, img, img, device=device)

    macs = defaultdict(int)

    def hook(name):
        def fn(mod, inp, out):
            if isinstance(mod, nn.Linear):
                macs[name] += out.numel() * mod.in_features
            elif mod.transposed:
                # Each *input* element is scattered over the kernel.
                macs[name] += inp[0].numel() * (mod.out_channels // mod.groups) * math.prod(mod.kernel_size)
            else:
                macs[name] += out.numel() * (mod.in_channels // mod.groups) * math.prod(mod.kernel_size)
        return fn

    handles = []
    for name, mod in net.named_modules():
        if isinstance(mod, CONVS + (nn.Linear,)):
            top = name.split('.')[0]
            handles.append(mod.register_forward_hook(hook(top)))

    with torch.no_grad():
        out = net(lq)
    for h in handles:
        h.remove()

    total = sum(macs.values())
    params = sum(p.numel() for p in net.parameters())
    print(f'\n=== {os.path.basename(args.config)} ===')
    print(f'  {net_type} | input {tuple(lq.shape)} -> output {tuple(out.shape)} | batch 1\n')
    for name, n in sorted(macs.items(), key=lambda kv: -kv[1]):
        print(f'    {name:<18} {n / 1e9:9.3f} GMACs  {100 * n / total:5.1f}%')
    print(f'    {"TOTAL":<18} {total / 1e9:9.3f} GMACs')
    print(f'    {"params":<18} {params / 1e6:9.3f} M (trainable, registered modules only)')

    if args.budget is not None:
        ok = total / 1e9 <= args.budget
        print(f'\n  budget {args.budget:.1f} GMACs: {"OK" if ok else "OVER"}')
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
