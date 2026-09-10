#!/usr/bin/env python3
"""Pre-flight check for a training config: build the network, run one
forward/backward at the config's own batch size, and report peak GPU memory.

Catches shape errors and OOM in seconds instead of after a scheduler wait. The
per-arch `if __name__ == '__main__'` blocks cannot be used for this: the arch
package's __init__ auto-imports every *_arch.py, so executing one as a script
or with `python -m` registers its class twice and trips the registry assert.
This script imports the class instead.

Two model families are handled, dispatched on the config's `model_type`,
because they train at different precision and take different-shaped input:

  MambaFusionModel (burst fusion, e.g. MF_STHAT_*): lq is [B, N, 4, h, h]
  (the full burst), forward runs under bf16 autocast
  (MambaFusionModel.optimize_parameters wraps it explicitly).

  MambaIRv2Model (standalone trunk, e.g. M0_*): lq is [B, 4, h, h] (a single
  keyframe -- MambaIRv2Model.feed_data squeezes the dataset's burst axis),
  forward runs in plain fp32 (SRModel.optimize_parameters has no autocast at
  all). Probing this path in bf16 would understate its real memory use by
  roughly 2x on the activations.

Usage:
    python analysis/shape_check.py                      # default L5 Bayer config
    python analysis/shape_check.py main/configs/MF_STHAT_L5_PackedControl.yml
    python analysis/shape_check.py main/configs/M0_MambaIRv2_Keyframe.yml
    python analysis/shape_check.py <config> --batch 4   # probe a specific batch

    # Find the largest batch_size_per_gpu that fits while holding a fixed
    # effective batch (batch_size_per_gpu * num_gpu * accumulation_steps).
    # Tries accumulation_steps=1 first (descending divisors of
    # effective_batch / num_gpu), so it always prefers no accumulation over
    # matching the same effective batch a different way.
    python analysis/shape_check.py main/configs/M0_MambaIRv2_Keyframe.yml --max-batch 32
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from burstISP.utils.registry import ARCH_REGISTRY
import burstISP.archs  # noqa: F401  (populates ARCH_REGISTRY)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CFG = os.path.join(REPO_ROOT, 'main', 'configs', 'MF_STHAT_L6_FlowFusion.yml')


def human(n):
    return f'{n / 1e6:.3f}M'


def check(cfg_path, batch_override=None):
    with open(cfg_path) as f:
        opt = yaml.safe_load(f)

    net_opt = dict(opt['network_g'])
    net_type = net_opt.pop('type')
    ds = opt['datasets']['train']
    batch = batch_override if batch_override is not None else ds['batch_size_per_gpu']

    # MambaIRv2Model runs the plain standalone trunk on a single keyframe, in
    # fp32; every other model_type here is the burst-fusion path, in bf16.
    plain = opt.get('model_type') == 'MambaIRv2Model'
    img_size = net_opt['img_size']

    if plain:
        in_chans = net_opt.get('in_chans', 3)
        out_chans = net_opt.get('out_chans', 3)
        # The arch's own kwarg is `upscale`; top-level `scale` is a fallback
        # for configs that only set that (see main/configs/M0_*.yml header).
        scale = net_opt.get('upscale', opt.get('scale'))
        print(f'\n=== {os.path.basename(cfg_path)} ===')
        print(f'  arch {net_type} (standalone, fp32) | batch {batch} | keyframe only | '
              f'{img_size}x{img_size} packed -> {img_size * scale}x{img_size * scale}')
    else:
        n_frames = net_opt['num_frames']
        scale = net_opt['scale']
        out_chans = 3
        print(f'\n=== {os.path.basename(cfg_path)} ===')
        print(f'  arch {net_type} (bf16 autocast) | batch {batch} | {n_frames} frames | '
              f'{img_size}x{img_size} packed -> {img_size * scale}x{img_size * scale}')

    net = ARCH_REGISTRY.get(net_type)(**net_opt).cuda()

    total = sum(p.numel() for p in net.parameters())
    print('  parameters:')
    for name, mod in net.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n:
            print(f'    {name:<12} {human(n):>9}  {100 * n / total:5.1f}%')
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'    {"TOTAL":<12} {human(total):>9}   (trainable {human(trainable)})')

    if plain:
        lq = torch.randn(batch, in_chans, img_size, img_size, device='cuda')
    else:
        lq = torch.randn(batch, n_frames, 4, img_size, img_size, device='cuda')
    gt = torch.randn(batch, out_chans, img_size * scale, img_size * scale, device='cuda')
    expected = (batch, out_chans, img_size * scale, img_size * scale)

    torch.cuda.reset_peak_memory_stats()
    net.train()

    if plain:
        # SRModel.optimize_parameters has no autocast: plain fp32 end to end.
        out = net(lq)
    else:
        # Mirrors MambaFusionModel.optimize_parameters: bf16 autocast, float() cast,
        # plain L1 on linear RGB.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(lq)
    out = out.float()

    if tuple(out.shape) != expected:
        print(f'  FORWARD  FAIL  got {tuple(out.shape)}, expected {expected}')
        return False
    print(f'  forward  ok    {tuple(out.shape)}')

    loss = torch.nn.functional.l1_loss(out, gt)
    loss.backward()

    missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        # find_unused_parameters is false in these configs, so DDP would error here.
        print(f'  BACKWARD FAIL  {len(missing)} parameter(s) got no gradient, first few:')
        for n in missing[:5]:
            print(f'      {n}')
        return False
    print(f'  backward ok    loss {loss.item():.4f}, all parameters received gradients')

    # Aux path: the flow-supervision return. Only the burst-fusion arch
    # supports return_aux; MambaIRv2.forward has no such kwarg.
    if not plain and opt.get('train', {}).get('flow_opt'):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            _, aux = net(lq, return_aux=True)

        # Alignment runs on the Bayer grid when pre_align is on, so lv1 sits at
        # 2x the packed crop -- which is exactly the resolution the dataset's
        # flow_vectors come at, and what MambaFusionModel.flow_loss assumes.
        fac = 2 if net_opt.get('pre_align', False) else 1
        want = {'lv1': img_size * fac, 'lv2': img_size * fac // 2, 'lv3': img_size * fac // 4}
        bad = []
        for lvl, side in want.items():
            got = tuple(aux['flows'][lvl].shape)
            if got != (batch, n_frames, 2, side, side):
                bad.append(f'{lvl}: got {got}, expected {(batch, n_frames, 2, side, side)}')
        if bad:
            print('  AUX      FAIL  flow shapes do not match the supervision grid:')
            for b in bad:
                print(f'      {b}')
            return False
        print(f'  aux      ok    flows lv1 {want["lv1"]}^2 / lv2 {want["lv2"]}^2 / '
              f'lv3 {want["lv3"]}^2, GT flow_vectors are {img_size * 2}^2')

    peak = torch.cuda.max_memory_allocated() / 2**30
    total_mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f'  peak memory    {peak:.2f} GiB of {total_mem:.1f} GiB '
          f'({100 * peak / total_mem:.0f}%) at batch {batch}')
    if peak / total_mem > 0.85:
        print('  NOTE: over 85% of the card. Training allocates more than this single '
              'step does (DDP gradient buckets, optimizer state). Consider a smaller '
              'batch_size_per_gpu with accumulation_steps raised to compensate '
              '(both model families now support it -- see sr_model.py / '
              'mambafusion_model.py optimize_parameters).')
    return True


def find_max_batch(cfg_path, effective_batch):
    """Search descending divisors of effective_batch / num_gpu for the
    largest batch_size_per_gpu that fits, so the reported accumulation_steps
    is always the smallest that holds the effective batch (1 if it fits at
    all). Config's own batch_size_per_gpu is not read; only num_gpu is."""
    with open(cfg_path) as f:
        opt = yaml.safe_load(f)
    num_gpu = opt.get('num_gpu', 1)
    if effective_batch % num_gpu != 0:
        print(f'\n=== {os.path.basename(cfg_path)}: effective_batch {effective_batch} is not '
              f'divisible by num_gpu {num_gpu} ===')
        return False
    per_gpu_budget = effective_batch // num_gpu

    candidates = sorted({d for d in range(1, per_gpu_budget + 1) if per_gpu_budget % d == 0},
                        reverse=True)

    print(f'\n=== {os.path.basename(cfg_path)}: searching for max batch_size_per_gpu '
          f'(effective batch {effective_batch}, {num_gpu} GPU(s) -> per-GPU budget '
          f'{per_gpu_budget}) ===')

    for b in candidates:
        accumulation_steps = per_gpu_budget // b
        print(f'\n--- batch_size_per_gpu={b}, accumulation_steps={accumulation_steps} '
              f'(effective batch {b * num_gpu * accumulation_steps}) ---')
        try:
            ok = check(cfg_path, batch_override=b)
        except torch.cuda.OutOfMemoryError:
            print(f'  OOM at batch_size_per_gpu={b}')
            ok = False
        torch.cuda.empty_cache()
        if ok:
            print(f'\nRECOMMENDATION: batch_size_per_gpu: {b}  accumulation_steps: '
                  f'{accumulation_steps}')
            return True

    print(f'\nFAILED: no batch_size_per_gpu down to 1 fits on this GPU '
          f'(effective batch {effective_batch} may not be reachable here at all).')
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='*', default=[DEFAULT_CFG],
                    help='config YAML path(s); defaults to the L6 config')
    ap.add_argument('--batch', type=int, default=None,
                    help="override batch_size_per_gpu (probe what fits)")
    ap.add_argument('--max-batch', type=int, default=None, metavar='EFFECTIVE_BATCH',
                    help='instead of one probe, search for the largest batch_size_per_gpu '
                         'that fits while holding this effective batch fixed, and report '
                         'the accumulation_steps needed')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('ERROR: needs a GPU (DCNv4 is a CUDA extension). Run on a GPU node.')
        return 1

    ok = True
    for cfg in args.configs:
        if args.max_batch is not None:
            ok &= find_max_batch(cfg, args.max_batch)
            continue
        try:
            ok &= check(cfg, args.batch)
        except torch.cuda.OutOfMemoryError:
            print(f'  OOM at batch {args.batch or "config default"} — lower '
                  f'batch_size_per_gpu and raise accumulation_steps to compensate.')
            ok = False
        torch.cuda.empty_cache()

    print('\n' + ('ALL CHECKS PASSED' if ok else 'CHECKS FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
