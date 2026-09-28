#!/usr/bin/env python3
"""KGTSMamba pre-launch sanity check. One GPU, ~10-20 min. Run before the first full job.

    python analysis/kgts_sanity.py --config main/configs/M1_KGTSMamba.yml [--iters 1500]

Three stages, each printing PASS/FAIL (or numbers to eyeball):

1. scan   -- KGTS on CUDA (mamba_ssm selective_scan_fn, grouped B, both directions
             stacked) vs the pure-torch ref_scan on CPU: forward AND gradients. Only
             the reference path was tested when KGTS was written; this is the first
             time the kernel's grouped-B backward runs.
2. memory -- one bf16 fwd+bwd at the config's batch_size_per_gpu on a random burst,
             with use_checkpoint as configured and forced on. Peak GB per GPU.
3. overfit -- a fixed batch from the real training generator, trained with the
             config's recipe (bf16, L1 + flow curriculum at its t=0 weight). What to
             look for, every --log_every iters:
               loss / psnr  keep falling / rising (psnr on the batch should reach
                            well past where M0 sits on val, ~40+ dB, within 1-2k it)
               |W_c|        leaves 0 in the first few iters -> KGTS started injecting
               inject       mean ||s_out - s_in|| / ||s_in|| over the KGTS calls;
                            should rise from 0 and stay non-trivial
               epe          lv1 flow end-point error vs the generator (packed px)
             then, on the same batch: PSNR with the real burst vs with every frame
             replaced by the keyframe ("all-ref"). A clear drop means KGTS output
             matters. It is only a smoke signal -- the real burst test is
             analysis/burst_ablation.py --dataset synburst on a trained checkpoint.
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import burstISP.archs  # noqa: F401  (populates ARCH_REGISTRY)
from burstISP.archs import build_network
from burstISP.archs.KGTSMamba import kgts_arch
from burstISP.archs.KGTSMamba.kgts_arch import KGTS
from burstISP.data.synthetic_burst_dataset import SyntheticBurstDataset
from burstISP.utils.options import ordered_yaml

DEV = 'cuda'


def psnr(a, b):
    mse = F.mse_loss(a.float().clamp(0, 1), b.float().clamp(0, 1))
    return -10 * math.log10(max(mse.item(), 1e-12))


def stage_scan(net_opt):
    """KGTS with the config's shapes, small P: kernel vs reference, fp32."""
    kg = dict(net_opt.get('kgts', {}))
    kg.pop('d', None); kg.pop('ds', None)
    d, ds = net_opt.get('token', {}).get('d', 16), net_opt['embed_dim']
    torch.manual_seed(0)
    k_cpu = KGTS(ds, d, **kg)
    for p in (k_cpu.W_c.weight, k_cpu.gamma.weight, k_cpu.beta.weight):
        torch.nn.init.normal_(p, std=0.05)          # leave the identity init, or grads are trivial
    k_gpu = KGTS(ds, d, **kg).to(DEV)
    k_gpu.load_state_dict(k_cpu.state_dict())

    P, L = 64, 14 * net_opt.get('token', {}).get('k', 2) ** 2
    x, s = torch.randn(P, L, d), torch.randn(P, ds)
    valid = torch.rand(P, L) > 0.1

    outs, grads = [], []
    for k, dev in ((k_cpu, 'cpu'), (k_gpu, DEV)):
        xs, ss = x.to(dev).requires_grad_(), s.to(dev).requires_grad_()
        out = k(ss, k.precompute(xs, valid.to(dev)))
        out.pow(2).mean().backward()
        outs.append(out.detach().cpu())
        grads.append({'x': xs.grad.cpu(), 's': ss.grad.cpu(),
                      **{n: p.grad.cpu() for n, p in k.named_parameters() if p.grad is not None}})
    assert grads[1]['x'].abs().sum() > 0, 'no gradient reached the tokens'

    rel = lambda a, b: ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    worst = max((rel(grads[1][n], grads[0][n]), n) for n in grads[0])
    fwd = rel(outs[1], outs[0])
    ok = fwd < 1e-3 and worst[0] < 1e-2
    print(f'[scan]    fwd rel err {fwd:.2e}   worst grad rel err {worst[0]:.2e} ({worst[1]})   '
          f'{"PASS" if ok else "FAIL"}')
    return ok


def stage_memory(net_opt, batch, n_frames, lq_hw):
    for ckpt in sorted({bool(net_opt.get('use_checkpoint', False)), True}):
        net = build_network({**net_opt, 'use_checkpoint': ckpt}).to(DEV).train()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        burst = torch.rand(batch, n_frames, 4, *lq_hw, device=DEV)
        try:
            t = time.time()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out, aux = net(burst, return_aux=True)
            (out.float().mean() + sum(f.float().mean() for f in aux['flows'].values())).backward()
            torch.cuda.synchronize()
            print(f'[memory]  batch {batch}  use_checkpoint={ckpt}:  peak '
                  f'{torch.cuda.max_memory_allocated() / 2**30:.2f} GB  ({time.time() - t:.1f}s fwd+bwd)')
        except torch.cuda.OutOfMemoryError:
            print(f'[memory]  batch {batch}  use_checkpoint={ckpt}:  OOM')
        del net
        torch.cuda.empty_cache()


def stage_overfit(opt, net_opt, iters, batch, log_every):
    ds_opt = dict(opt['datasets']['train'])
    ds = SyntheticBurstDataset(ds_opt)
    torch.manual_seed(0)
    samples = [ds[i] for i in torch.randperm(len(ds))[:batch].tolist()]
    lq = torch.stack([s['lq'] for s in samples]).to(DEV)
    gt = torch.stack([s['gt'] for s in samples]).to(DEV)
    flow_gt = torch.stack([s['flow_vectors'] for s in samples]).to(DEV)
    B, N = lq.shape[:2]

    net = build_network(net_opt).to(DEV).train()
    tr = opt['train']
    optim = torch.optim.AdamW(net.parameters(), lr=tr['optim_g']['lr'], betas=tr['optim_g']['betas'])
    lam = tr.get('flow_lambda', {}).get('values', [0.0])[0] if tr.get('flow_opt') else 0.0
    clip = opt['datasets']['train'].get('grad_clip_norm', tr.get('grad_clip_norm', 1.0))

    ratios = []
    def hook(mod, inp, out):
        s = inp[0].detach().float()
        ratios.append(((out.detach().float() - s).norm() / s.norm().clamp_min(1e-12)).item())
    net.kgts.register_forward_hook(hook)

    def flow_terms(flows):
        gt_ = flow_gt.reshape(B * N, 2, *flow_gt.shape[-2:]).float()
        loss, epe = 0, None
        for lv, f in flows.items():
            f = f.reshape(B * N, 2, *f.shape[-2:]).float()
            h, w = f.shape[-2:]
            target = F.adaptive_avg_pool2d(gt_, (h, w)) * (h / gt_.shape[-2])
            loss = loss + torch.sqrt((f - target) ** 2 + 1e-6).mean()
            if lv == 'lv1':   # report in packed px whatever grid lv1 lives on
                epe = ((f - target).norm(dim=1).mean() * (lq.shape[-2] / h)).item()
        return loss / len(flows), epe

    print(f'[overfit] {B} fixed bursts, {iters} iters, lr {tr["optim_g"]["lr"]}, flow lambda {lam}')
    t0 = time.time()
    for it in range(1, iters + 1):
        ratios.clear()
        optim.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out, aux = net(lq, return_aux=True)
        out = out.float()
        l_pix = F.l1_loss(out, gt)
        l_flow, epe = flow_terms(aux['flows'])
        (l_pix + lam * l_flow).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optim.step()
        if not math.isfinite(l_pix.item()):
            print(f'[overfit] non-finite loss at iter {it}: FAIL')
            return
        if it % log_every == 0 or it == 1:
            print(f'  it {it:5d}  l1 {l_pix.item():.5f}  psnr {psnr(out, gt):6.2f}  '
                  f'|W_c| {net.kgts.W_c.weight.norm().item():.4f}  '
                  f'inject {sum(ratios) / len(ratios):.2e}  epe {epe:.3f}  '
                  f'({(time.time() - t0) / it:.2f}s/it)')

    net.eval()
    with torch.no_grad():
        real = psnr(net(lq), gt)
        ref = N // 2 if net.align.ref_idx is None else net.align.ref_idx % N
        allref = psnr(net(lq[:, ref:ref + 1].expand_as(lq).contiguous()), gt)
    print(f'[overfit] eval on the batch: real burst {real:.2f} dB   all-ref burst {allref:.2f} dB   '
          f'burst gain {real - allref:+.2f} dB')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='main/configs/M1_KGTSMamba.yml')
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=2, help='overfit batch size')
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--skip', default='', help='comma list of stages to skip: scan,memory,overfit')
    args = ap.parse_args()
    assert torch.cuda.is_available(), 'needs a GPU (the point is to exercise the CUDA scan)'
    assert kgts_arch.selective_scan_fn is not None, 'mamba_ssm is not importable'

    with open(args.config) as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    net_opt = dict(opt['network_g'])
    net_opt.pop('type')
    net_opt = {'type': 'KGTSMamba', **net_opt}
    skip = set(filter(None, args.skip.split(',')))

    if 'scan' not in skip:
        stage_scan({k: v for k, v in net_opt.items() if k != 'type'})
    if 'memory' not in skip:
        size = net_opt.get('img_size', 48)
        stage_memory(net_opt, opt['datasets']['train']['batch_size_per_gpu'],
                     opt['datasets']['train'].get('num_frames', 14), (size, size))
    if 'overfit' not in skip:
        stage_overfit(opt, net_opt, args.iters, args.batch, args.log_every)


if __name__ == '__main__':
    main()
