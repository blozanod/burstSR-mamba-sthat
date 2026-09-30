#!/usr/bin/env python3
"""CPU checks for KGTSMamba's optional features (docs/KGTS_ADVERSARIAL_REVIEW.md). No GPU, no mamba_ssm.

    python analysis/kgts_cpu_checks.py [--rev 236dbba] [--images DIR] [--skip bitexact,...]

1. bitexact  every option off == the KGTSMamba code at --rev (read from git): state dict,
             output, aux flows and every gradient, train mode, also with W_c / W_q / gamma
             perturbed so the scan output matters.
2. options   every option builds and runs fwd+bwd with a gradient on every parameter; the
             zero-init ones (depth_embed, untie_out, inject_first, refine, aux_head) are exact
             identities at init.
3. roles     an all-selector KGTS is exactly invariant to the order of the non-key frames.
4. fused     kgts.fused (the Triton end-state kernels, run by Triton's CPU interpreter) against
             the scan path: pooled states, KGTS output and every gradient. Skipped without triton.
5. geometry  (needs --images: a folder of natural images, e.g. Zurich's train/canon) affine_lk
             against the DBSR generator's own geometry on synthetic bursts; local_residual
             follows a moving patch and leaves the rest on the global model.

mamba_ssm is stubbed with its own pure-torch reference scan (selective_scan_ref), so the
MambaIRv2 body runs -- slowly: the checks use a tiny body. Exits 1 if anything fails.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile
import types

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)
# kgts_fused's Triton kernels run on CPU tensors through the interpreter (must be set before triton.jit)
os.environ.setdefault('TRITON_INTERPRET', '1')


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """mamba_ssm's pure-torch reference, reproduced (the stub's selective_scan_fn)."""
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B, C = B.float(), C.float()
    x = A.new_zeros((batch, dim, dstate))
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if B.dim() == 2:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    elif B.dim() == 3:
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    else:
        deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, repeat(B, 'B G N L -> B (G H) N L', H=dim // B.shape[1]), u)
    if C.dim() == 4:
        C = repeat(C, 'B G N L -> B (G H) N L', H=dim // C.shape[1])
    ys = []
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum('bdn,dn->bd', x, C) if C.dim() == 2 else
                  torch.einsum('bdn,bn->bd', x, C[:, :, i]) if C.dim() == 3 else
                  torch.einsum('bdn,bdn->bd', x, C[:, :, :, i]))
    out = torch.stack(ys, dim=2)
    out = out if D is None else out + u * rearrange(D, 'd -> d 1')
    out = (out if z is None else out * F.silu(z)).to(dtype_in)
    return (out, x) if return_last_state else out


def install_stubs():
    """mamba_ssm -> the reference scan; burstISP.archs -> a bare package (its __init__ imports
    every arch, DCNv4 and kornia included)."""
    m = types.ModuleType('mamba_ssm'); m.__path__ = []
    ops = types.ModuleType('mamba_ssm.ops'); ops.__path__ = []
    ssi = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    ssi.selective_scan_fn = ssi.selective_scan_ref = selective_scan_ref
    sys.modules.update({'mamba_ssm': m, 'mamba_ssm.ops': ops, 'mamba_ssm.ops.selective_scan_interface': ssi})
    from burstISP.utils.registry import ARCH_REGISTRY
    pkg = types.ModuleType('burstISP.archs'); pkg.__path__ = [os.path.join(REPO, 'burstISP/archs')]

    def build_network(opt):                     # what burstISP.models imports from the package
        opt = dict(opt)
        return ARCH_REGISTRY.get(opt.pop('type'))(**opt)
    pkg.build_network = build_network
    sys.modules['burstISP.archs'] = pkg


install_stubs()
from burstISP.archs.KGTSMamba import flow_align_arch as fa  # noqa: E402
from burstISP.archs.KGTSMamba import kgts_fused  # noqa: E402
from burstISP.archs.KGTSMamba.kgts_arch import KGTS, TokenBank, TokenRefine  # noqa: E402
from burstISP.archs.KGTSMamba.kgts_mamba_arch import KGTSMamba  # noqa: E402

BODY = dict(upscale=8, img_size=16, embed_dim=24, d_state=4, depths=[2, 2], num_heads=[2, 2], window_size=8,
            inner_rank=8, num_tokens=8, mlp_ratio=1., upsampler='pixelshuffle', upsample_feat=16, in_chans=4, out_chans=3)
M1ISH = dict(align=dict(type='packed', flow_feat=8, global_motion='affine'),
             token=dict(c=16, d=16, k=2, pos_freqs=[0.5, 1, 2, 4], norm=True, mark_ref=True, pin_ref=True,
                        tap_pos='backward'),
             kgts=dict(n=4, expand=2, heads=4, a_max=0.5, affinity=True, dt_norm=False))
OK = [True]


def check(name, cond, info=''):
    OK[0] &= bool(cond)
    print(f'  {"PASS" if cond else "FAIL"}  {name}  {info}')


def load_rev(rev):
    """KGTSMamba as it was at `rev`, as a private package (its registry decorators stripped)."""
    root = tempfile.mkdtemp(prefix='kgts_rev_')
    pkg = os.path.join(root, 'kgts_rev', 'KGTSMamba')
    os.makedirs(pkg)
    for d in (os.path.dirname(pkg), pkg):
        open(os.path.join(d, '__init__.py'), 'w').close()
    for f in ('kgts_arch', 'flow_align_arch', 'kgts_mamba_arch', 'mambairv2_arch'):
        src = subprocess.run(['git', '-C', REPO, 'show', f'{rev}:burstISP/archs/KGTSMamba/{f}.py'],
                             check=True, capture_output=True, text=True).stdout
        open(os.path.join(pkg, f + '.py'), 'w').write(src.replace('@ARCH_REGISTRY.register()', ''))
    sys.path.insert(0, root)
    from kgts_rev.KGTSMamba import kgts_mamba_arch
    return kgts_mamba_arch.KGTSMamba


def run(m, x, seed=1):
    torch.manual_seed(seed)
    out, aux = m(x, return_aux=True)
    loss = out.pow(2).mean() + sum(f.float().pow(2).mean() for f in aux['flows'].values())
    loss.backward()
    return out.detach(), aux, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def stage_bitexact(rev):
    print(f'[bitexact] every option off vs {rev}')
    Old = load_rev(rev)
    cfgs = {'packed / affine / M1 token+scan settings': M1ISH,
            'bayer / dt_norm / tap_pos target': dict(align=dict(type='bayer', flow_feat=8),
                                                     token=dict(c=16, d=16, k=2, tap_pos='target'),
                                                     kgts=dict(n=4, expand=1, heads=2, dt_norm=True))}
    for name, cfg in cfgs.items():
        for perturb in (False, True):
            torch.manual_seed(0); a = Old(**BODY, **cfg).train()
            torch.manual_seed(0); b = KGTSMamba(**BODY, **cfg).train()
            if perturb:
                for m in (a, b):
                    torch.manual_seed(5)
                    for p in (m.kgts.W_c.weight, m.kgts.gamma.weight) + ((m.kgts.W_q.weight,) if m.kgts.W_q is not None else ()):
                        torch.nn.init.normal_(p, std=0.1)
            sa, sb = a.state_dict(), b.state_dict()
            same = list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
            x = torch.rand(2, 5, 4, 12, 20)
            (oa, fa_, ga), (ob, fb, gb) = run(a, x), run(b, x)
            d = max([(oa - ob).abs().max().item()] + [(fa_['flows'][k] - fb['flows'][k]).abs().max().item() for k in fa_['flows']]
                    + [(ga[k] - gb[k]).abs().max().item() for k in ga]) if list(ga) == list(gb) else float('nan')
            check(f'{name}{", W_c/gamma/W_q perturbed" if perturb else ""}', same and d == 0,
                  f'state dict equal {same}, max |diff| over output, flows and {len(ga)} grads {d:.1e}')


def build(**over):
    cfg = {k: dict(v) for k, v in M1ISH.items()}
    for k, v in over.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    torch.manual_seed(0)
    return KGTSMamba(**{**BODY, 'depths': [2, 2, 2], 'num_heads': [2, 2, 2]}, **cfg)


def stage_options():
    print('[options] identity at init / fwd+bwd')
    x = torch.rand(2, 6, 4, 16, 16)
    m0 = build().eval(); torch.manual_seed(1); ref = m0(x)
    for name, over in (('kgts.depth_embed', dict(kgts=dict(depth_embed=True))),
                       ('kgts.untie_out', dict(kgts=dict(untie_out=True))),
                       ('inject_first', dict(inject_first=True)),
                       ('refine.at [0, 1]', dict(refine=dict(at=[0, 1]))),
                       ('aux_head', dict(aux_head=True))):
        m = build(**over).eval(); torch.manual_seed(1)
        out = m(x)
        check(f'{name} is an identity at init', torch.equal(out, ref), f'max |diff| {(out - ref).abs().max():.1e}')
    m = build(inject_first=True, refine=dict(at=[1]), aux_head=True,
              align=dict(global_motion='lk', local=dict(tau=0.25, win=5), token_blocks=2),
              kgts=dict(roles=['int', 'int', 'geo', 'con'], out_norm='group', depth_embed=True, untie_out=True)).train()
    for W in [m.kgts.W_c] + list(m.kgts.W_cs):
        torch.nn.init.normal_(W.weight, std=0.1)
    out, aux = m(x, return_aux=True)
    (out.pow(2).mean() + aux['burst'].pow(2).mean() + sum(f.pow(2).mean() for f in aux['flows'].values())).backward()
    nograd = [n for n, p in m.named_parameters() if p.grad is None]
    check('all options together: every parameter gets a finite gradient',
          not nograd and all(torch.isfinite(p.grad).all() for p in m.parameters()), f'missing: {nograd[:5]}')
    tr = TokenRefine(16, 24)
    t, v = torch.randn(10, 12, 16), torch.rand(10, 12) > 0.2
    t = t * v[..., None]
    check('TokenRefine is an identity at init', torch.equal(tr(t, v, torch.randn(10, 24)), t))


def stage_roles():
    print('[roles] exact invariance to the order of the non-key frames')
    torch.manual_seed(0)
    N, ref = 7, 3
    bank = TokenBank(16, 16, k=2)
    feats = torch.randn(1, N, 16, 8, 8)
    flow = torch.randn(1, N, 2, 8, 8) * 0.3 + torch.randn(1, N, 2, 1, 1)
    s = torch.randn(64, 24)
    perm = [5, 0, 6, ref, 1, 4, 2]
    for roles in (['geo', 'con', 'geo', 'con'], None):
        k = KGTS(24, 16, n=4, expand=2, heads=4, roles=roles)
        if k.W_q is not None:
            torch.nn.init.normal_(k.W_q.weight, std=0.3)
        ys = []
        for order in (list(range(N)), perm):
            tok = bank(feats[:, order], flow[:, order], ref, extras=roles is not None)
            ys.append(k.pooled(k.norm_s(s), k.precompute(*tok)))
        rel = ((ys[0] - ys[1]).norm() / ys[0].norm()).item()
        if roles:
            check('all-selector pooled states', rel == 0, f'relative change {rel:.1e}')
        else:
            print(f'  (frame-order heads for comparison: relative change {rel:.1%})')


def stage_fused():
    print('[fused] Triton end-state kernels (CPU interpreter) vs the scan path')
    if kgts_fused.triton is None:
        print('  SKIPPED: triton is not importable'); return
    for name, P, L, kg in (('roles + affinity, 8 heads (M1 shape)', 13, 8,
                            dict(d=16, n=16, expand=2, heads=8, out_norm='group', a_max_sel=8.0,
                                 roles=['int'] * 4 + ['geo'] * 2 + ['con'] * 2)),
                           ('no roles, no affinity, padded widths', 11, 8,
                            dict(d=12, n=6, expand=2, heads=4, affinity=False))):
        torch.manual_seed(0)
        d = kg.pop('d')
        ks = [KGTS(24, d, n_calls=2, depth_embed=True, untie_out=True, fused=f, **kg) for f in (False, True)]
        for p in [ks[0].W_c.weight, ks[0].gamma.weight, ks[0].beta.weight, ks[0].W_cs[0].weight] + \
                 ([ks[0].W_q.weight] if ks[0].W_q is not None else []):
            torch.nn.init.normal_(p, std=0.2)
        ks[1].load_state_dict(ks[0].state_dict())
        ks[1].use_fused = lambda t: True                 # the kernels, on CPU tensors
        x, s = torch.randn(P, L, d), torch.randn(P, 24) * 3
        valid = torch.rand(P, L) > 0.2
        ex = {'pos': torch.rand(P, L, 2) * 2 - 0.5, 'cons': -torch.rand(P, L), 'is_ref': torch.arange(L) // 4 == 1}
        res = []
        for k in ks:
            xs, ss = x.clone().requires_grad_(), s.clone().requires_grad_()
            cache = k.precompute(xs, valid, ex)
            y, out = k.pooled(k.norm_s(ss), cache), k(ss, cache, 1)
            (out.pow(2).mean() + y.pow(2).mean()).backward()
            res.append((y.detach(), out.detach(), {'x': xs.grad, 's': ss.grad,
                        **{n: p.grad for n, p in k.named_parameters() if p.grad is not None}}))
        rel = lambda a, b: ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
        (y0, o0, g0), (y1, o1, g1) = res
        worst = max((rel(g1[n], g0[n]), n) for n in g0)
        check(name, rel(y1, y0) < 1e-5 and rel(o1, o0) < 1e-5 and worst[0] < 1e-4 and set(g0) == set(g1),
              f'pooled {rel(y1, y0):.1e}  out {rel(o1, o0):.1e}  worst grad {worst[0]:.1e} ({worst[1]})')


def stage_geometry(images, n=30):
    print(f'[geometry] affine_lk vs the DBSR generator ({n} bursts from {images})')
    import random
    import cv2
    from burstISP.data.dbsr import processing_utils as prutils
    from burstISP.data.dbsr import synthetic_burst_generation as syn
    from burstISP.data.dbsr.data_format_utils import npimage_to_torch
    from burstISP.data.synthetic_burst_dataset import BURST_TRANSFORMATION_PARAMS, IMAGE_PROCESSING_PARAMS, \
        SyntheticBurstDataset
    from burstISP.models.mambafusion_model import backward_flow
    files = sorted(sum((glob.glob(os.path.join(images, e)) for e in ('*.jpg', '*.png', '*.jpeg')), []))
    if not files:
        print(f'  SKIPPED: no images in {images}'); return
    random.seed(0); torch.manual_seed(0)
    bc, crop = BURST_TRANSFORMATION_PARAMS['border_crop'], 384
    lqs, trus = [], []
    for i in range(n):
        img = npimage_to_torch(cv2.imread(files[i % len(files)], cv2.IMREAD_COLOR), normalize=True, input_bgr=True)
        fr = prutils.random_resized_crop(img, [crop + 2 * bc] * 2)
        burst, _, _, fv, _ = syn.rgb2rawburst(fr, 14, 4, dict(BURST_TRANSFORMATION_PARAMS), dict(IMAGE_PROCESSING_PARAMS))
        order = SyntheticBurstDataset._center_ref_order(14)
        lqs.append(burst[order]); trus.append(fv[order])
    lq, fv = torch.stack(lqs).float(), torch.stack(trus).float()
    h = lq.shape[-1]
    tru = F.avg_pool2d(backward_flow(fv.flatten(0, 1)), 2).view(n, 14, 2, h, h) * 0.5
    est = fa.affine_lk(lq, 7)
    keep = torch.arange(14) != 7
    e = (est - tru)[:, keep][..., 3:-3, 3:-3].norm(dim=2)
    check('affine_lk from zero motion, mean EPE < 0.12 packed px', e.mean() < 0.12,
          f'mean {e.mean():.3f}  median {e.flatten(2).mean(-1).median():.3f} packed px')
    glob_ = tru.clone()
    dense = tru + 0.3 * torch.randn_like(tru)
    dense[..., 0, 10:20, 10:20] += 2.0; dense[..., 1, 10:20, 10:20] -= 1.0
    hyb = fa.local_residual(dense, glob_)
    inn = (hyb - tru - torch.tensor([2., -1.]).view(1, 1, 2, 1, 1))[..., 12:18, 12:18].norm(dim=2).mean()
    out = (hyb - tru)[..., 30:40, 30:40].norm(dim=2).mean()
    check('local_residual follows a moving patch and keeps the global model elsewhere', inn < 0.2 and out < 0.1,
          f'EPE inside {inn:.3f}, outside {out:.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rev', default='236dbba', help='git revision the options-off model must reproduce')
    ap.add_argument('--images', default=None, help='folder of natural images for the geometry stage')
    ap.add_argument('--skip', default='')
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(',')))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if 'bitexact' not in skip:
        stage_bitexact(args.rev)
    if 'options' not in skip:
        stage_options()
    if 'roles' not in skip:
        stage_roles()
    if 'fused' not in skip:
        stage_fused()
    if 'geometry' not in skip and args.images:
        stage_geometry(args.images)
    print('ALL PASS' if OK[0] else 'SOME CHECKS FAILED')
    sys.exit(0 if OK[0] else 1)


if __name__ == '__main__':
    main()
