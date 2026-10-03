#!/usr/bin/env python3
"""CPU smoke test for analysis/diagnostics: fake data, a tiny KGTSMamba, every script end to end.

    python analysis/diagnostics/smoke_test.py [--dir /tmp/kgts_diag_smoke] [--keep]

No GPU, no datasets, no checkpoint needed (mamba_ssm stubbed as in analysis/kgts_cpu_checks.py).
It writes procedural "Zurich" images, a SyntheticBurstVal-format set generated from them, a tiny
config derived from M1 (every M1 flag on, widths cut) and a random checkpoint whose burst branch
is switched on (W_c perturbed), then:
  1. unit checks of the measurement code: the bands partition the MSE exactly; a blur shows up as
     attenuation and white noise as additive; the colour fit recovers a known affine map; paired
     routing makes a burst's output independent of its batch; oracle_flow is in the convention the
     model's LK flow is in (and kgts_sanity's geometry stage passes on these bursts);
  2. run_all.py --cpu on a few bursts, and checks every script wrote its outputs.
Numbers from the random model mean nothing; this only proves the code paths run.
"""
import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

OK = [True]


def check(name, cond, info=''):
    OK[0] &= bool(cond)
    print(f'  {"PASS" if cond else "FAIL"}  {name}  {info}', flush=True)


def fake_image(rng, size=448):
    import cv2
    img = np.zeros((size, size, 3), np.float32)
    for s in (2, 4, 8, 16, 32, 64):
        n = rng.standard_normal((size // s + 2, size // s + 2, 3)).astype(np.float32)
        img += cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC) * (s / 64) ** 0.5
    img = (img - img.min()) / (img.max() - img.min())
    for _ in range(25):
        c = tuple(float(v) for v in rng.uniform(0, 1, 3))
        x, y = (int(v) for v in rng.integers(0, size, 2))
        r = int(rng.integers(8, 60))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, c, -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + int(r * rng.uniform(0.3, 2))), c, -1)
    return (img * 255).clip(0, 255).astype(np.uint8)


def make_data(root, n_img=10, n_val=6):
    import cv2
    rng = np.random.default_rng(0)
    for split in ('train', 'test'):
        d = os.path.join(root, 'zurich', split, 'canon')
        os.makedirs(d, exist_ok=True)
        for i in range(n_img):
            cv2.imwrite(os.path.join(d, f'{i}.jpg'), fake_image(rng))
    gen = C.GeneratedBursts(os.path.join(root, 'zurich'), 'test', n=n_val, seed=123)
    for i in range(n_val):
        s = gen[i]
        bd, gd = os.path.join(root, 'val', 'bursts', f'{i:04d}'), os.path.join(root, 'val', 'gt', f'{i:04d}')
        os.makedirs(bd, exist_ok=True)
        os.makedirs(gd, exist_ok=True)
        q = lambda t: (t.permute(1, 2, 0).numpy() * 2 ** 14).round().clip(0, 2 ** 16 - 1).astype(np.uint16)
        for k in range(s['burst'].shape[0]):
            cv2.imwrite(os.path.join(bd, f'im_raw_{k:02d}.png'), q(s['burst'][k]))
        cv2.imwrite(os.path.join(gd, 'im_rgb.png'), q(s['gt']))
        with open(os.path.join(gd, 'meta_info.pkl'), 'wb') as f:
            pickle.dump({'shot_noise_level': s['shot'], 'read_noise_level': s['read']}, f)


def make_config(root):
    opt = C.load_opt(os.path.join(C.REPO, 'main', 'configs', 'M1_KGTSMamba.yml'))
    opt = yaml.safe_load(yaml.safe_dump(json_safe(opt)))
    opt['name'] = 'SMOKE_KGTS'
    opt['datasets']['train']['dataroot'] = os.path.join(root, 'zurich')
    opt['datasets']['val']['dataroot'] = os.path.join(root, 'val')
    n = opt['network_g']
    n.update(embed_dim=24, d_state=4, depths=[2, 2], num_heads=[2, 2], window_size=8, inner_rank=8, num_tokens=8,
             mlp_ratio=1.0, upsample_feat=16)
    n['align'].update(flow_feat=8)
    n['token'].update(c=16, d=16)
    n['kgts'].update(n=4, expand=2, heads=4, roles=['int', 'int', 'geo', 'con'])
    n['refine'] = {'at': [1], 'hidden': 16}
    path = os.path.join(root, 'smoke.yml')
    with open(path, 'w') as f:
        yaml.safe_dump(opt, f, sort_keys=False)
    return path, opt


def json_safe(o):
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o


def make_ckpt(root, opt, build_network):
    torch.manual_seed(0)
    net = build_network(dict(opt['network_g']))
    with torch.no_grad():                         # switch the burst branch on (W_c etc. are zero-init)
        for W in [net.kgts.W_c] + list(getattr(net.kgts, 'W_cs', [])):
            torch.nn.init.normal_(W.weight, std=0.2)
        for p in (net.kgts.gamma.weight, net.kgts.W_q.weight):
            torch.nn.init.normal_(p, std=0.05)
    d = os.path.join(root, 'experiments', opt['name'], 'models')
    os.makedirs(d, exist_ok=True)
    sd = net.state_dict()
    torch.save({'params': sd, 'params_ema': sd}, os.path.join(d, 'net_g_1000.pth'))
    return net.eval()


def unit_checks(root, opt, net):
    print('[unit] measurement code')
    dev = torch.device('cpu')
    g = C.GeneratedBursts(os.path.join(root, 'zurich'), 'test', n=3, seed=5)
    s = g[0]
    gt = C.crop(s['gt'])
    S = C.Spectrum(*gt.shape[-2:], dev)
    e = torch.randn_like(gt) * 0.01 + 0.02 * C.gblur(torch.randn_like(gt)[None], 4.0)[0]
    X = S.fft(e)
    mse = e.pow(2).mean().item()
    check('bands partition the MSE', abs(S.bands(X).sum().item() / mse - 1) < 1e-4 and abs(S.split(X).sum().item() / mse - 1) < 1e-4,
          f'{S.bands(X).sum().item():.6e} / {S.split(X).sum().item():.6e} vs {mse:.6e}')
    check('opponent basis is orthonormal', torch.allclose(C.OPPONENT @ C.OPPONENT.T, torch.eye(3), atol=1e-6))
    Xg = S.fft(gt - gt.mean((-2, -1), keepdim=True))
    blur = C.gblur(gt[None], 2.0)[0] - gt
    Xb = S.fft(blur)
    att = (S.bands(Xb, Xg) ** 2 / S.bands(Xg)) / S.bands(Xb)
    mtf = 1 + S.bands(Xb, Xg) / S.bands(Xg)
    check('a blur reads as attenuation, MTF falling with frequency', att[2:].min() > 0.9 and bool((mtf[1:] < mtf[:-1]).all()),
          f'atten {att.numpy().round(3)} MTF {mtf.numpy().round(3)}')
    Xn = S.fft(torch.randn_like(gt) * 0.01)
    attn = (S.bands(Xn, Xg) ** 2 / S.bands(Xg)) / S.bands(Xn)
    check('white noise reads as additive', attn.max() < 0.05, f'atten {attn.numpy().round(4)}')

    import colour_fit as CF
    P = gt.reshape(3, -1).T.double()
    A = torch.tensor([[0.97, 0.02, 0.0], [0.01, 1.03, -0.02], [0.0, 0.03, 0.95]], dtype=torch.float64)
    pred = P @ A.T + torch.tensor([0.004, -0.002, 0.003], dtype=torch.float64)
    fit, _ = CF.fit_apply(pred, P, 'affine', torch.linspace(0, 1, 16, dtype=torch.float64) ** 2,
                          torch.ones(P.shape[0], dtype=torch.bool))
    check('affine fit recovers a known colour map', (fit - P).abs().max() < 1e-6, f'max err {(fit - P).abs().max():.1e}')
    fit_t, _ = CF.fit_apply(pred, P, 'tone', torch.linspace(0, 1, 16, dtype=torch.float64) ** 2,
                            torch.ones(P.shape[0], dtype=torch.bool))
    check('tone fit is at least as good as affine on it', (fit_t - P).pow(2).mean() <= (pred - P).pow(2).mean() * 1e-2)

    run = C.Runner(net, dev, 'paired', 0)
    b = torch.stack([g[i]['burst'] for i in range(2)])
    lq, _ = C.arrange(b, range(1, 14), C.ref_slot(net, 14))
    o1, o2, o12 = run(lq[:1]), run(lq[:1]), run(lq)
    check('paired routing: same burst, same output, whatever the batch', torch.equal(o1, o2) and
          torch.allclose(o1[0], o12[0], atol=1e-5), f'max |diff| batch vs alone {(o1[0] - o12[0]).abs().max():.1e}')
    run.routing.close()
    o3, o4 = net(lq[:1]), net(lq[:1])
    print(f'  (for scale: unpaired stochastic routing, two forwards differ by {(o3 - o4).abs().max():.1e})')

    from burstISP.archs.KGTSMamba.flow_align_arch import affine_lk
    fo = C.oracle_flow(torch.stack([g[i]['flow'] for i in range(3)]))
    bb = torch.stack([g[i]['burst'] for i in range(3)])
    lk = affine_lk(bb, 0)                                            # generator order: keyframe at 0
    m = 4
    epe = (lk - fo)[:, 1:, :, m:-m, m:-m].norm(dim=2).mean().item()
    epe_neg = (lk + fo)[:, 1:, :, m:-m, m:-m].norm(dim=2).mean().item()
    check('oracle_flow is in LK\'s convention (backward field, packed px)', epe < 0.25 and epe_neg > 4 * epe,
          f'EPE {epe:.3f} px (negated oracle: {epe_neg:.3f})')
    import kgts_sanity
    ok = kgts_sanity.stage_geometry(opt, {'type': 'KGTSMamba', **opt['network_g']}, n=2)
    check('kgts_sanity geometry stage on these bursts (taps land on the generator\'s samples)', ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=None)
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--limit', type=int, default=3)
    args = ap.parse_args()
    root = args.dir or tempfile.mkdtemp(prefix='kgts_diag_smoke_')
    os.makedirs(root, exist_ok=True)
    print(f'smoke test in {root}')
    _, build_network = C.init_backend(cpu=True)
    sys.path.insert(0, os.path.join(C.REPO, 'analysis'))
    make_data(root)
    cfg, opt = make_config(root)
    net = make_ckpt(root, opt, build_network)
    unit_checks(root, opt, net)

    print('[run_all] every script, CPU, tiny model')
    out = os.path.join(root, 'out')
    cmd = [sys.executable, os.path.join(HERE, 'run_all.py'), '--config', cfg, '--exp_root', os.path.join(root, 'experiments'),
           '--out', out, '--cpu', '--limit', str(args.limit), '--n', str(args.limit), '--batch', '2', '--workers', '0',
           '--ns', '1,2,8,14']
    rc = subprocess.run(cmd).returncode
    check('run_all exits 0', rc == 0, f'exit {rc}')
    for s in ('error_bands', 'colour_fit', 'oracle_geometry', 'burst_length'):
        have = [os.path.isfile(os.path.join(out, f'{s}{x}')) for x in ('.json', '.txt', '.png', '_per_image.csv')]
        check(f'{s} wrote json / txt / png / csv', all(have), str(have))
    check('SUMMARY.md written', os.path.isfile(os.path.join(out, 'SUMMARY.md')))
    print('ALL PASS' if OK[0] else 'SOME CHECKS FAILED')
    if not args.keep and not args.dir:
        shutil.rmtree(root, ignore_errors=True)
    sys.exit(0 if OK[0] else 1)


if __name__ == '__main__':
    main()
