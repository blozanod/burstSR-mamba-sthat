#!/usr/bin/env python3
"""How the model learns: the same bursts through every saved checkpoint.

    python analysis/visualize_progress.py --config main/configs/M1_KGTSMamba.yml
        [--source auto|synburst|zurich|realbsr] [--checkpoints_dir DIR] [--output_path DIR]

Runs at the end of every training job (analysis/run_analysis.py, stage 2) and on any GPU node
mid-run (main/jobs/analysis_job.sh): it only reads the checkpoints already saved.

--source (auto: synburst for configs whose val set is SyntheticBurstDataset, else realbsr)
  synburst  official SyntheticBurstVal bursts -- Zurich RAW-to-RGB test images through the DBSR
            generator: the domain SyntheticBurst models train and are scored in.
  zurich    bursts generated from a Zurich RAW-to-RGB split with the same protocol, deterministic
            per index (--split test, or train to watch the training distribution itself).
  realbsr   the original behaviour: ten RealBSR-RAW test bursts, rendered with their camera pkl.

synburst / zurich write under --output_path:
  <burst>/<step>.png, gt.png, keyframe.png  full 384x384 outputs per checkpoint, rendered to sRGB
                           with the burst's own camera parameters (DBSR's process_linear_image_rgb:
                           gains, CCM, gamma, tone curve), so they look like the Zurich photo
  <burst>_strip.png        the GT's most detailed 96x96 HR window (zoom x3) across checkpoints:
                           keyframe (bilinear) | checkpoints ... | GT, PSNR of the full image on each;
                           second row: |error|, one scale per burst (3x the last checkpoint's RMS)
  curves.csv, curves.png   per checkpoint, over --metric_bursts bursts: PSNR (official metric), PSNR
                           with every frame replaced by the keyframe (all-ref), the burst gain
                           between them, the aux head's burst-only PSNR, and the error energy per
                           frequency band (analysis/diagnostics/common.py) -- when the burst branch
                           starts to matter, and which frequencies each phase of training fixes
Bursts shown: --bursts, else the --n_vis most detailed GTs of the first --pool bursts. Every
checkpoint sees the same bursts and the same MambaIRv2 routing noise (paired routing), so changes
between columns are the weights, not the noise. EMA weights unless --weights raw.
"""
import argparse
import copy
import csv
import glob
import os
import pickle as pkl
import random
import re
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'analysis', 'diagnostics'))


def get_step_number(filename):
    """Extracts the numerical step from the checkpoint filename."""
    match = re.search(r'\d+', os.path.basename(filename))
    return match.group() if match else "unknown_step"


def list_checkpoints(d):
    """[(step, path)] sorted by step; net_g_latest.pth only if there is no numbered checkpoint."""
    paths = glob.glob(os.path.join(d, '*.pth'))
    num = sorted((int(get_step_number(p)), p) for p in paths if get_step_number(p).isdigit())
    if num:
        return num
    return [(-1, p) for p in sorted(paths)]


def load_weights(model, path, weights='ema'):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    keys = (['params_ema', 'params'] if weights == 'ema' else ['params']) + ['state_dict']
    for k in keys:
        if k in sd:
            model.load_state_dict(sd[k], strict=True)
            return k
    model.load_state_dict(sd, strict=True)
    return '<root>'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoints_dir', type=str, default=None,
                        help='Directory containing the model .pth files (default experiments/<name>/models)')
    parser.add_argument('--source', choices=['auto', 'synburst', 'zurich', 'realbsr'], default='auto')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='dataset root (default: config val dataroot for synburst, train dataroot for zurich, '
                             'dataset/RealBSR_RAW_testpatch for realbsr)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='default analysis/outputs/<name>/progress')
    parser.add_argument('--split', default='test', help='zurich: Zurich RAW-to-RGB split to generate from')
    parser.add_argument('--bursts', default='', help='comma list of burst indices to show (default: most detailed)')
    parser.add_argument('--n_vis', type=int, default=8, help='bursts shown')
    parser.add_argument('--pool', type=int, default=None,
                        help='first N bursts to pick from (default: all 300 synburst / 100 zurich)')
    parser.add_argument('--metric_bursts', type=int, default=32, help='bursts the curves average over')
    parser.add_argument('--max_cols', type=int, default=8, help='checkpoints per strip (evenly spaced, last kept)')
    parser.add_argument('--weights', choices=['ema', 'raw'], default='ema')
    parser.add_argument('--amp', action='store_true', help='bf16 autocast (validation runs fp32)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', help='smoke-test mode (stubbed mamba_ssm; tiny configs only)')
    args = parser.parse_args()

    import common as C  # analysis/diagnostics/common.py
    device, build_network = C.init_backend(args.cpu)
    opt = C.load_opt(args.config)
    source = args.source
    if source == 'auto':
        source = 'synburst' if opt['datasets']['val'].get('type') == 'SyntheticBurstDataset' else 'realbsr'
    ckpt_dir = args.checkpoints_dir or os.path.join(REPO, 'experiments', opt['name'], 'models')
    out = args.output_path or os.path.join(REPO, 'analysis', 'outputs', opt['name'], 'progress')
    ckpts = list_checkpoints(ckpt_dir)
    if not ckpts:
        print(f"Error: No .pth files found in {ckpt_dir}")
        return
    os.makedirs(out, exist_ok=True)
    print(f'{len(ckpts)} checkpoints in {ckpt_dir}; source {source}; output {out}')
    model = build_network(dict(opt['network_g'])).to(device).eval()
    if source == 'realbsr':
        run_realbsr(args, opt, model, ckpts, device, out)
    else:
        run_synthetic(args, opt, model, ckpts, device, out, source, C)


# ----------------------------------------------------------------------------- SyntheticBurst domain

def run_synthetic(args, opt, model, ckpts, device, out, source, C):
    import cv2
    single = opt.get('model_type') == 'MambaIRv2Model'          # keyframe-only models (M0) take [B, C, h, w]
    if source == 'synburst':
        from burstISP.data.dbsr.synthetic_burst_val_set import SyntheticBurstVal
        root = args.input_dir or opt['datasets']['val']['dataroot']
        vs = SyntheticBurstVal(root=root)
        pool = min(args.pool or len(vs), len(vs))

        def get(i):
            burst, gt, meta = vs[i]
            return {'burst': burst, 'gt': gt, 'name': meta['burst_name'], 'isp': C.isp_params(meta)}
    else:
        root = args.input_dir or opt['datasets']['train']['dataroot']
        gen = C.GeneratedBursts(root, args.split, n=args.pool or 100, seed=args.seed)
        pool = len(gen)

        def get(i):
            s = gen[i]
            return {'burst': s['burst'], 'gt': s['gt'], 'name': s['name'], 'isp': s['isp']}

    cache = {}

    def burst(i):
        if i not in cache:
            cache[i] = get(i)
        return cache[i]

    def detail(gt):
        g = C.crop(gt).mean(0, keepdim=True)[None]
        return (g - C.gblur(g, 3.0)).pow(2).mean().item()

    if args.bursts:
        vis = [int(b) for b in args.bursts.split(',') if b]
    else:
        print(f'ranking {pool} bursts by GT detail ...')
        scores = [(detail(get(i)['gt']), i) for i in range(pool)]
        vis = [i for _, i in sorted(scores, reverse=True)[:args.n_vis]]
    metric = sorted(set(np.linspace(0, pool - 1, min(args.metric_bursts, pool)).round().astype(int).tolist()))
    todo = sorted(set(vis) | set(metric))
    print(f'showing bursts {vis}; curves over {len(metric)} bursts')

    run = C.Runner(model, device, 'paired', args.seed, args.amp)
    has_aux = getattr(model, 'aux', None) is not None
    S = None
    rows, tiles = [], {i: {} for i in vis}
    windows = {}
    for step, path in ckpts:
        key = load_weights(model, path, args.weights)
        model.eval()
        print(f'checkpoint {os.path.basename(path)} [{key}]', flush=True)
        acc = []
        for i in todo:
            b = burst(i)
            fr, gt = b['burst'][None], b['gt'].to(device)
            N = fr.shape[1]
            lq, _ = C.arrange(fr, range(1, N), C.ref_slot(model, N))
            if single:
                o = run(lq[:, C.ref_slot(model, N)])
                outs, aux = [o[0]], None
            else:
                allref = fr[:, :1].expand_as(fr).contiguous()
                o = run(torch.cat([lq, allref]), return_aux=has_aux)
                o, aux = (o[0], o[1].get('burst')) if has_aux else (o, None)
                outs = [o[0], o[1]]
            e = C.error(outs[0], gt)
            if S is None:
                S = C.Spectrum(*e.shape[-2:], device)
            r = {'psnr': C.psnr_from_mse(e.pow(2).mean().item()), 'bands': S.bands(S.fft(e)).cpu().numpy()}
            if len(outs) > 1:
                r['psnr_allref'] = C.psnr_from_mse(C.error(outs[1], gt).pow(2).mean().item())
            if aux is not None:
                r['psnr_aux'] = C.psnr_from_mse(C.error(aux[0], gt).pow(2).mean().item())
            if i in metric:
                acc.append(r)
            if i in tiles:
                d = os.path.join(out, b['name'])
                os.makedirs(d, exist_ok=True)
                ex = b['gt'].mean().item()
                cv2.imwrite(os.path.join(d, f'{step}.png'), C.render_srgb(outs[0], b['isp'], ex)[..., ::-1])
                if i not in windows:
                    windows[i] = best_window(b['gt'], C)
                    cv2.imwrite(os.path.join(d, 'gt.png'), C.render_srgb(b['gt'], b['isp'], ex)[..., ::-1])
                    key_rgb = keyframe_rgb(b['burst'][0], b['gt'].shape[-2:])
                    cv2.imwrite(os.path.join(d, 'keyframe.png'), C.render_srgb(key_rgb, b['isp'], ex)[..., ::-1])
                    tiles[i]['key'] = (key_rgb, C.psnr_from_mse(C.error(key_rgb.to(device), gt).pow(2).mean().item()))
                tiles[i][step] = (outs[0].cpu(), r['psnr'])
        row = {'step': step, 'ckpt': os.path.basename(path), 'n': len(acc),
               'psnr': float(np.mean([a['psnr'] for a in acc]))}
        if acc and 'psnr_allref' in acc[0]:
            row['psnr_allref'] = float(np.mean([a['psnr_allref'] for a in acc]))
            row['burst_gain'] = row['psnr'] - row['psnr_allref']
        if acc and 'psnr_aux' in acc[0]:
            row['psnr_aux'] = float(np.mean([a['psnr_aux'] for a in acc]))
        bands = np.mean([a['bands'] for a in acc], 0)
        for k, nm in enumerate(C.BAND_NAMES):
            row[f'band_{nm}_db'] = float(10 * np.log10(max(bands[k], 1e-20)))
        rows.append(row)
        print('  ' + '  '.join(f'{k} {v:.3f}' for k, v in row.items() if isinstance(v, float)), flush=True)
    run.routing.close()

    for i in vis:
        b = burst(i)
        strip = make_strip(tiles[i], [s for s, _ in ckpts], b, windows[i], args.max_cols, C)
        cv2.imwrite(os.path.join(out, f'{b["name"]}_strip.png'), strip[..., ::-1])
    write_curves(rows, out, opt, source, C)
    print(f'\nDone: {len(vis)} strips, curves.csv / curves.png in {os.path.abspath(out)}')


def keyframe_rgb(packed, hw):
    """Packed RGGB keyframe (4, h, w) -> naive RGB (R, mean G, B) bilinearly upsampled to (3, *hw)."""
    rgb = torch.stack([packed[0], 0.5 * (packed[1] + packed[2]), packed[3]])[None]
    return torch.nn.functional.interpolate(rgb, size=tuple(hw), mode='bilinear', align_corners=False)[0]


def best_window(gt, C, size=96, stride=8):
    """Top-left (y, x) of the size x size HR window with the most high-pass GT energy, inside the
    official metric's crop."""
    g = gt.mean(0, keepdim=True)[None]
    e = (g - C.gblur(g, 2.0)).pow(2)[0, 0]
    b = C.BOUNDARY
    e = e[b:-b, b:-b]
    pooled = torch.nn.functional.avg_pool2d(e[None, None], size, stride)[0, 0]
    k = int(pooled.argmax())
    return b + (k // pooled.shape[1]) * stride, b + (k % pooled.shape[1]) * stride


def make_strip(tiles, steps, b, win, max_cols, C, size=96, zoom=3):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import cm
    have = [s for s in steps if s in tiles]
    if len(have) > max_cols:
        pick = np.linspace(0, len(have) - 1, max_cols).round().astype(int)
        have = [have[k] for k in sorted(set(pick.tolist()))]
    y, x = win
    gt = b['gt']
    ex = gt.mean().item()
    sl = (slice(None), slice(y, y + size), slice(x, x + size))
    last = tiles[have[-1]][0][sl]
    scale = 3 * float((last - gt[sl]).pow(2).mean().sqrt()) + 1e-6

    def up(img):
        return cv2.resize(img, (size * zoom, size * zoom), interpolation=cv2.INTER_NEAREST)

    def label(img, text):
        img = np.ascontiguousarray(img)
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            for j, t in enumerate(text.split('\n')):
                cv2.putText(img, t, (6, 22 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th, cv2.LINE_AA)
        return img

    def err(img):
        e = (img[sl] - gt[sl]).abs().mean(0).clamp(0, scale) / scale
        return up((cm.magma(e.numpy())[..., :3] * 255).astype(np.uint8))

    cols = [('keyframe', *tiles['key'])] + [(f'{s // 1000}k' if s >= 1000 else str(s), *tiles[s]) for s in have]
    top, bot = [], []
    for name, img, p in cols:
        top.append(label(up(C.render_srgb(img[sl], b['isp'], ex)), f'{name}\n{p:.2f} dB'))
        bot.append(err(img))
    top.append(label(up(C.render_srgb(gt[sl], b['isp'], ex)), 'GT'))
    bot.append(label(np.zeros_like(bot[0]), f'|err| 0..{scale:.3f}'))
    sep = np.full((size * zoom, 4, 3), 255, np.uint8)
    row = lambda ts: np.concatenate(sum([[t, sep] for t in ts], [])[:-1], 1)
    return np.concatenate([row(top), np.full((4, row(top).shape[1], 3), 255, np.uint8), row(bot)], 0)


def write_curves(rows, out, opt, source, C):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, 'curves.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps = [r['step'] / 1000 for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for k, lab, sty in (('psnr', 'burst', 'C0o-'), ('psnr_allref', 'all-ref (no new information)', 'C1o--'),
                        ('psnr_aux', 'aux head (burst only)', 'C2s:')):
        if k in rows[0]:
            ax[0].plot(steps, [r[k] for r in rows], sty, ms=4, label=lab)
    ax[0].set_xlabel('iteration (k)')
    ax[0].set_ylabel('PSNR (dB, official metric)')
    ax[0].legend(fontsize=8)
    ax[0].set_title(f'{rows[0]["n"]} {source} bursts')
    if 'burst_gain' in rows[0]:
        ax[1].plot(steps, [r['burst_gain'] for r in rows], 'C3o-', ms=4)
    ax[1].axhline(0, color='0.7', lw=0.8)
    ax[1].set_xlabel('iteration (k)')
    ax[1].set_ylabel('burst gain (dB)')
    ax[1].set_title('burst vs all-ref')
    for nm in C.BAND_NAMES:
        ax[2].plot(steps, [r[f'band_{nm}_db'] for r in rows], 'o-', ms=3, label=nm)
    ax[2].set_xlabel('iteration (k)')
    ax[2].set_ylabel('error energy in band (dB, MSE units)')
    ax[2].set_title('which frequencies training fixes')
    ax[2].legend(fontsize=8, title='band (packed Nyquist)')
    fig.suptitle(f'{opt["name"]}: progress over checkpoints')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'curves.png'), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- RealBSR (original behaviour)

def run_realbsr(args, opt, model, ckpts, device, out):
    import cv2
    from burstISP.utils.img_util import imwrite, generate_processed_image_channel3
    from burstISP.data.burst_image_dataset import BurstImageDataset

    input_dir = args.input_dir or os.path.join(REPO, 'dataset', 'RealBSR_RAW_testpatch')
    num_frames = opt['datasets']['val'].get('num_frames', opt['network_g'].get('num_frames', 5))
    input_dirs = ["010_0023", "010_0104", "013_0265", "010_0292", "020_0543",
                  "014_0674", "006_0291", "007_0065", "020_0047", "027_0388"]

    dataset_opt = {'dataroot': input_dir, 'num_frames': num_frames, 'phase': 'test'}
    dataset = BurstImageDataset(dataset_opt)
    random.seed(42)

    # Pre-find the dataset indices for the target images so we don't iterate the
    # entire dataset for every checkpoint.
    print("Scanning dataset for target images to optimize I/O...")
    target_indices = []
    for i in range(len(dataset)):
        data = dataset[i]
        directory = os.path.basename(data['lq_path'])
        if directory in input_dirs:
            target_indices.append((i, directory))
            if len(target_indices) == len(input_dirs):
                break

    for ckpt_path in [p for _, p in ckpts]:
        step = get_step_number(ckpt_path)
        print(f"\n{'=' * 50}")
        print(f"Loading checkpoint: {os.path.basename(ckpt_path)} (Step: {step})")
        load_weights(model, ckpt_path, args.weights)
        model.eval()

        for idx, directory in target_indices:
            print(f"  -> Processing burst: {directory}")
            img_out_folder = os.path.join(out, directory)
            os.makedirs(img_out_folder, exist_ok=True)

            data = dataset[idx]
            with open(data['meta'], "rb") as f:
                meta_data = pkl.load(f)

            input_tensor = data['lq'].unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    output_tensor = model(input_tensor)
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            output_tensor = output_tensor.squeeze(0).float()

            try:
                vis_img = generate_processed_image_channel3(
                    output_tensor, copy.deepcopy(meta_data), return_np=True, black_level_substracted=True)
            except Exception as e:
                print(f"  -> Error during ISP processing for {directory}: {e}")
                continue

            # Title the image as the checkpoint step
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            imwrite(vis_img_bgr, os.path.join(img_out_folder, f"{step}.png"))

    print(f"\nAll inferences complete. Results organized in: {os.path.abspath(out)}")


if __name__ == '__main__':
    main()
