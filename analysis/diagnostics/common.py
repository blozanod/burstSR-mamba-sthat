"""Shared plumbing for the KGTSMamba diagnostics in analysis/diagnostics/.

Every script here evaluates one trained checkpoint (EMA weights by default: the ones validation
and release use) and writes to analysis/outputs/diagnostics/<run name>/<checkpoint>/:
<script>.txt (the report, also printed), <script>.json (headline numbers + readout lines, which
run_all.py collects into SUMMARY.md), <script>_per_image.csv and <script>.png.

Conventions shared by every script
----------------------------------
* Metric. The official SyntheticBurst PSNR (burstISP/metrics/synburst_psnr.py): output clamped
  to [0, 1], truncated to 14 bits, 40 px border ignored. Every error image is that quantised
  output minus the GT on that crop, so the band energies of an image add up to exactly the MSE
  its headline PSNR comes from.
* Precision. fp32, as MambaFusionModel.test() runs validation (--amp for bf16 autocast).
* Routing. MambaIRv2's ASSM draws a hard Gumbel-softmax route in eval as well
  (docs/KGTS_ADVERSARIAL_REVIEW.md §7), so two forwards of one burst differ. 'paired' (default)
  draws that noise from a generator seeded per ASSM call, shared across the batch: every arm of a
  comparison (burst vs all-ref, LK vs oracle flow, N = 1..14) sees the same routing noise and the
  deltas carry none of it, while each forward is still a regular draw of the model's own eval
  noise. 'stochastic' leaves training validation's behaviour; 'argmax' removes the noise (a mode
  the model never trained in).
* Burst layout. Sources return frames in generator order (keyframe at 0, as SyntheticBurstVal
  stores them). arrange() puts the keyframe in the slot the network reads for an n-frame burst
  (align.ref_idx, else n // 2), the layout SyntheticBurstDataset feeds in training.
* Frequencies. In units of the packed-LR Nyquist F_N = 1/16 cycles per HR px: one packed pixel is
  8 HR px, and a single frame samples each of R and B once per packed pixel. Bands:
      <0.5 F_N   large-scale tone / colour / exposure
      0.5-1      up to the packed Nyquist: one frame resolves this for every channel
      1-2        packed -> Bayer (LR-RGB) Nyquist: the demosaicking range, partly in one frame (G)
      2-4, >4    beyond what any single frame samples: only sub-pixel burst information (or the
                 body's priors) puts anything here
  Spectra are of the mirror-extended crop (no wrap-around edge), so Parseval holds exactly and
  the bands partition the MSE. The 2-band split weights each frequency's energy by the amplitude
  response G(f) of a Gaussian of sigma 3.0 HR px (G = 1/2 at F_N, i.e. "a Gaussian low-pass at
  about one LR pixel") and the rest by 1 - G: an exact partition with a soft edge.

On a machine without CUDA (or with --cpu) the scripts run in smoke-test mode: mamba_ssm is
replaced by its pure-torch reference scan through analysis/kgts_cpu_checks.py's stubs. That is
only practical with a tiny network (see analysis/diagnostics/README.md, "Smoke test").
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

F_N = 1.0 / 16                     # packed-LR Nyquist, cycles per HR px
BAND_EDGES = (0.5, 1.0, 2.0, 4.0)  # in F_N units; 5 bands
BAND_NAMES = ('<0.5', '0.5-1', '1-2', '2-4', '>4')
BAND_NOTES = ('tone / colour / exposure', 'up to packed Nyquist', 'demosaicking range',
              'burst sub-pixel only', 'burst sub-pixel only (fine)')
SPLIT_SIGMA = 3.0                  # HR px: Gaussian with G(F_N) = 1/2
BOUNDARY = 40                      # official boundary_ignore
# orthonormal opponent basis: rows luma, red-blue, green-magenta (|e|^2 is preserved per pixel)
OPPONENT = torch.tensor([[1 / math.sqrt(3)] * 3,
                         [1 / math.sqrt(2), 0.0, -1 / math.sqrt(2)],
                         [1 / math.sqrt(6), -2 / math.sqrt(6), 1 / math.sqrt(6)]])


# ----------------------------------------------------------------------------- backend / model

def init_backend(cpu=False):
    """Returns (device, build_network). Must run before anything imports burstISP.archs."""
    if torch.cuda.is_available() and not cpu:
        import burstISP.archs  # noqa: F401  (populates ARCH_REGISTRY: needs mamba_ssm + DCNv4)
        from burstISP.archs import build_network
        return torch.device('cuda'), build_network
    # smoke-test mode: kgts_cpu_checks installs the mamba_ssm reference-scan stub and a bare
    # burstISP.archs package on import (and imports the KGTSMamba modules, registering them)
    sys.path.insert(0, os.path.join(REPO, 'analysis'))
    import kgts_cpu_checks  # noqa: F401
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    return torch.device('cpu'), sys.modules['burstISP.archs'].build_network


def load_opt(path):
    from burstISP.utils.options import ordered_yaml
    with open(path) as f:
        return yaml.load(f, Loader=ordered_yaml()[0])


def resolve_ckpt(opt, ckpt, exp_root):
    """A path, an iteration number, or 'latest' (highest net_g_<iter>.pth, else net_g_latest.pth)
    under <exp_root>/<name>/models."""
    if os.path.isfile(ckpt):
        return os.path.abspath(ckpt)
    models = os.path.join(exp_root, opt['name'], 'models')
    if ckpt.isdigit():
        path = os.path.join(models, f'net_g_{ckpt}.pth')
    elif ckpt == 'latest':
        its = [(int(m), p) for p in glob.glob(os.path.join(models, 'net_g_*.pth'))
               for m in [os.path.basename(p)[6:-4]] if m.isdigit()]
        path = max(its)[1] if its else os.path.join(models, 'net_g_latest.pth')
    else:
        path = ckpt
    if not os.path.isfile(path):
        raise FileNotFoundError(f'checkpoint not found: {path} (from --ckpt {ckpt})')
    return path


def build_model(opt, ckpt, device, build_network, weights='ema'):
    """The config's network_g with the checkpoint's params_ema (weights='ema', falls back to
    params) or params, strict. Returns (model in eval mode, the state-dict key used)."""
    net = build_network(dict(opt['network_g']))
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    key = 'params_ema' if weights == 'ema' and 'params_ema' in sd else 'params'
    if key not in sd:                       # a bare state dict
        net.load_state_dict(sd, strict=True)
        key = '<root>'
    else:
        net.load_state_dict(sd[key], strict=True)
    if type(net).__name__ != 'KGTSMamba':
        raise TypeError(f'these diagnostics are written for KGTSMamba, the config builds {type(net).__name__}')
    return net.to(device).eval(), key


def ref_slot(model, n):
    """The slot the network reads the keyframe from in an n-frame burst (KGTSMamba.forward)."""
    r = model.align.ref_idx
    return (n // 2 if r is None else r) % n


def arrange(frames, keep, slot):
    """frames: (B, F, ...) in generator order, keyframe at 0. keep: generator indices of the
    non-key frames to use, in scan order. Returns (B, len(keep) + 1, ...) with the keyframe at
    `slot` (= ref_slot(model, len(keep) + 1)), and the generator index of every slot."""
    keep = list(keep)
    order = keep[:slot] + [0] + keep[slot:]
    return frames[:, order], order


class Routing:
    """Replaces torch.nn.functional.gumbel_softmax (MambaIRv2's ASSM routes with it, in eval too).

    'paired': the Gumbel noise of the k-th call of a forward comes from a generator seeded with
    (seed, k) and has the logits' shape minus the batch dim, broadcast over the batch. Every
    forward is then a legitimate draw of the model's eval-time noise, every batch element and
    every forward of the run gets the same draw (reset() before each forward), and paired arms
    differ only in what is being compared. 'argmax': no noise. 'stochastic': untouched.
    The output is computed exactly like F.gumbel_softmax(hard=True)."""

    def __init__(self, mode='paired', seed=0):
        self.mode, self.seed, self.calls = mode, seed, 0
        self._orig = F.gumbel_softmax
        if mode != 'stochastic':
            torch.nn.functional.gumbel_softmax = self._gumbel

    def reset(self):
        self.calls = 0

    def _gumbel(self, logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
        if self.mode == 'argmax':
            y = (logits / tau).softmax(dim)
        else:
            g = torch.Generator(device=logits.device)
            g.manual_seed(self.seed * 1_000_003 + self.calls)
            self.calls += 1
            noise = -torch.empty(logits.shape[1:], device=logits.device, dtype=logits.dtype) \
                .exponential_(generator=g).log()
            y = ((logits + noise) / tau).softmax(dim)
        if not hard:
            return y
        idx = y.max(dim, keepdim=True)[1]
        return torch.zeros_like(logits).scatter_(dim, idx, 1.0) - y.detach() + y

    def close(self):
        torch.nn.functional.gumbel_softmax = self._orig


class Runner:
    """model(lq) under no_grad with the run's precision and routing. Outputs come back fp32."""

    def __init__(self, model, device, routing='paired', seed=0, amp=False):
        self.model, self.device, self.amp = model, device, amp
        self.routing = Routing(routing, seed)

    def __call__(self, lq, return_aux=False):
        self.routing.reset()
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            out = self.model(lq.to(self.device, non_blocking=True), return_aux=return_aux)
        if return_aux:
            out, aux = out
            return out.float(), {k: (v.float() if torch.is_tensor(v) else v) for k, v in aux.items()}
        return out.float()


# ----------------------------------------------------------------------------- data

class ValBursts(torch.utils.data.Dataset):
    """The official SyntheticBurstVal set (300 bursts), generator order, with the noise levels
    from its meta_info.pkl (nan if absent)."""

    def __init__(self, root):
        from burstISP.data.dbsr.synthetic_burst_val_set import SyntheticBurstVal
        if not os.path.isdir(os.path.join(root, 'bursts')):
            raise FileNotFoundError(f'no SyntheticBurstVal under {root} (expects bursts/ and gt/)')
        self.vs = SyntheticBurstVal(root=root)

    def __len__(self):
        return len(self.vs)

    def __getitem__(self, i):
        burst, gt, meta = self.vs[i]
        return {'burst': burst, 'gt': gt, 'name': meta['burst_name'],
                'shot': float(meta.get('shot_noise_level', float('nan'))),
                'read': float(meta.get('read_noise_level', float('nan')))}


class GeneratedBursts(torch.utils.data.Dataset):
    """SyntheticBurst-protocol bursts WITH the generator's flow, deterministic per index.

    Same generator, constants and crop as SyntheticBurstDataset's training samples (official
    DBSR settings; no flip), from the Zurich RAW-to-RGB split given -- 'test' is the split
    SyntheticBurstVal was generated from, so these are validation-like bursts that ship flow.
    Image i is spread evenly over the split; every random draw (crop, motion, CCM, gains, noise)
    is seeded from (seed, i), so every arm and every rerun sees the same bursts."""

    def __init__(self, root, split='test', n=300, seed=0, frames=14):
        self.files = sorted(glob.glob(os.path.join(root, split, 'canon', '*.jpg')),
                            key=lambda p: (0, int(os.path.basename(p)[:-4])) if os.path.basename(p)[:-4].isdigit()
                            else (1, os.path.basename(p)))
        if not self.files:
            raise FileNotFoundError(f'no Zurich images under {os.path.join(root, split, "canon")}')
        self.n, self.seed, self.frames, self.split = min(n, len(self.files)), seed, frames, split

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import cv2
        from burstISP.data.dbsr import processing_utils as prutils
        from burstISP.data.dbsr import synthetic_burst_generation as syn
        from burstISP.data.dbsr.data_format_utils import npimage_to_torch
        from burstISP.data.synthetic_burst_dataset import (BURST_TRANSFORMATION_PARAMS, CROP_SZ, DOWNSAMPLE_FACTOR,
                                                           IMAGE_PROCESSING_PARAMS)
        s = self.seed * 1_000_003 + i
        random.seed(s)
        np.random.seed(s % 2 ** 32)
        torch.manual_seed(s)
        path = self.files[i * len(self.files) // self.n]
        img = npimage_to_torch(cv2.imread(path, cv2.IMREAD_COLOR), normalize=True, input_bgr=True)
        bc = BURST_TRANSFORMATION_PARAMS['border_crop']
        crop = prutils.random_resized_crop(img, [c + 2 * bc for c in CROP_SZ])
        burst, gt, _, flow, meta = syn.rgb2rawburst(crop, self.frames, DOWNSAMPLE_FACTOR,
                                                    burst_transformation_params=dict(BURST_TRANSFORMATION_PARAMS),
                                                    image_processing_params=dict(IMAGE_PROCESSING_PARAMS),
                                                    interpolation_type='bilinear')
        return {'burst': burst.float(), 'gt': gt[:, bc:-bc, bc:-bc].float(), 'flow': flow.float(),
                'name': f'{self.split}_{i:04d}_{os.path.basename(path)[:-4]}',
                'shot': float(meta['shot_noise_level']), 'read': float(meta['read_noise_level'])}


def loader(ds, args):
    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    return torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                       pin_memory=torch.cuda.is_available())


def oracle_flow(flow_vectors):
    """Generator flow_vectors (..., 2, 2h, 2w): FORWARD field, LR-RGB px -> the BACKWARD field in
    packed px (..., 2, h, w), the field the taps read (token.tap_pos: backward) and LK returns,
    projected exactly as MambaFusionModel.flow_loss projects train.flow_target: backward onto lv1.
    analysis/kgts_sanity.py's geometry stage checks this lands every tap within 0.005 packed px of
    the generator's sample."""
    from burstISP.models.mambafusion_model import backward_flow
    shp = flow_vectors.shape
    b = backward_flow(flow_vectors.reshape(-1, 2, *shp[-2:]).float())
    return (F.avg_pool2d(b, 2) * 0.5).view(*shp[:-2], shp[-2] // 2, shp[-1] // 2)


# ----------------------------------------------------------------------------- metric / spectra

def quantise(x):
    """The official metric's 14-bit truncation (calculate_psnr_synburst)."""
    return (x.clamp(0.0, 1.0) * 2 ** 14).short().float() / 2 ** 14


def crop(x, b=BOUNDARY):
    return x[..., b:-b, b:-b]


def error(pred, gt):
    """(..., 3, H, W) -> the official metric's error image on its crop."""
    return crop(quantise(pred)) - crop(gt.to(pred.device))


def psnr_from_mse(mse):
    return -10.0 * math.log10(max(float(mse), 1e-20))


def gblur(x, sigma):
    """Separable Gaussian, reflect padding. x: (M, C, h, w)."""
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k, C = k / k.sum(), x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


class Spectrum:
    """Band accounting on (C, h, w) images (one per call), in MSE units: the 5 bands of a squared
    error image sum to its MSE = mean over channels and pixels of e^2.

    fft(x): spectrum of the mirror-extended image, (C, 2h, 2w) complex.
    bands(X, Y=None): per band, sum Re(X conj Y) / norm -- energy if Y is None, else a cross term.
    split(X): [low, high] energy with the soft Gaussian partition (G(f), 1 - G(f))."""

    def __init__(self, h, w, device, sigma=SPLIT_SIGMA):
        fy = torch.fft.fftfreq(2 * h, device=device)
        fx = torch.fft.fftfreq(2 * w, device=device)
        f = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()                       # cycles / HR px
        r = f / F_N
        band = sum((r >= e).long() for e in BAND_EDGES)                          # 0..4
        self.onehot = F.one_hot(band, len(BAND_NAMES)).float()                  # (2h, 2w, 5)
        self.low = torch.exp(-2 * math.pi ** 2 * sigma ** 2 * f ** 2)           # Gaussian amplitude response
        self.h, self.w = h, w

    def fft(self, x):
        x = torch.cat([x, x.flip(-1)], -1)
        return torch.fft.fft2(torch.cat([x, x.flip(-2)], -2).float())

    def _norm(self, C):
        # sum |X|^2 over the extension = 4hw * 4 * sum |x|^2 (Parseval x mirror); MSE divides by C h w
        return 16.0 * (self.h * self.w) ** 2 * C

    def _prod(self, X, Y):
        return (X.abs() ** 2) if Y is None else (X * Y.conj()).real

    def bands(self, X, Y=None):
        p = self._prod(X, Y).sum(0)                                              # over channels
        return torch.einsum('hw,hwb->b', p, self.onehot) / self._norm(X.shape[0])

    def split(self, X, Y=None):
        p = self._prod(X, Y).sum(0)
        lo = (p * self.low).sum()
        return torch.stack([lo, p.sum() - lo]) / self._norm(X.shape[0])


def opponent(x):
    """(3, h, w) RGB -> (3, h, w) [luma, red-blue, green-magenta], orthonormal."""
    return torch.einsum('oc,chw->ohw', OPPONENT.to(x.device, x.dtype), x)


def strata(gt, q=(0.5, 0.85)):
    """Per-pixel structure class of the GT crop (3, h, w): 0 flat (< 50th percentile of the
    luma gradient magnitude, smoothed), 1 texture, 2 edge (top 15%)."""
    y = gt.mean(0, keepdim=True)[None]
    gx = F.pad(y, (1, 1, 0, 0), mode='replicate')
    gy = F.pad(y, (0, 0, 1, 1), mode='replicate')
    g = ((gx[..., 2:] - gx[..., :-2]) ** 2 + (gy[..., 2:, :] - gy[..., :-2, :]) ** 2).sqrt()
    g = gblur(g, 1.5)[0, 0]
    t = torch.quantile(g.flatten().float(), torch.tensor(q, device=g.device))
    return (g >= t[0]).long() + (g >= t[1]).long()


# ----------------------------------------------------------------------------- stats / output

def mci(x):
    """mean and 95% half-width (normal approximation) of a sample, nan-safe."""
    x = np.asarray([v for v in x if v == v], dtype=np.float64)
    if len(x) == 0:
        return float('nan'), float('nan')
    h = 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float('nan')
    return float(x.mean()), float(h)


def fci(x, nd=3):
    m, h = mci(x)
    return f'{m:+.{nd}f} ± {h:.{nd}f}'


def terciles(values):
    """Index lists of the low / mid / high third of `values` (nan values dropped)."""
    idx = [i for i, v in enumerate(values) if v == v]
    idx.sort(key=lambda i: values[i])
    k = len(idx)
    return [idx[:k // 3], idx[k // 3:2 * k // 3], idx[2 * k // 3:]] if k >= 3 else []


def base_parser(doc):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='the training config (network_g, dataset roots, name)')
    ap.add_argument('--ckpt', default='latest',
                    help="checkpoint path, an iteration number, or 'latest' under <exp_root>/<name>/models")
    ap.add_argument('--weights', choices=['ema', 'raw'], default='ema', help='params_ema (default) or params')
    ap.add_argument('--exp_root', default=os.path.join(REPO, 'experiments'))
    ap.add_argument('--out', default=None, help='output dir (default analysis/outputs/diagnostics/<name>/<ckpt>)')
    ap.add_argument('--batch', type=int, default=4, help='bursts per forward')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None, help='first N bursts only (quick runs)')
    ap.add_argument('--routing', choices=['paired', 'argmax', 'stochastic'], default='paired')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--amp', action='store_true', help='bf16 autocast (validation itself runs fp32)')
    ap.add_argument('--cpu', action='store_true', help='smoke-test mode (stubbed mamba_ssm; tiny configs only)')
    return ap


class Context:
    """Everything a diagnostic needs after argument parsing: device, model, runner, output dir, report."""

    def __init__(self, args, script):
        self.args, self.script = args, script
        self.device, build_network = init_backend(args.cpu)
        self.opt = load_opt(args.config)
        self.ckpt = resolve_ckpt(self.opt, args.ckpt, args.exp_root)
        self.model, self.key = build_model(self.opt, self.ckpt, self.device, build_network, args.weights)
        self.run = Runner(self.model, self.device, args.routing, args.seed, args.amp)
        tag = os.path.splitext(os.path.basename(self.ckpt))[0] + ('_ema' if self.key == 'params_ema' else '')
        self.out = args.out or os.path.join(REPO, 'analysis', 'outputs', 'diagnostics', self.opt['name'], tag)
        os.makedirs(self.out, exist_ok=True)
        self.lines = []
        self.t0 = time.time()
        try:
            commit = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'],
                                    capture_output=True, text=True).stdout.strip()
        except OSError:
            commit = '?'
        self.meta = {'script': script, 'config': os.path.abspath(args.config), 'ckpt': self.ckpt,
                     'weights': self.key, 'routing': args.routing, 'amp': args.amp, 'seed': args.seed,
                     'device': torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'cpu',
                     'commit': commit, 'date': time.strftime('%Y-%m-%d %H:%M')}
        n_par = sum(p.numel() for p in self.model.parameters()) / 1e6
        self.log(f'== {script}  {self.opt["name"]}  {os.path.basename(self.ckpt)} [{self.key}]  '
                 f'{n_par:.2f}M params  routing={args.routing}  {"bf16" if args.amp else "fp32"}  '
                 f'{self.meta["device"]}  commit {commit}')

    def log(self, s=''):
        print(s, flush=True)
        self.lines.append(s)

    def val_root(self, override=None):
        return override or self.opt['datasets']['val']['dataroot']

    def zurich_root(self, override=None):
        return override or self.opt['datasets']['train']['dataroot']

    def path(self, suffix):
        return os.path.join(self.out, f'{self.script}{suffix}')

    def save(self, headline, readout, rows=None, extra=None):
        """Writes <script>.txt / .json / _per_image.csv. headline: flat dict of key numbers;
        readout: list of interpretation lines (heuristics, labelled as such)."""
        self.log()
        self.log('-- readout (heuristic thresholds; read the tables, not just this) --')
        for r in readout:
            self.log(f'  * {r}')
        self.log(f'(took {(time.time() - self.t0) / 60:.1f} min; outputs in {self.out})')
        with open(self.path('.txt'), 'w') as f:
            f.write('\n'.join(self.lines) + '\n')
        with open(self.path('.json'), 'w') as f:
            json.dump({'meta': self.meta, 'headline': headline, 'readout': readout, **(extra or {})}, f,
                      indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        if rows:
            keys = list(rows[0].keys())
            with open(self.path('_per_image.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
        self.run.routing.close()


def plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as pyplot
    return pyplot
