import math
import torch
from collections import OrderedDict
from torch.nn import functional as F
from contextlib import nullcontext

from burstISP.utils.registry import MODEL_REGISTRY
from burstISP.loss import build_loss
from burstISP.models.sr_model import SRModel
from burstISP.utils import get_root_logger
from burstISP.utils.img_util import differentiable_benchmark_isp


def backward_flow(flow, iters=3):
    """Forward field -> backward field, same grid and units. flow: (..., 2, h, w).

    The generator's flow_vectors are a FORWARD field f: frame i's pixel j shows
    what the reference shows at j + f(j). A warp that samples frame i at
    p - w(p) (FlowAlign.warp, BurstAlign.warp) needs the BACKWARD field w: the
    reference's p shows what frame i shows at p - w(p), so w(p) = f(p - w(p)).
    They agree to first order only, off by |grad f| * |f|: up to 0.14 LR-RGB px
    at SyntheticBurst's 1 deg rotations and 24 GT px shifts. Solved by fixed
    point (bilinear); each step shrinks the error by |grad f| (<= 0.02).
    """
    shp = flow.shape
    f = flow.reshape(-1, 2, *shp[-2:]).float()
    h, w = shp[-2:]
    # p - w(p) leaves the grid by up to |w|: extrapolate f linearly there (exact for the
    # generator's affine fields) rather than clamping it
    m = min(int(f.abs().max().ceil()) + 1, h - 1, w - 1)
    fpad = 2 * F.pad(f, (m,) * 4, mode='replicate') - F.pad(f, (m,) * 4, mode='reflect')
    ys, xs = torch.meshgrid(torch.arange(h, device=f.device, dtype=f.dtype),
                            torch.arange(w, device=f.device, dtype=f.dtype), indexing='ij')
    b = f
    for _ in range(iters):
        grid = torch.stack(((2 * (xs - b[:, 0] + m) + 1) / (w + 2 * m) - 1,
                            (2 * (ys - b[:, 1] + m) + 1) / (h + 2 * m) - 1), -1)
        b = F.grid_sample(fpad, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return b.reshape(shp)


@MODEL_REGISTRY.register()
class MambaFusionModel(SRModel):
    """MambaFusion model for image restoration."""
    def __init__(self, opt):
        # Set before super().__init__, which may call init_training_settings.
        # A model built for inference never runs optimize_parameters, but this
        # keeps the attribute defined either way.
        self.cri_flow = self.cri_aux = self.cri_photo = None
        self.burst_aug = None
        super(MambaFusionModel, self).__init__(opt)

    @staticmethod
    def _schedule(sched, name):
        """(milestones, values, unit) of a piecewise-linear lambda(t) config dict."""
        ms, vs = list(sched.get('milestones', [0])), list(sched.get('values', [1.0]))
        if len(ms) != len(vs):
            raise ValueError(f'train.{name}.milestones and .values must be the same length, got {len(ms)} and {len(vs)}')
        unit = sched.get('unit', 'micro')
        if unit not in ('micro', 'iter'):
            raise ValueError(f"train.{name}.unit must be 'micro' or 'iter', got {unit!r}")
        return ms, vs, unit

    def _lambda(self, sched, micro_step):
        """lambda(t) of a _schedule, linear between milestones, flat outside them.
        optimize_parameters is called with train.py's MICRO-step (it counts every forward, so
        with accumulation_steps: k it runs k x faster than the iteration counter everything
        else is logged and scheduled by). unit: 'micro' keeps that (the historical behaviour);
        'iter' converts to optimizer iterations, the unit the milestones are written in."""
        ms, vs, unit = sched
        t = micro_step
        if unit == 'iter':
            acc = self.opt['train'].get('accumulation_steps', 1)
            t = (micro_step + acc - 1) // acc
        if t <= ms[0]:
            return float(vs[0])
        if t >= ms[-1]:
            return float(vs[-1])
        for i in range(1, len(ms)):
            if t <= ms[i]:
                span = max(ms[i] - ms[i - 1], 1)
                return float(vs[i - 1] + (t - ms[i - 1]) / span * (vs[i] - vs[i - 1]))
        return float(vs[-1])

    def init_training_settings(self):
        """SRModel's setup plus the optional alignment-flow criterion.

        The flow term supervises BurstAlign's per-level flow estimates against
        the generator's ground-truth `flow_vectors` (PLAN L6 step 8). It is a
        curriculum, not a constraint: lambda decays to zero so that late in
        training the network is free to pick a sampling pattern that beats
        perfect geometric registration.
        """
        super(MambaFusionModel, self).init_training_settings()
        train_opt = self.opt['train']

        if train_opt.get('flow_opt'):
            self.cri_flow = build_loss(train_opt['flow_opt']).to(self.device)
            # Piecewise-linear lambda(t). Duplicate a milestone to get a step.
            self.flow_sched = self._schedule(train_opt.get('flow_lambda', {}), 'flow_lambda')
            self.flow_milestones, self.flow_values = self.flow_sched[:2]
            # Equal weight per level; keys must exist in the arch's aux dict.
            self.flow_levels = list(train_opt.get('flow_levels', ['lv1', 'lv2', 'lv3']))
            # 'forward': the generator's flow_vectors as they are. 'backward': the field a
            # p - flow(p) warp needs (see backward_flow); KGTSMamba pairs it with token.tap_pos.
            self.flow_target = train_opt.get('flow_target', 'forward')
            if self.flow_target not in ('forward', 'backward'):
                raise ValueError(f"train.flow_target must be 'forward' or 'backward', got {self.flow_target!r}")
        else:
            self.cri_flow = None

        # Burst-only auxiliary reconstruction (KGTSMamba aux_head): aux['burst'] vs gt.
        bare = self.get_bare_model(self.net_g)
        has_aux = getattr(bare, 'aux', None) is not None
        if train_opt.get('aux_opt'):
            if not has_aux:
                raise ValueError('train.aux_opt is set but network_g has no aux head (KGTSMamba aux_head: true)')
            self.cri_aux = build_loss(train_opt['aux_opt']).to(self.device)
            self.aux_sched = self._schedule(train_opt.get('aux_lambda', {}), 'aux_lambda')
        elif has_aux:
            raise ValueError('network_g has an aux head but train.aux_opt is not set: its parameters would '
                             'get no gradient (DDP with find_unused_parameters: false fails on that)')

        # Self-supervised photometric flow loss, for bursts without flow_vectors (real data): the
        # keyframe vs every frame warped by FlowAlign's lv1 flow, on the blurred channel mean
        # (blur: noise and Bayer aliasing), masked to in-view pixels, + first-order smoothness.
        if train_opt.get('photo_opt'):
            self.cri_photo = build_loss(train_opt['photo_opt']).to(self.device)
            self.photo_sched = self._schedule(train_opt.get('photo_lambda', {}), 'photo_lambda')
            self.photo_smooth = float(train_opt.get('photo_smooth', 0.1))
            self.photo_sigma = float(train_opt.get('photo_sigma', 1.0))
            self.photo_when = train_opt.get('photo_when', 'no_gt')
            if self.photo_when not in ('no_gt', 'always'):
                raise ValueError(f"train.photo_when must be 'no_gt' or 'always', got {self.photo_when!r}")

        # Burst augmentation: random subset / order of the non-key frames (see augment_burst).
        self.burst_aug = dict(train_opt['burst_aug']) if train_opt.get('burst_aug') else None

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device) # [B, N, C, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device) # [B, C, H, W]
        # Ground-truth geometric flow, [B, N, 2, 2h, 2w] in LR-RGB pixel units.
        # Only SyntheticBurstDataset in train mode ships it; the official val
        # set does not, so this is None on every validation pass.
        self.flow_gt = data['flow_vectors'].to(self.device) if 'flow_vectors' in data else None
        # [B, N, 1, 2h, 2w], 0 where the generator's flow does not hold (SyntheticBurstDataset outliers)
        self.flow_mask = data['flow_mask'].to(self.device) if 'flow_mask' in data else None

    def flow_lambda(self, current_iter):
        """lambda(t), linearly interpolated between the configured milestones
        and held flat outside them. current_iter is optimize_parameters' micro-step;
        see _lambda for train.flow_lambda.unit."""
        return self._lambda(self.flow_sched, current_iter)

    def ref_index(self, n):
        """The keyframe slot the network reads for an n-frame burst (its ref_idx, else n // 2)."""
        bare = self.get_bare_model(self.net_g)
        ref = getattr(getattr(bare, 'align', None), 'ref_idx', None)
        return (n // 2 if ref is None else ref) % n

    def augment_burst(self, lq, flow_gt, flow_mask=None):
        """train.burst_aug = {prob, min_frames, shuffle}. The burst is a set: with probability
        prob keep the keyframe and a random subset of the other frames (size >= min_frames - 1),
        and with shuffle permute the non-key frames' order on every batch. The keyframe is moved
        to the slot the network reads (n // 2 of the new length). Trains robustness to burst
        length (real bursts vary; frames get dropped) and to frame order (the scan is a
        recurrence). The flow ground truth (and its mask) is indexed alongside."""
        aug = self.burst_aug
        B, N = lq.shape[:2]
        ref = self.ref_index(N)
        others = [i for i in range(N) if i != ref]
        if aug.get('shuffle', True):
            others = [others[i] for i in torch.randperm(len(others)).tolist()]
        n = N
        if N > 1 and torch.rand(()).item() < aug.get('prob', 0.0):
            n = int(torch.randint(max(1, aug.get('min_frames', 2)), N + 1, ()).item())
        others = others[:n - 1]
        order = others[:self.ref_index(n)] + [ref] + others[self.ref_index(n):]
        pick = lambda t: None if t is None else t[:, order]
        return lq[:, order], pick(flow_gt), pick(flow_mask)

    def photo_loss(self, flows, lq):
        """See init_training_settings (photo_opt). flows['lv1']: (B, N, 2, h1, w1), that grid's px."""
        from burstISP.archs.KGTSMamba.flow_align_arch import gaussian_blur
        fl = flows['lv1'].float()
        B, N = fl.shape[:2]
        h, w = lq.shape[-2:]
        if fl.shape[-2:] != (h, w):                        # Bayer grid: fold to packed like PostAlign
            fl = F.avg_pool2d(fl.flatten(0, 1), kernel_size=2).view(B, N, 2, h, w) * 0.5
        ref = self.ref_index(N)
        lum = gaussian_blur(lq.float().mean(2).flatten(0, 1)[:, None], self.photo_sigma).view(B, N, 1, h, w)
        ys, xs = torch.meshgrid(torch.arange(h, device=fl.device, dtype=fl.dtype),
                                torch.arange(w, device=fl.device, dtype=fl.dtype), indexing='ij')
        sx, sy = xs - fl[:, :, 0], ys - fl[:, :, 1]            # content of ref p sits at p - flow(p)
        grid = torch.stack(((2 * sx + 1) / w - 1, (2 * sy + 1) / h - 1), -1).flatten(0, 1)
        warped = F.grid_sample(lum.flatten(0, 1), grid, mode='bilinear', padding_mode='border',
                               align_corners=False).view(B, N, 1, h, w)
        inside = ((sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)).float().unsqueeze(2)
        keep = torch.arange(N, device=fl.device) != ref
        target = lum[:, ref:ref + 1].expand_as(warped)
        l_photo = self.cri_photo(warped[:, keep].flatten(0, 1), target[:, keep].flatten(0, 1),
                                 weight=inside[:, keep].flatten(0, 1))
        smooth = (fl[..., 1:, :] - fl[..., :-1, :]).abs().mean() + (fl[..., 1:] - fl[..., :-1]).abs().mean()
        return l_photo + self.photo_smooth * smooth

    def flow_loss(self, flows, gt=None, mask=None):
        """Charbonnier between each level's predicted flow and the generator's
        flow (gt, default self.flow_gt), average-pooled to that level. mask
        [B, N, 1, 2h, 2w]: pixels where the ground truth does not hold (outliers)
        are left out; a level's pixel counts only if all of its footprint is valid.

        The sign needs no negation here. `synthetic_burst_dataset.py:196` says
        the content the reference sees at p sits at `p - flow(p)` in frame i,
        and `BurstAlign.warp` samples at exactly `p - flow`, so the predicted
        flow and `flow_vectors` share a sign. The negation into DCN
        offset units lives in `BurstAlign.scatter_flow`, downstream of this.
        They share the convention only to first order, though: flow_vectors is
        a forward field and a `p - flow` warp wants the backward one (see
        backward_flow; up to 0.07 packed px apart). train.flow_target: backward
        supervises the latter.

        The reference frame is included on purpose: its ground-truth flow is
        identically zero (`flow_vectors = sample_pos_inv_all - ...[:1]`), so it
        contributes a free "do not move" constraint.

        The per-level conversion is derived from the tensors' own shapes, so it
        is correct in both the Bayer and the packed domain: pooling GT from
        96x96 to a level of side h scales magnitudes by h/96, which reproduces
        the x1 / x0.5 / x0.25 of `oracle_warp`'s pattern.
        """
        gt = self.flow_gt if gt is None else gt
        B, N = gt.shape[0], gt.shape[1]
        gt = gt.reshape(B * N, 2, gt.shape[-2], gt.shape[-1]).float()
        if self.flow_target == 'backward':
            gt = backward_flow(gt)
        gt_h = gt.shape[-2]

        missing = [lv for lv in self.flow_levels if lv not in flows]
        if missing:
            raise KeyError(f'train.flow_levels names {missing}, which BurstAlign does not '
                           f'return; available levels are {sorted(flows)}')

        l_flow = 0
        for lvl in self.flow_levels:
            pred = flows[lvl].reshape(B * N, 2, *flows[lvl].shape[-2:]).float()
            h, w = pred.shape[-2:]
            # adaptive_avg_pool2d == avg_pool2d for the integer ratios here, and
            # degrades gracefully if a crop size is not an exact power-of-two
            # multiple of the level size.
            target = F.adaptive_avg_pool2d(gt, (h, w)) * (h / gt_h)
            weight = None
            if mask is not None:
                m = mask.reshape(B * N, 1, *mask.shape[-2:]).float()
                weight = (F.adaptive_avg_pool2d(m, (h, w)) > 0.999).float()
            l_flow = l_flow + self.cri_flow(pred, target, weight=weight)

        return l_flow / max(len(self.flow_levels), 1)

    @staticmethod
    def compand(x, mu=24.0):
        # Signed mu law
        return torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)

    # Modified from sr_model.py to include bf16 and alignment loss
    def optimize_parameters(self, current_iter):
        accumulation_steps = self.opt['train'].get('accumulation_steps', 1)

        if (current_iter - 1) % accumulation_steps == 0:
            self.optimizer_g.zero_grad()

        is_sync_step = (current_iter % accumulation_steps == 0)
        
        # Use DDP's no_sync() if we are accumulating, otherwise do nothing
        sync_context = self.net_g.no_sync if (not is_sync_step and self.opt['dist']) else nullcontext
        
        # Forward Pass
        with sync_context():
            lq, flow_gt, flow_mask = self.lq, self.flow_gt, getattr(self, 'flow_mask', None)
            if self.burst_aug:
                lq, flow_gt, flow_mask = self.augment_burst(lq, flow_gt, flow_mask)
            # lambda is read up front: when the curriculum has decayed to zero
            # there is nothing to gain from carrying the aux flows. The flow
            # head stays on the gradient path either way, through the DCN
            # offsets, so find_unused_parameters can stay false.
            lam = self.flow_lambda(current_iter) if self.cri_flow else 0.0
            want_flow = self.cri_flow is not None and lam > 0 and flow_gt is not None
            lam_photo = self._lambda(self.photo_sched, current_iter) if self.cri_photo else 0.0
            want_photo = lam_photo > 0 and (flow_gt is None or self.photo_when == 'always')
            # the aux head's loss is always taken (its weight may be 0): unused parameters break DDP
            want_aux = self.cri_aux is not None

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if want_flow or want_photo or want_aux:
                    self.output, aux = self.net_g(lq, return_aux=True)
                else:
                    self.output, aux = self.net_g(lq), None

            self.output = self.output.float()
            l_total = 0
            loss_dict = OrderedDict()

            # Projection into SRGB via Signed Mu-Law. Config-gated: compand
            # defaults to true (RealBSR behavior); PLAN.md L3 sets it to false
            # to train on plain linear RGB.
            if self.opt['train'].get('compand', True):
                cp, cg = self.compand(self.output), self.compand(self.gt)
            else:
                cp, cg = self.output, self.gt

            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(cp, cg)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            # Edge Loss
            if self.cri_edge:
                l_edge = self.cri_edge(cp, cg)
                l_total += l_edge
                loss_dict['l_edge'] = l_edge

            # Alignment flow loss (curriculum; lambda decays to zero)
            if want_flow:
                l_flow = lam * self.flow_loss(aux['flows'], flow_gt, flow_mask)
                l_total += l_flow
                loss_dict['l_flow'] = l_flow
                loss_dict['flow_lambda'] = torch.tensor(lam, device=self.device)

            # Self-supervised photometric flow loss (bursts without flow ground truth)
            if want_photo:
                l_photo = lam_photo * self.photo_loss(aux['flows'], lq)
                l_total += l_photo
                loss_dict['l_photo'] = l_photo

            # Burst-only auxiliary reconstruction (aux_head)
            if want_aux:
                pa = aux['burst'].float()
                pa = self.compand(pa) if self.opt['train'].get('compand', True) else pa
                l_aux = self._lambda(self.aux_sched, current_iter) * self.cri_aux(pa, cg)
                l_total += l_aux
                loss_dict['l_aux'] = l_aux

            # Backpropagation
            l_total = l_total / accumulation_steps
            l_total.backward()

            if is_sync_step:
                clip_norm = self.opt['datasets']['train'].get('grad_clip_norm',1.0)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), clip_norm)
                self.optimizer_g.step()

                if self.ema_decay > 0:
                    self.model_ema(decay=self.ema_decay)

                self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def test(self):
        if hasattr(self, 'net_g_ema'):
            model = self.net_g_ema
        else:
            model = self.get_bare_model(self.net_g)

        model.eval()
        with torch.no_grad():
            self.output = model(self.lq)
        model.train()

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        # Log ReZero alpha
        bare_model = self.get_bare_model(self.net_g)
        if hasattr(bare_model, 'alpha_residual'):
            alpha_val = bare_model.alpha_residual.item()
            log_str += f'\t # rezero_alpha: {alpha_val:.10f}\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
            if hasattr(bare_model, 'alpha_residual'):
                tb_logger.add_scalar('train/rezero_alpha', alpha_val, current_iter)
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq[:, self.lq.shape[1]//2].detach().cpu()  # show center frame
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
