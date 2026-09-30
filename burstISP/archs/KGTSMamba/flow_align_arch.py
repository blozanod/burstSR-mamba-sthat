import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from burstISP.utils.registry import ARCH_REGISTRY


def affine_flow_fit(flow, margin=4, iters=3, size=None):
    """Per-frame robust least-squares fit of flow(p) = M p + t (6 dof), differentiable in `flow`.

    flow: (M, 2, h, w), any dtype; fitted in fp32 and returned in flow's dtype.
    margin: px dropped at each border of the fit region, where content leaves the view.
    size: (h, w) of the valid region at the top-left (the unpadded image); default all.

    Weights are Cauchy IRLS (detached), so textureless or occluded patches, where the
    dense estimate wanders, do not pull the fit. The model is exact for SyntheticBurst:
    the generator moves each frame by one affine map (shift + rotation), and its
    flow_vectors are that map's flow field (fit residual 1e-5 px).
    """
    M, _, h, w = flow.shape
    hv, wv = size or (h, w)
    my, mx = min(margin, (hv - 1) // 2), min(margin, (wv - 1) // 2)
    # fp32 throughout: under bf16 autocast the matmuls would come out bf16 and linalg.solve
    # refuses mixed dtypes (A is promoted back to fp32 by `eye`, B is not)
    with torch.autocast(device_type=flow.device.type, enabled=False):
        ys, xs = torch.meshgrid(torch.arange(h, device=flow.device, dtype=torch.float32),
                                torch.arange(w, device=flow.device, dtype=torch.float32), indexing='ij')
        # centred, unit-range coordinates keep the 3x3 normal equations well conditioned
        X = torch.stack([xs / w - 0.5, ys / h - 0.5, torch.ones_like(xs)], -1).view(1, h * w, 3)
        Y = flow.float().flatten(2).transpose(1, 2)                         # (M, hw, 2)
        m = torch.zeros(h, w, device=flow.device)
        m[my:hv - my, mx:wv - mx] = 1
        w0 = m.view(1, h * w).expand(M, -1)
        wt = w0
        eye = 1e-6 * torch.eye(3, device=flow.device)
        for it in range(iters + 1):
            Xw = X * wt[..., None]
            beta = torch.linalg.solve(Xw.transpose(1, 2) @ X + eye, Xw.transpose(1, 2) @ Y)   # (M, 3, 2)
            if it == iters:
                break
            with torch.no_grad():
                r = (Y - X @ beta).norm(dim=-1)                              # (M, hw)
                c = 2 * (r * w0).sum(1, keepdim=True) / w0.sum(1, keepdim=True)
                wt = w0 / (1 + (r / c.clamp_min(1e-3)) ** 2)
        fit = (X @ beta).transpose(1, 2).reshape(M, 2, h, w)
    return fit.to(flow.dtype)


def gaussian_blur(x, sigma):
    """Separable Gaussian, replicate padding. x: (M, C, h, w)."""
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k, C = k / k.sum(), x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


@torch.no_grad()
def affine_lk(burst, ref_idx, init=None, size=None, sigmas=(2.0, 1.4, 1.0, 0.7), iters=4, margin=2):
    """Per-frame robust photometric affine registration to the keyframe (Gauss-Newton / Lucas-Kanade).

    burst: (B, N, C, h, w) packed. Returns the backward flow (B, N, 2, h, w), fp32 packed px, one
    affine map per frame, the keyframe's exactly 0. Residual on the channel-mean image,
    r(p) = I_i(p - w(p)) + b - I_ref(p), w(p) = M p + t: 6 motion + 1 offset per frame, Tukey
    biweight IRLS on the median-centred residual with a MAD scale, coarse to fine by blur (sigmas),
    Levenberg-damped, steps clipped to 1 px. Starts from zero motion and, if `init` (a flow of the
    same shape) is given, from its affine fit too; per frame the start with the lower final median
    absolute deviation wins.

    Why: KGTS reads each tap's sub-pixel position straight off this flow, and nothing downstream can
    correct it. On DBSR bursts (32x32 packed crops, 100 BSD100 bursts x 13 frames, all noise levels)
    it lands at ~0.08 packed px mean EPE (median ~0.065) from zero motion, the same as from a start
    0.33 px off, against ~0.35 for the affine fit of the learned flow reported in KGTSAlign's
    docstring. It needs no training, so it is right from step 0; ~7 MMACs per 48x48 frame (two
    starts x 16 iterations), ~0.1 GMACs per 14-frame burst. Detached: it is a measurement of the
    burst (like the generator's flow), not a learned estimate.
    Robustness: a free photometric gain and Cauchy / mean-|r| weights (the first version) let a
    moving object covering 1/16 of the frame pull the estimate to 0.20 px (0.09 now). An object
    covering 1/4 of the frame still pulls it to ~1.3 px: that is what KGTSAlign's `local` hybrid
    is for -- a wrong global estimate is a coherent residual, so the gate hands such regions to
    the dense flow.
    """
    B, N, C, h, w = burst.shape
    hv, wv = size or (h, w)
    dev = burst.device
    with torch.autocast(device_type=dev.type, enabled=False):
        lum = burst.float().mean(2, keepdim=True).flatten(0, 1)                      # (M, 1, h, w)
        refl = lum.view(B, N, 1, h, w)[:, ref_idx:ref_idx + 1].expand(B, N, 1, h, w).flatten(0, 1)
        M = B * N
        ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                                torch.arange(w, device=dev, dtype=torch.float32), indexing='ij')
        X = torch.stack([xs / w - 0.5, ys / h - 0.5, torch.ones_like(xs)], -1).view(h * w, 3)
        m0 = torch.zeros(h, w, device=dev)
        m0[margin:hv - margin, margin:wv - margin] = 1
        m0 = m0.view(1, h * w)
        eye = torch.eye(7, device=dev)
        nan = torch.tensor(float('nan'), device=dev)

        def run(beta):
            b = torch.zeros(M, 1, device=dev)
            for sig in sigmas:
                I, R = gaussian_blur(lum, sig), gaussian_blur(refl, sig).view(M, -1)
                for _ in range(iters):
                    fl = (X @ beta).transpose(1, 2)                                   # (M, 2, hw)
                    sx, sy = xs.reshape(1, -1) - fl[:, 0], ys.reshape(1, -1) - fl[:, 1]
                    grid = torch.stack(((2 * sx + 1) / w - 1, (2 * sy + 1) / h - 1), -1).view(M, h, w, 2)
                    Iw = F.grid_sample(I, grid, mode='bilinear', padding_mode='border', align_corners=False)
                    gx = (F.pad(Iw, (1, 1, 0, 0), mode='replicate')[..., 2:]
                          - F.pad(Iw, (1, 1, 0, 0), mode='replicate')[..., :-2]).view(M, -1) / 2
                    gy = (F.pad(Iw, (0, 0, 1, 1), mode='replicate')[..., 2:, :]
                          - F.pad(Iw, (0, 0, 1, 1), mode='replicate')[..., :-2, :]).view(M, -1) / 2
                    Iw = Iw.view(M, -1)
                    inside = (sx >= 1) & (sx <= wv - 2) & (sy >= 1) & (sy <= hv - 2)
                    msk = inside.float() * m0
                    r = Iw + b - R
                    rn = torch.where(msk > 0, r, nan)
                    med = rn.nanmedian(1, keepdim=True).values
                    dev_ = (rn - med).abs()
                    s = (1.4826 * dev_.nanmedian(1, keepdim=True).values).clamp_min(1e-6)
                    wt = msk * (1 - ((r - med) / (4.685 * s)) ** 2).clamp_min(0) ** 2     # Tukey biweight
                    J = torch.cat([-gx[..., None] * X, -gy[..., None] * X, torch.ones_like(Iw)[..., None]], -1)
                    Jw = J * wt[..., None]                                            # (M, hw, 7)
                    H = Jw.transpose(1, 2) @ J
                    H = H + 1e-3 * torch.diag_embed(H.diagonal(dim1=1, dim2=2)) + 1e-8 * eye
                    d = -torch.linalg.solve(H, (Jw * r[..., None]).sum(1))           # (M, 7)
                    beta = beta + torch.stack([d[:, 0:3], d[:, 3:6]], -1).clamp(-1.0, 1.0)
                    b = b + d[:, 6:7]
            return beta, dev_.nanmedian(1).values

        starts = [torch.zeros(M, 3, 2, device=dev)]
        if init is not None:                                                         # least-squares affine fit
            Y = init.detach().float().flatten(0, 1).flatten(2).transpose(1, 2)        # (M, hw, 2)
            Xm = X * m0.view(-1, 1)
            starts.append(torch.linalg.solve(Xm.T @ X + 1e-6 * torch.eye(3, device=dev), Xm.T @ Y))
        best, best_cost = run(starts[0])
        for beta in starts[1:]:
            beta, cost = run(beta)
            better = cost < best_cost
            best = torch.where(better.view(M, 1, 1), beta, best)
            best_cost = torch.where(better, cost, best_cost)
        flow = (X @ best).transpose(1, 2).reshape(B, N, 2, h, w)
        flow[:, ref_idx] = 0
    return flow


def local_residual(dense, glob, tau=0.25, win=5):
    """Hybrid flow for bursts with local motion: the global (per-frame affine) flow, plus the dense
    estimate's residual where it is coherently different from it.

    The dense estimate's error is mostly incoherent noise (~0.3-0.5 packed px per pixel); a moving
    object or parallax is a spatially coherent residual. A win x win box filter keeps the latter and
    averages the former down, and the gate ramps from 0 at |smoothed residual| = tau to 1 at 2 tau
    (packed px, detached). On SyntheticBurst (pure affine motion) it stays ~closed; on real bursts it
    hands moving regions back to the dense flow instead of the wrong global model."""
    r = (dense - glob).float()
    B, N = r.shape[:2]
    rs = F.avg_pool2d(r.flatten(0, 1), win, 1, win // 2, count_include_pad=False).view_as(r)
    gate = ((rs.norm(dim=2, keepdim=True) - tau) / tau).clamp(0, 1).detach()
    return glob + gate * rs


class TokenBlock(nn.Module):
    """Per-frame residual block for the token encoder: depthwise 7x7 -> LN -> 2x pointwise MLP
    (ConvNeXt). The last pointwise is zero-init, so a stack of these is an identity at init."""
    def __init__(self, c, k=7, mult=2):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c)
        self.norm = nn.LayerNorm(c)
        self.pw1, self.pw2 = nn.Linear(c, mult * c), nn.Linear(mult * c, c)
        nn.init.zeros_(self.pw2.weight); nn.init.zeros_(self.pw2.bias)

    def forward(self, x):
        y = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x).permute(0, 2, 3, 1)))))
        return x + y.permute(0, 3, 1, 2).to(x.dtype)


class PreAlign(nn.Module):
    """ Projects input RGGB frames into feature space and
    upsamples to Bayer grid resolution via PixelShuffle.

    Allows alignment and downstream modules to operate on pixel grid instead 
    of packed RGGB subpixel space.
    """
    def __init__(self, in_channels=4, num_feat=64):
        super(PreAlign, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, num_feat, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat * 4, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x = self.proj(x)
        x = self.pixel_shuffle(x)
        return x.view(B, N, -1, H * 2, W * 2)

class PostAlign(nn.Module):
    """ Projects Bayer-grid features and flow back to packed RGGB space.

    Mirror of PreAlign: alignment runs on the full Bayer grid (H, W), and the
    rest of the model runs on the packed grid (H/2, W/2), which is ~4x cheaper.

    Features: PixelUnshuffle(2) folds each 2x2 Bayer block into channels, then
    convs compress 4*num_feat -> num_feat. The first conv does the contraction
    so the second runs at num_feat (0.15M at num_feat=64, like PreAlign).

    Flow: parameter-free. One packed pixel covers a 2x2 Bayer block, so the
    flow is average-pooled 2x and halved into packed pixel units -- the same
    projection the synthetic dataset's oracle_warp applies to flow_vectors,
    and in the same sign convention (content the reference sees at p sits at
    p - flow(p)).
    """
    def __init__(self, num_feat=64, out_feat=None):
        super(PostAlign, self).__init__()
        out_feat = out_feat or num_feat
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.proj = nn.Sequential(
            nn.Conv2d(num_feat * 4, out_feat, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, feats, flow):
        """
        Args:
            feats (Tensor): Bayer-grid features, (B, N, C, H, W).
            flow  (Tensor): Bayer-grid flow in Bayer pixel units, (B, N, 2, H, W),
                e.g. FlowAlign's flows['lv1'].

        Returns:
            feats (Tensor): packed features, (B, N, out_feat, H/2, W/2).
            flow  (Tensor): packed flow in packed pixel units, (B, N, 2, H/2, W/2).
        """
        B, N, C, H, W = feats.size()
        if H % 2 or W % 2:
            raise ValueError(f'PostAlign needs an even Bayer grid, got {H}x{W}')
        if flow.shape[-2:] != (H, W):
            raise ValueError(f'flow grid {tuple(flow.shape[-2:])} != feature grid {(H, W)}')

        x = self.pixel_unshuffle(feats.reshape(B * N, C, H, W))
        x = self.proj(x)

        f = F.avg_pool2d(flow.reshape(B * N, 2, H, W).float(), kernel_size=2) * 0.5

        return x.view(B, N, -1, H // 2, W // 2), f.view(B, N, 2, H // 2, W // 2)

@ARCH_REGISTRY.register()
class FlowAlign(nn.Module):
    """
    Alignment module to calculate flow between images in burst

    Uses PCD-style architecture, with a 3-tier pyramid. However,
    it computes no DCN - low-pass filter. Feature warp is downstream
    and implicit.

    Frame count independent. Works with burst of 1 to N.

    Args:
    in_chans: def 4
    num_feat: def 64
    num_frames: def 14. Kept for config compatibility only; not used to build
        any layer and not used to pick the reference frame.
    r: radius for cost_vol
    ref_idx: def None. Position of the reference frame in the burst.
    """
    def __init__(self, in_channels=4, num_feat=64, num_frames=14, r=2, ref_idx=None):
        super(FlowAlign, self).__init__()
        kernel_size = 3

        #  Inits
        self.num_frames = num_frames
        self.num_feat = num_feat
        self.r = r
        self.ref_idx = ref_idx

        # Feature Extraction
        self.feat_extractor_lv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_feat, kernel_size=kernel_size, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=kernel_size, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=kernel_size, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.feat_extractor_lv2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=kernel_size, padding=1, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=kernel_size, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.feat_extractor_lv3 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=kernel_size, padding=1, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=kernel_size, padding=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # flow head + residual flow convs
        self.flow_head_lv3 = nn.Conv2d((2*r+1)**2, 2, kernel_size=kernel_size, padding=1) # cost vol outputs (2r+1)^2 chans
        self.flow_conv_lv2 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, 2, kernel_size=kernel_size, padding=1)
            )
        self.flow_conv_lv1 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, 2, kernel_size=kernel_size, padding=1)
            )
        self.flow_conv_casc = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, 2, kernel_size=kernel_size, padding=1)
            )

        self._init_flow_weights()

    def _init_flow_weights(self):
        """Zero-initialise the residual flow layers.

        At t=0 every refinement stage passes the upsampled coarse flow through
        unchanged, so the lv3 cost-volume head alone drives early training.
        """
        for proj in [self.flow_conv_lv2, self.flow_conv_lv1, self.flow_conv_casc]:
            layer = proj[-1]
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)

    def warp(self, x, flow, align_corners=False):
        """
        Internal only: warps pyramid features so each refinement stage sees the
        residual misalignment. Never applied to the module's outputs.

        x: [B, C, H, W]
        flow: [B, 2, H, W] where flow[:,0] = x_shift, flow[:,1] = y_shift
        align_corners: set to False, determines formula used
        return: [B, C, H, W]
        """
        B, C, H, W = x.size()
        # fp32 coordinates: bf16 cannot hold sub-pixel positions past ~32 px
        flow = flow.float()

        # base grid
        ys, xs = torch.meshgrid(torch.arange(H, device=x.device, dtype=flow.dtype),
                                torch.arange(W, device=x.device, dtype=flow.dtype), indexing='ij')

        sample_x = xs - flow[:,0]
        sample_y = ys - flow[:,1]

        # selects correct formula based on align_corners
        if align_corners:
            gx = 2 * sample_x / (W - 1) - 1
            gy = 2 * sample_y / (H - 1) - 1
        else:
            gx = (2 * sample_x + 1) / W - 1
            gy = (2 * sample_y + 1) / H - 1

        # shift
        grid = torch.stack((gx, gy), -1)
        return F.grid_sample(x.float(), grid, mode='bilinear',
                            padding_mode='zeros', align_corners=align_corners).to(x.dtype)

    def cost_vol(self, ref, cur, r):
        """
        ref: [B, C, H, W]
        cur: [B, C, H, W]
        r: radius (int)
        out: [B, (2r+1)^2, H, W]

        decoder: dx, dy
        """
        B, C, H, W = ref.shape
        ref_n = F.normalize(ref, p=2, dim=1)
        cur_n = F.normalize(cur, p=2, dim=1)
        cur_pad = F.pad(cur_n, (r,r,r,r), mode='constant', value=0)

        cost_list = []
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                cur_slice = cur_pad[:,:, r+dy:r+dy+H, r+dx:r+dx+W]
                cost_list.append((ref_n * cur_slice).sum(dim=1))

        return torch.stack(cost_list, dim=1)

    def up_flow(self, f, size):
        """f: [B, 2, h, w] in level-h pixel units -> [B, 2, *size] in fine pixels

        Upsamples to the finer level's actual size rather than a fixed x2: the
        stride-2 extractors round up on odd inputs, so 2*h can overshoot the
        finer level by one pixel. The per-axis scale keeps the flow in the
        finer level's pixel units either way.
        """
        h, w = f.shape[-2:]
        up = F.interpolate(f, size=size, mode='bilinear', align_corners=False)
        scale = up.new_tensor([size[1] / w, size[0] / h]).view(1, 2, 1, 1)
        return up * scale

    def _ref_pairs(self, feat, B, N, ref_idx):
        """Returns the reference features repeated once per frame, [B*N, C, h, w]."""
        C, h, w = feat.shape[-3:]
        ref = feat.view(B, N, C, h, w)[:, ref_idx]
        return ref.unsqueeze(1).expand(B, N, C, h, w).reshape(B * N, C, h, w)

    def forward(self, x, ref_idx=None):
        """
        Args:
            x (Tensor): Burst of shape (B, N, C, H, W). N may be any value >= 1
                and need not match ``num_frames``.
            ref_idx (int, optional): index of the reference frame for this
                call. Overrides the constructor's ``ref_idx``; if both are
                None, N // 2 is used. Negative values count from the end.

        Returns:
            flows (dict): the estimated flow at each pyramid level, each
                (B, N, 2, h, w) in that level's own pixel units and in the
                same sign convention as the generator's ``flow_vectors`` --
                the content the reference sees at p sits at ``p - flow(p)``.
                The reference frame's own entry is its self-flow (~0).
            feats (Tensor): the unwarped lv1 features, (B, N, num_feat, H, W),
                for the downstream module to warp with ``flows['lv1']``.
        """
        B, N, C, H, W = x.size()
        if ref_idx is None:
            ref_idx = self.ref_idx
        if ref_idx is None:
            ref_idx = N // 2
        if not -N <= ref_idx < N:
            raise IndexError(f'ref_idx {ref_idx} out of range for a burst of {N} frames')
        ref_idx = ref_idx % N
        x_reshaped = x.reshape(B * N, C, H, W)

        # feat extraction
        feat1 = self.feat_extractor_lv1(x_reshaped) # B C H W
        feat2 = self.feat_extractor_lv2(feat1) # B C H/2 W/2
        feat3 = self.feat_extractor_lv3(feat2) # B C H/4 W/4

        ref1 = self._ref_pairs(feat1, B, N, ref_idx)
        ref2 = self._ref_pairs(feat2, B, N, ref_idx)
        ref3 = self._ref_pairs(feat3, B, N, ref_idx)

        # lv3 flow
        corr3 = self.cost_vol(ref3, feat3, r=self.r) # estimate pixel displacement coarsely, r = 2 -> 32 real pixels
        flow3 = self.flow_head_lv3(corr3)

        # lv2 flow
        flow2_up = self.up_flow(flow3, feat2.shape[-2:])
        warp2 = self.warp(feat2, flow2_up)
        flow2 = flow2_up + self.flow_conv_lv2(torch.cat((warp2, ref2), dim=1))

        # lv1 flow
        flow1_up = self.up_flow(flow2, feat1.shape[-2:])
        warp1 = self.warp(feat1, flow1_up)
        flow1 = flow1_up + self.flow_conv_lv1(torch.cat((warp1, ref1), dim=1))

        # lv1 cascade
        warp1b = self.warp(feat1, flow1)
        flow1b = flow1 + self.flow_conv_casc(torch.cat((warp1b, ref1), dim=1))

        # Sizes come from the tensors, not from H // 2 ** k: the stride-2
        # extractors round up on odd inputs, and a view that disagrees by one
        # pixel would throw rather than degrade.
        flows = {lvl: f.view(B, N, 2, *f.shape[-2:])
                 for lvl, f in (('lv3', flow3), ('lv2', flow2), ('lv1', flow1b))}

        return flows, feat1.view(B, N, self.num_feat, H, W)


class KGTSAlign(nn.Module):
    """KGTSMamba's burst front-end: flow + per-frame token features on the packed grid.

    One module so the whole alignment cost shows up under one name
    (`model.align`) in parameter and FLOP counts.

    type='bayer': PreAlign (packed -> Bayer grid, flow_in_chans wide) ->
        FlowAlign on the 2x grid -> PostAlign folds features and flow back to
        the packed grid. Flow is estimated at full Bayer resolution; every
        FlowAlign conv runs on 4x the pixels.
    type='packed': FlowAlign directly on the packed RGGB frames (its
        in_channels are the raw channels), ~4x cheaper per FlowAlign conv.
        Token features come from a small encoder over [raw frame; FlowAlign
        lv1 features]: flow_feat can then be sized for matching alone (it
        only has to find correspondences), while the raw samples -- the
        sub-pixel content KGTS exists to deliver -- reach the tokens directly.

    Both return feats (B, N, token_feat, h, w) and flow (B, N, 2, h, w) on the
    packed grid in packed-pixel units, plus the FlowAlign pyramid dict (in
    that type's own grid units: Bayer px for 'bayer', packed px for 'packed').

    Args:
        in_chans: raw channels of the packed burst (4 for RGGB).
        type: 'bayer' | 'packed'.
        flow_feat: FlowAlign num_feat (pyramid + flow-head width).
        flow_in_chans: FlowAlign in_channels. 'bayer': PreAlign's output
            width, defaults to flow_feat. 'packed': must equal in_chans (the
            raw frames go straight in); defaults to it.
        token_feat: width of the per-frame features handed to TokenBank (its c).
        r: lv3 cost-volume radius, in lv3 pixels; costs (2r+1)^2 channels.
            lv3 is 4x coarser than the grid FlowAlign runs on, so one lv3 px
            is 2 packed px (16 GT px) in 'bayer' mode and 4 packed px (32 GT
            px) in 'packed' mode. r=2 covers the protocol's 24 GT px max
            translation in 'bayer'; r=1 already covers it in 'packed'.
        num_frames: kept for config compatibility; builds nothing.
        ref_idx: reference-frame index (None -> N // 2). Also the keyframe.
        global_motion: None | 'affine' | 'lk'. 'affine' replaces the packed flow handed
            to the tokens by a robust per-frame affine fit of it
            (affine_flow_fit); the returned pyramid stays dense, so the flow
            loss still supervises every pixel. 'lk' goes further: a robust
            photometric Gauss-Newton registration of each frame to the keyframe
            on the burst itself (affine_lk), started from zero motion and from
            that fit, detached -- ~4x less tap-position error than the fit
            (0.076 vs ~0.33 packed px on DBSR bursts), right from step 0.
            Being detached, it leaves FlowAlign's flow heads to train.flow_opt /
            train.photo_opt alone: keep one of them active on every step (M1 holds
            flow_lambda at a floor), else DDP with find_unused_parameters: false
            fails. KGTS reads each tap's sub-pixel
            position off this flow and nothing downstream can correct it: at
            x8, 0.1 packed px of flow error is 0.8 HR px of misplaced sample.
            A dense per-pixel estimate from 5x5-receptive-field convs on noisy
            RAW cannot average its noise away; one fit per frame pools
            thousands of pixels into 6 numbers. Measured on the DBSR generator
            (packed FlowAlign, flow_feat 32, flow loss only, 2.5k CPU iters):
            interior lv1 EPE 0.51 -> 0.35 packed px. About half of that
            error was per-frame global bias, which a fit cannot remove and
            further training has to; the local half it removes outright.
            Exact for SyntheticBurst (one affine map per frame; the fit of
            its flow_vectors is off by 1e-5 px). For real bursts with local
            motion (RealBSR) combine it with `local`, or leave it None.
        gm_margin: packed px at each border excluded from the fit (content that
            leaves the view has no correspondence; SyntheticBurst's largest
            shift is ~3.6 packed px).
        lk: dict of affine_lk options (sigmas, iters, margin) for global_motion 'lk'.
        local: None | dict(tau, win): hybrid flow (local_residual) -- the global
            model plus the dense estimate's residual where it is coherently
            different (moving objects, parallax). Needs global_motion. What makes
            a global motion model safe on real bursts; ~closed on SyntheticBurst.
        token_blocks: residual TokenBlocks (depthwise 7x7 + pointwise MLP, zero-init)
            after the token encoder, per frame. The encoder is otherwise 5 convs
            deep (RF 11 packed px, ~78k params) against the keyframe's 24-layer
            body: whatever the body learns, every other frame reaches it only
            through these features. Each block adds +6 px of RF and ~0.045
            GMACs per 48x48 frame at token_feat 64.
    """
    def __init__(self, in_chans=4, type='bayer', flow_feat=64, flow_in_chans=None, token_feat=64,
                 r=2, num_frames=14, ref_idx=None, global_motion=None, gm_margin=4, lk=None, local=None,
                 token_blocks=0):
        super().__init__()
        if type not in ('bayer', 'packed'):
            raise ValueError(f"align.type must be 'bayer' or 'packed', got {type!r}")
        if global_motion not in (None, 'affine', 'lk'):
            raise ValueError(f"align.global_motion must be None, 'affine' or 'lk', got {global_motion!r}")
        if local is not None and global_motion is None:
            raise ValueError('align.local refines a global motion model: set align.global_motion too')
        self.type = type
        self.ref_idx = ref_idx
        self.token_feat = token_feat
        self.global_motion, self.gm_margin = global_motion, gm_margin
        self.lk, self.local = dict(lk or {}), (None if local is None else dict(local))

        if type == 'bayer':
            flow_in_chans = flow_in_chans or flow_feat
            self.prealign = PreAlign(in_chans, flow_in_chans)
            self.flowalign = FlowAlign(flow_in_chans, flow_feat, num_frames=num_frames, r=r, ref_idx=ref_idx)
            self.postalign = PostAlign(flow_feat, token_feat)
        else:
            flow_in_chans = flow_in_chans or in_chans
            if flow_in_chans != in_chans:
                raise ValueError(f"align.type='packed' feeds the raw frames to FlowAlign, so flow_in_chans "
                                 f"({flow_in_chans}) must equal in_chans ({in_chans})")
            self.flowalign = FlowAlign(in_chans, flow_feat, num_frames=num_frames, r=r, ref_idx=ref_idx)
            self.token_enc = nn.Sequential(
                nn.Conv2d(in_chans + flow_feat, token_feat, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(token_feat, token_feat, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
            )
        self.token_blocks = nn.Sequential(*[TokenBlock(token_feat) for _ in range(token_blocks)]) \
            if token_blocks else None

    def forward(self, burst, ref_idx, size=None):
        """burst: (B, N, in_chans, h, w) packed, normalised. size: (h, w) of the unpadded
        image at the top-left, for the global-motion fit (default: all of it).
        Returns feats (B, N, token_feat, h, w), flow (B, N, 2, h, w) fp32 packed px, flows dict."""
        if self.type == 'bayer':
            flows, feats = self.flowalign(self.prealign(burst), ref_idx=ref_idx)
            feats, flow = self.postalign(feats, flows['lv1'])
        else:
            B, N, C, h, w = burst.shape
            flows, feats = self.flowalign(burst, ref_idx=ref_idx)
            x = torch.cat([burst, feats], 2).reshape(B * N, -1, h, w)
            feats = self.token_enc(x).view(B, N, self.token_feat, h, w)
            flow = flows['lv1'].float()
        if self.token_blocks is not None:
            feats = self.token_blocks(feats.flatten(0, 1)).view_as(feats)
        dense = flow
        if self.global_motion == 'affine':
            flow = affine_flow_fit(flow.flatten(0, 1), self.gm_margin, size=size).view_as(flow)
        elif self.global_motion == 'lk':
            fit = affine_flow_fit(flow.flatten(0, 1), self.gm_margin, size=size).view_as(flow)
            flow = affine_lk(burst, ref_idx, init=fit, size=size, **self.lk)
        if self.local is not None:
            flow = local_residual(dense, flow, **self.local)
        return feats, flow, flows