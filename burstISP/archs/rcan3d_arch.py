import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from burstISP.utils.registry import ARCH_REGISTRY


class ChannelAttention3D(nn.Module):
    """RCAN's squeeze-and-excitation channel attention, pooled over (T, H, W).

    One scalar per channel, shared by every frame and every pixel: it can
    re-weight *what* a feature means, not *where* or *in which frame*.
    """

    def __init__(self, num_feat, reduction):
        super(ChannelAttention3D, self).__init__()
        squeeze = max(num_feat // reduction, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, squeeze, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(squeeze, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class RCAB3D(nn.Module):
    """Residual channel attention block with 3D convolutions:
    conv3d -> ReLU -> conv3d -> channel attention, plus identity."""

    def __init__(self, num_feat, reduction, kernel_size, res_scale=1.0):
        super(RCAB3D, self).__init__()
        pad = tuple(k // 2 for k in kernel_size)
        self.body = nn.Sequential(
            nn.Conv3d(num_feat, num_feat, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat, num_feat, kernel_size, padding=pad),
            ChannelAttention3D(num_feat, reduction),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class ResidualGroup3D(nn.Module):
    """num_block RCAB3Ds and a closing conv3d, wrapped in a short skip."""

    def __init__(self, num_feat, num_block, reduction, kernel_size, res_scale=1.0):
        super(ResidualGroup3D, self).__init__()
        pad = tuple(k // 2 for k in kernel_size)
        self.body = nn.Sequential(
            *[RCAB3D(num_feat, reduction, kernel_size, res_scale) for _ in range(num_block)],
            nn.Conv3d(num_feat, num_feat, kernel_size, padding=pad),
        )

    def forward(self, x):
        return x + self.body(x)


class FrozenRAFT(nn.Module):
    """Pretrained torchvision RAFT, frozen, estimating reference -> frame flow
    on the packed-RGGB grid.

    RAFT was trained on sRGB video, and it works at 1/8 resolution, so a
    48x48 packed frame (6x6 at 1/8) is far too small to feed it directly.
    Each frame is therefore turned into a pseudo-sRGB image (R, mean(G1, G2),
    B, gamma 1/2.2, scaled to RAFT's [-1, 1]), bilinearly upsampled by
    `upscale` for estimation, and the flow is area-pooled back to the packed
    grid and rescaled into packed-pixel units. At the default x4, SyntheticBurst's
    worst-case 24 GT px shift (3 packed px) is 12 px at RAFT's input.

    Runs in fp32 under no_grad regardless of the caller's autocast: it is a
    fixed preprocessing step, not part of the trained network.
    """

    def __init__(self, variant='raft_large', weights=None, iters=12, upscale=4, chunk=64):
        super(FrozenRAFT, self).__init__()
        from torchvision.models.optical_flow import (Raft_Large_Weights, Raft_Small_Weights, raft_large,
                                                     raft_small)
        builders = {'raft_large': (raft_large, Raft_Large_Weights.DEFAULT),
                    'raft_small': (raft_small, Raft_Small_Weights.DEFAULT)}
        if variant not in builders:
            raise ValueError(f'RCAN3D align must be one of {sorted(builders)} or none, got {variant!r}')
        build, default_weights = builders[variant]

        if weights is None:
            # Downloads to $TORCH_HOME/hub/checkpoints on first use. Compute
            # nodes may have no network: prefetch on a login node first (see
            # main/configs/R3D_RCAN3D_RAFT.yml), or pass a local path instead.
            self.net = build(weights=default_weights, progress=False)
        else:
            self.net = build(weights=None, progress=False)
            self.net.load_state_dict(torch.load(weights, map_location='cpu'))
        self.requires_grad_(False)
        self.eval()

        self.iters = iters
        self.upscale = upscale
        self.chunk = chunk

    def train(self, mode=True):
        # Always eval: RAFT's context encoder has BatchNorm.
        return super(FrozenRAFT, self).train(False)

    @staticmethod
    def pseudo_srgb(x):
        """[..., 4 (RGGB), h, w] linear RAW -> [..., 3, h, w] in [-1, 1]."""
        rgb = torch.stack([x[..., 0, :, :], 0.5 * (x[..., 1, :, :] + x[..., 2, :, :]), x[..., 3, :, :]], dim=-3)
        return rgb.clamp(0, 1).pow(1 / 2.2) * 2 - 1

    @torch.no_grad()
    def forward(self, x, ref):
        """
        Args:
            x (Tensor): packed-RGGB burst, [B, N, 4, h, w].
            ref (int): reference frame index.

        Returns:
            Tensor: [B, N, 2, h, w] flow (dx, dy) in packed pixels, such that
            frame i sampled at p + flow_i(p) lines up with the reference at p.
            The reference's own flow is exactly zero.
        """
        B, N, _, h, w = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            # RAFT needs H, W divisible by 8.
            H = int(math.ceil(h * self.upscale / 8)) * 8
            W = int(math.ceil(w * self.upscale / 8)) * 8
            img = F.interpolate(self.pseudo_srgb(x).reshape(B * N, 3, h, w), size=(H, W),
                                mode='bilinear', align_corners=False).reshape(B, N, 3, H, W)

            others = [i for i in range(N) if i != ref]
            img1 = img[:, ref:ref + 1].expand(-1, N - 1, -1, -1, -1).reshape(-1, 3, H, W)
            img2 = img[:, others].reshape(-1, 3, H, W)

            flows = []
            for s in range(0, img1.shape[0], self.chunk):
                flows.append(self.net(img1[s:s + self.chunk], img2[s:s + self.chunk],
                                      num_flow_updates=self.iters)[-1])
            flow = F.adaptive_avg_pool2d(torch.cat(flows, 0), (h, w))
            flow = flow * torch.tensor([w / W, h / H], device=flow.device).view(1, 2, 1, 1)

            out = x.new_zeros(B, N, 2, h, w)
            out[:, others] = flow.reshape(B, N - 1, 2, h, w)
        return out


def flow_warp(feat, flow):
    """Backward-warp feat [M, C, h, w] by flow [M, 2, h, w]: out(p) = feat(p + flow(p)).

    Always resamples in fp32 (grid_sample needs input and grid to share a
    dtype, and the flow is fp32 while the features may be bf16).
    """
    M, _, h, w = feat.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=feat.device, dtype=torch.float32),
                            torch.arange(w, device=feat.device, dtype=torch.float32), indexing='ij')
    gx = 2 * (xs + flow[:, 0].float()) / max(w - 1, 1) - 1
    gy = 2 * (ys + flow[:, 1].float()) / max(h - 1, 1) - 1
    return F.grid_sample(feat.float(), torch.stack([gx, gy], dim=-1), mode='bilinear',
                         padding_mode='border', align_corners=True)


@ARCH_REGISTRY.register()
class RCAN3D(nn.Module):
    """RCAN with 3D convolutions, for RAW burst super-resolution.

    The burst is treated as a (T, H, W) volume: RCAN's residual-in-residual
    trunk runs unchanged except that every trunk conv is a 3D conv over
    (frame, y, x). The trunk follows RCAN (Zhang et al., ECCV 2018); the 3D
    form of it is the one used for volumetric microscopy by 3D-RCAN (Chen et
    al., Nature Methods 2021). Burst/video SR with 3D convs and no explicit
    motion compensation is DUF (Jo et al., CVPR 2018) and FSTRN (Li et al.,
    CVPR 2019) -- that is what `align: none` gives.

    With `align: raft_large` (default) every frame's features are
    backward-warped onto the reference by a frozen, pretrained RAFT before
    the trunk, so the 3D kernels see a registered stack and the time axis
    means "same scene point, different sub-pixel sample". RAFT is kept out
    of the module tree on purpose: it is not in parameters() (so neither the
    optimizer, DDP, nor the MAC/param counts see it) and not in state_dict()
    (so checkpoints hold only the trained network and load strictly with or
    without it).

    Pipeline, on a packed-RGGB burst [B, N, 4, h, w]:
        head      per-frame conv2d 4 -> C              [B, N, C, h, w]
        align     warp each frame's features onto the reference with RAFT
                  flow (identity for the reference and for align: none)
        body      num_group x ResidualGroup3D, conv3d, + head (long skip),
                  on [B, C, N, h, w]
        fuse      conv3d with kernel (N, 1, 1): a learned per-channel
                  weighting of all N frames, collapsing the time axis
                                                       [B, C, h, w]
        upsample  RCAN's upsampler, (conv2d C -> 4C, PixelShuffle 2) x log2(scale)
        tail      conv2d C -> 3                        [B, 3, h*scale, w*scale]

    The cost is dominated by the body's 3D convs, 27 * C^2 * N * h * w MACs
    each: 2.007 GMACs at C=48, N=14, 48x48. See the config header
    (main/configs/R3D_RCAN3D_RAFT.yml) for the full budget, or measure it
    with `python analysis/count_macs.py <config>`.

    Frame order is the dataset's: the reference sits at index N // 2
    (SyntheticBurstDataset._center_ref_order), at the centre of every
    temporal kernel's footprint. In SyntheticBurst the other frames are
    i.i.d. random shifts/rotations, so which frames are temporal neighbours
    is arbitrary -- temporal locality is an inductive bias this model tests.

    Args:
        num_frames (int): burst length N. Fixed at build time: the fuse
            conv's time kernel spans exactly N frames.
        in_chans (int): input channels per frame (4 for packed RGGB).
        out_chans (int): output channels (3 for linear RGB).
        num_feat (int): trunk width C.
        num_group (int): number of residual groups.
        num_block (int): RCAB3Ds per residual group.
        reduction (int): channel-attention squeeze ratio.
        temporal_kernel (int): time extent of every trunk kernel (odd).
        spatial_kernel (int): spatial extent of every trunk kernel (odd).
        res_scale (float): RCAB residual scaling (EDSR-style; 1.0 = off).
        scale (int): packed-LR -> GT ratio; must be a power of 2.
        align (str): 'raft_large', 'raft_small', or 'none'.
        flow_weights (str): optional local RAFT state_dict path, instead of
            torchvision's download.
        flow_iters (int): RAFT refinement iterations.
        flow_upscale (int): bilinear upsampling of the pseudo-sRGB frames
            before RAFT (RAFT works at 1/8 resolution).
        img_size, is_train: accepted and ignored, so the shared tooling
            (analysis/shape_check.py, analysis/burst_ablation.py) can pass the
            keys it passes to MambaFusionNet.
    """

    def __init__(self, num_frames=14, in_chans=4, out_chans=3, num_feat=48, num_group=4, num_block=4,
                 reduction=8, temporal_kernel=3, spatial_kernel=3, res_scale=1.0, scale=8,
                 align='raft_large', flow_weights=None, flow_iters=12, flow_upscale=4,
                 img_size=None, is_train=True):
        super(RCAN3D, self).__init__()
        if temporal_kernel % 2 == 0 or spatial_kernel % 2 == 0:
            raise ValueError(f'RCAN3D kernels must be odd, got temporal_kernel={temporal_kernel}, '
                             f'spatial_kernel={spatial_kernel}')
        if scale < 2 or scale & (scale - 1):
            raise ValueError(f'RCAN3D upsamples by repeated x2 PixelShuffle; scale must be a power '
                             f'of 2, got {scale}')

        self.num_frames = num_frames
        self.ref = num_frames // 2
        self.scale = scale
        kernel = (temporal_kernel, spatial_kernel, spatial_kernel)
        pad = (temporal_kernel // 2, spatial_kernel // 2, spatial_kernel // 2)

        # Bypass nn.Module.__setattr__ so RAFT is a plain attribute, not a
        # submodule: see the class docstring. Moved to the input's device
        # lazily in forward(), since .to()/.cuda() on this module skip it.
        align = None if align in (None, 'none', False) else align
        object.__setattr__(self, 'flow_net', None if align is None else FrozenRAFT(
            align, weights=flow_weights, iters=flow_iters, upscale=flow_upscale))

        # Per-frame 2D head, so that features are warped rather than raw RGGB:
        # the head can encode local structure into channels before the one
        # bilinear resampling, and the warp never mixes frames.
        self.conv_first = nn.Conv2d(in_chans, num_feat, spatial_kernel, padding=spatial_kernel // 2)

        self.body = nn.Sequential(
            *[ResidualGroup3D(num_feat, num_block, reduction, kernel, res_scale) for _ in range(num_group)])
        self.conv_after_body = nn.Conv3d(num_feat, num_feat, kernel, padding=pad)

        # Collapse time: one weight per (frame, in-channel, out-channel). Each
        # output channel is a learned linear combination of every frame.
        self.temporal_fuse = nn.Conv3d(num_feat, num_feat, (num_frames, 1, 1), padding=0)

        upsample = []
        for _ in range(scale.bit_length() - 1):
            upsample += [nn.Conv2d(num_feat, 4 * num_feat, 3, padding=1), nn.PixelShuffle(2)]
        self.upsample = nn.Sequential(*upsample)
        self.conv_last = nn.Conv2d(num_feat, out_chans, 3, padding=1)

    def forward(self, x):
        """
        Args:
            x (Tensor): packed-RGGB burst, [B, N, C_in, h, w].

        Returns:
            Tensor: [B, out_chans, h * scale, w * scale].
        """
        B, N, C, h, w = x.shape
        if N != self.num_frames:
            raise ValueError(f'RCAN3D was built for num_frames={self.num_frames}, got a burst of {N}')

        feat = self.conv_first(x.reshape(B * N, C, h, w))            # [B*N, F, h, w]

        if self.flow_net is not None:
            if next(self.flow_net.parameters()).device != x.device:
                self.flow_net.to(x.device)
            flow = self.flow_net(x, self.ref)                          # [B, N, 2, h, w]
            feat = flow_warp(feat, flow.reshape(B * N, 2, h, w)).to(feat.dtype)

        feat = feat.reshape(B, N, -1, h, w).permute(0, 2, 1, 3, 4)    # [B, F, N, h, w]
        feat = feat + self.conv_after_body(self.body(feat))

        feat = self.temporal_fuse(feat).squeeze(2)                     # [B, F, h, w]
        return self.conv_last(self.upsample(feat))
