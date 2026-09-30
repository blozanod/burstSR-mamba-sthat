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
    ys, xs = torch.meshgrid(torch.arange(h, device=flow.device, dtype=torch.float32),
                            torch.arange(w, device=flow.device, dtype=torch.float32), indexing='ij')
    # centred, unit-range coordinates keep the 3x3 normal equations well conditioned
    X = torch.stack([xs / w - 0.5, ys / h - 0.5, torch.ones_like(xs)], -1).view(1, h * w, 3)
    Y = flow.float().flatten(2).transpose(1, 2)                             # (M, hw, 2)
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
            r = (Y - X @ beta).norm(dim=-1)                                  # (M, hw)
            c = 2 * (r * w0).sum(1, keepdim=True) / w0.sum(1, keepdim=True)
            wt = w0 / (1 + (r / c.clamp_min(1e-3)) ** 2)
    return (X @ beta).transpose(1, 2).reshape(M, 2, h, w).to(flow.dtype)


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
        global_motion: None | 'affine'. 'affine' replaces the packed flow handed
            to the tokens by a robust per-frame affine fit of it
            (affine_flow_fit); the returned pyramid stays dense, so the flow
            loss still supervises every pixel. KGTS reads each tap's sub-pixel
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
            motion (RealBSR) leave it None.
        gm_margin: packed px at each border excluded from the fit (content that
            leaves the view has no correspondence; SyntheticBurst's largest
            shift is ~3.6 packed px).
    """
    def __init__(self, in_chans=4, type='bayer', flow_feat=64, flow_in_chans=None, token_feat=64,
                 r=2, num_frames=14, ref_idx=None, global_motion=None, gm_margin=4):
        super().__init__()
        if type not in ('bayer', 'packed'):
            raise ValueError(f"align.type must be 'bayer' or 'packed', got {type!r}")
        if global_motion not in (None, 'affine'):
            raise ValueError(f"align.global_motion must be None or 'affine', got {global_motion!r}")
        self.type = type
        self.ref_idx = ref_idx
        self.token_feat = token_feat
        self.global_motion, self.gm_margin = global_motion, gm_margin

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
        if self.global_motion == 'affine':
            flow = affine_flow_fit(flow.flatten(0, 1), self.gm_margin, size=size).view_as(flow)
        return feats, flow, flows