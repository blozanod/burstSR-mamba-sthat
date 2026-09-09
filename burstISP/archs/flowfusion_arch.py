import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from burstISP.utils.registry import ARCH_REGISTRY

class PreFlow(nn.Module):
    """ Projects input RGGB frames into feature space and
    upsamples to Bayer grid resolution via PixelShuffle.

    Allows alignment and downstream modules to operate on pixel grid instead 
    of packed RGGB subpixel space.
    """
    def __init__(self, in_channels=4, num_feat=64):
        super(PreFlow, self).__init__()
        # Expand to 4*num_feat only in the last conv, immediately before the
        # shuffle. Running both convs at 4*num_feat costs 0.60M for a shallow
        # input projection; this is 0.15M for the same output shape.
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

class BurstFlow(nn.Module):
    """ BurstFlow module to predict feature flow across burst

    Pyramidal structure to calculate coarse -> fine flow refinement
    
    Args:
    num_feat (int): Channel number of middle features. Default: 64
    num_frames (int): Number of frames in the burst. Default: 5
    offset_groups (int): Number of groups for DCN offset prediction. Default: 4
    """

    def __init__(self, in_channels=4, num_feat=64, num_frames=5, offset_groups=4, r=2):
        super(BurstFlow, self).__init__()
        # Index Buffers: DCNv4 works as [x1, y1, xn, yn, m1, mn]
        kernel_size = 3
        G, Kg = offset_groups, kernel_size * kernel_size
        grp = torch.arange(G) * (Kg * 3)
        pt = torch.arange(Kg) * 2

        idx_dx = (grp[:, None] + pt[None, :]).flatten()
        idx_dy = (idx_dx + 1)
        idx_m = (grp[:, None] + Kg * 2 + torch.arange(Kg)).flatten()

        all_idx = torch.cat([idx_dx, idx_dy, idx_m])
        assert torch.equal(
                all_idx.sort().values, torch.arange(G * Kg * 3)
            ), "Index union != torch.arange(108)"

        padded_offsets = int(math.ceil((G * Kg * 3) / 8) * 8)

        self.register_buffer("idx_dx", idx_dx, persistent=False)
        self.register_buffer("idx_dy", idx_dy, persistent=False)
        self.register_buffer("idx_m", idx_m, persistent=False)

        #  Inits
        self.num_frames = num_frames
        self.num_feat = num_feat
        self.r = r
        self.center_frame_idx = num_frames // 2

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

        # flow heads + offset convs
        self.flow_head_lv3 = nn.Conv2d((2*r+1)**2, 2, kernel_size=kernel_size, padding=1) # cost vol outputs (2r+1)^2 chans
        self.offset_conv_lv2 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, 2, kernel_size=kernel_size, padding=1)
            )
        self.offset_conv_lv1 = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, 2, kernel_size=kernel_size, padding=1)
            )
        self.offset_conv_casc = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, 2, kernel_size=kernel_size, padding=1)
            )

        self._init_offset_weights()

    def _init_offset_weights(self):
        """Zero-initialise all DCN projection layers.

        At t=0 all offsets and masks are zero DCNv4 starts as a regular
        convolution, giving stable early-training gradients.
        """
        for proj in [self.offset_conv_lv1, self.offset_conv_lv2, self.offset_conv_casc, self.offset_proj]:
            layer = proj[-1] if isinstance(proj, nn.Sequential) else proj
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)

    def warp(self, x, flow, align_corners=False):
        """
        x: [B, C, H, W]
        flow: [B, 2, H, W] where flow[:,0] = x_shift, flow[:,1] = y_shift
        align_corners: set to False, determines formula used
        return: [B, C, H, W]
        """
        B, C, H, W = x.size()

        # base grid
        ys, xs = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32), 
                                torch.arange(W, device=x.device, dtype=torch.float32), indexing='ij')

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
        return F.grid_sample(x, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=align_corners)

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

    def up_flow(self, f):
        """f: [B, 2, h, w] in level-h pixel units -> [B, 2, 2h, 2w] in fine pixels"""
        return F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False) * 2.0

    def scatter_flow(self, om, flow):
        """
        om: [B, 112, H, w] offset proj
        flow: [B, 2, H, W] accumulated flow (sample = p - flow)
        out: [B, 112, H, W] ready for DCNv4 block
        """
        # houses dx dx for all respective channels
        delta = torch.zeros_like(om)
        delta[:, self.idx_dx] = -flow[:, 0:1]
        delta[:, self.idx_dy] = -flow[:, 1:]

        # apply flow to om
        out = om + delta
        return out

    def forward(self, x):
        """
        Args:
            x (Tensor): Burst of shape (B, N, C, H, W).

        Returns:
            aligned  (Tensor): aligned features, (B, N, num_feat, H, W).
            ref1_b   (Tensor): reference-frame lv1 features, (B, num_feat, H, W).
            flows    (dict):   the estimated flow at each pyramid level, each
                (B, N, 2, h, w) in that level's own pixel units and in the
                same sign convention as the generator's ``flow_vectors`` --
                the content the reference sees at p sits at ``p - flow(p)``.
                Exposed for the flow-supervision loss and for offset_analysis;
                the negation into DCN offset units happens in ``scatter_flow``.
        """
        B,N,C,H,W = x.size()
        x_reshaped = x.view(B * N, C, H, W)

        # feat extraction
        feat1 = self.feat_extractor_lv1(x_reshaped) # B C H W
        feat2 = self.feat_extractor_lv2(feat1) # B C H/2 W/2
        feat3 = self.feat_extractor_lv3(feat2) # B C H/4 W/4

        ref1_b = feat1.view(B, N, -1, H, W)[:, self.center_frame_idx]
        ref1 = ref1_b.repeat_interleave(N, dim=0)
        ref2 = feat2.view(B, N, -1, H // 2, W // 2)[:, self.center_frame_idx].repeat_interleave(N, dim=0)
        ref3 = feat3.view(B, N, -1, H // 4, W // 4)[:, self.center_frame_idx].repeat_interleave(N, dim=0)

        # lv3 alignment
        corr3 = self.cost_vol(ref3, feat3, r=self.r) # estimate pixel displacement coarsely, r = 2 -> 32 real pixels
        flow3 = self.flow_head_lv3(corr3)

        # lv2 alignment
        flow2_up = self.up_flow(flow3)
        warp2 = self.warp(feat2, flow2_up)
        flow2 = flow2_up + self.offset_conv_lv2(torch.cat((warp2, ref2), dim=1))

        # lv1 alignment
        flow1_up = self.up_flow(flow2)
        warp1 = self.warp(feat1, flow1_up)
        flow1 = flow1_up + self.offset_conv_lv1(torch.cat((warp1, ref1), dim=1))

        # lv1 cascade
        warp1b = self.warp(feat1, flow1)
        flow1b = flow1 + self.offset_conv_casc(torch.cat((warp1b, ref1), dim=1))

        # Sizes come from the tensors, not from H // 2 ** k: the stride-2
        # extractors round up on odd inputs, and a view that disagrees by one
        # pixel would throw rather than degrade.
        flows = {lvl: f.view(B, N, 2, *f.shape[-2:])
                 for lvl, f in (('lv3', flow3), ('lv2', flow2), ('lv1', flow1b))}

        return flows

def gather(self, x, flow, align_corners=False):
        """
        x: [B, C, H, W]
        flow: [B, 2, H, W] where flow[:,0] = x_shift, flow[:,1] = y_shift
        align_corners: set to False, determines formula used
        return: [B, C, H, W]
        """
        B, C, H, W = x.size()

        # base grid
        ys, xs = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32), 
                                torch.arange(W, device=x.device, dtype=torch.float32), indexing='ij')

        sample_x = xs - flow[:,0]
        sample_y = ys - flow[:,1]

        # selects correct formula based on align_corners
        if align_corners:
            gx = 2 * sample_x / (W - 1) - 1
            gy = 2 * sample_y / (H - 1) - 1
        else:
            gx = (2 * sample_x + 1) / W - 1
            gy = (2 * sample_y + 1) / H - 1

        ones = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        grid = torch.stack((gx, gy), -1)
        gathered = F.grid_sample(x,    grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
        conf     = F.grid_sample(ones, grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
        return gathered, conf