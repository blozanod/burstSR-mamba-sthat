import torch
from .mambairv2_arch import MambaIRv2, UpsampleOneStep
from .kgts_arch import KGTS, TokenBank
from .flow_align_arch import PreAlign, FlowAlign, PostAlign
from einops import rearrange

class KGTSMamba(MambaIRv2):
    """Burst SR: MambaIRv2 body on the keyframe, one tied KGTS after every ASSB.
 
    In:  (B, N, 4, H, W) packed RGGB burst.
    Out: (B, out_chans, H*upscale, W*upscale). upscale is relative to the PACKED grid.
    All MambaIRv2 args (embed_dim, depths, upscale, window_size, ...) pass through **kwargs.
    """
 
    def __init__(self, out_chans=3, align_feat=64, d=16, n=8, k=2, ref_idx=None, flow_r=2, **kwargs):
        kwargs.setdefault('in_chans', 4)
        kwargs.setdefault('upsampler', 'pixelshuffledirect')
        super().__init__(**kwargs)          # builds conv_first, ASSBs, norm, conv_after_body; runs _init_weights
        assert self.upsampler == 'pixelshuffledirect', 'only the lightweight head is wired'
        self.ref_idx = ref_idx
 
        # upstream ties num_out_ch = in_chans (4); RAW -> RGB needs its own head
        self.upsample = UpsampleOneStep(self.upscale, self.embed_dim, out_chans, tuple(self.patches_resolution))
 
        # Burst branch. Built AFTER super().__init__ on purpose: MambaIRv2 calls
        # self.apply(_init_weights), which would overwrite KGTS's zero-init W_c and
        # identity FiLM. Never call self.apply(...) on this model again.
        in_chans = kwargs['in_chans']
        self.prealign = PreAlign(in_chans, align_feat)                                # packed -> Bayer grid
        self.flowalign = FlowAlign(align_feat, align_feat, r=flow_r, ref_idx=ref_idx)
        self.postalign = PostAlign(align_feat)                                        # Bayer -> packed grid
        self.bank = TokenBank(align_feat, d, k)
        self.kgts = KGTS(self.embed_dim, d, n)
 
    def forward_features(self, x, params, cache):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)                                   # (B, H*W, C), row-major == (b h w)
        B = x.shape[0]
        for layer in self.layers:
            x = layer(x, x_size, params)
            s = self.kgts(rearrange(x, 'b l c -> (b l) c'), cache)
            x = rearrange(s, '(b l) c -> b l c', b=B)
        x = self.norm(x)
        return self.patch_unembed(x, x_size)
 
    def forward(self, burst):
        B, N, C, h_ori, w_ori = burst.shape
        ref = (N // 2 if self.ref_idx is None else self.ref_idx) % N   # same default as FlowAlign
 
        # pad the WHOLE burst to a multiple of window_size, so tokens and MambaIRv2 share one grid
        mod = self.window_size
        h, w = -(-h_ori // mod) * mod, -(-w_ori // mod) * mod
        burst = torch.cat([burst, burst.flip(-2)], -2)[..., :h, :]
        burst = torch.cat([burst, burst.flip(-1)], -1)[..., :w]
        burst = (burst - self.mean.type_as(burst).unsqueeze(1)) * self.img_range
 
        # burst branch: flow on the Bayer grid, tokens on the packed grid
        flows, feats = self.flowalign(self.prealign(burst), ref_idx=ref)
        feats, flow = self.postalign(feats, flows['lv1'])            # (B, N, align_feat, h, w), packed units
        tok, valid = self.bank(feats, flow)                          # (B*h*w, N*k*k, d)
        cache = self.kgts.precompute(tok, valid)
 
        # keyframe body
        params = {'attn_mask': self.calculate_mask([h, w]).to(burst.device),
                  'rpi_sa': self.relative_position_index_SA}
        x = self.conv_first(burst[:, ref])
        x = self.conv_after_body(self.forward_features(x, params, cache)) + x
        x = self.upsample(x)
 
        x = x / self.img_range + self.mean.type_as(x)
        return x[..., :h_ori * self.upscale, :w_ori * self.upscale]
 
 
if __name__ == '__main__':
    model = KGTSMamba(upscale=4, embed_dim=48, d_state=8, depths=[5, 5, 5, 5], num_heads=[4, 4, 4, 4],
                      window_size=16, inner_rank=32, num_tokens=64, mlp_ratio=1., img_size=48).cuda()
    print(f'params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    for name, m in (('kgts', model.kgts), ('bank', model.bank), ('flowalign', model.flowalign),
                    ('prealign', model.prealign), ('postalign', model.postalign)):
        print(f'  {name}: {sum(p.numel() for p in m.parameters()) / 1e6:.3f}M')
    assert model.kgts.W_c.weight.abs().sum() == 0, 'W_c zero-init was overwritten'
    out = model(torch.randn(1, 14, 4, 40, 44).cuda())   # non-multiple of 16 on purpose: tests padding
    print('out', tuple(out.shape))                      # expect (1, 3, 160, 176)