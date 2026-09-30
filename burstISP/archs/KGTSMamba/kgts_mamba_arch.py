import torch
from torch.utils.checkpoint import checkpoint
from burstISP.utils.registry import ARCH_REGISTRY
from .mambairv2_arch import MambaIRv2
from .kgts_arch import KGTS, TokenBank
from .flow_align_arch import KGTSAlign
from einops import rearrange


@ARCH_REGISTRY.register()
class KGTSMamba(MambaIRv2):
    """Burst SR: MambaIRv2 body on the keyframe, one tied KGTS after every ASSB.

    In:  (B, N, in_chans, H, W) packed RGGB burst.
    Out: (B, out_chans, H*upscale, W*upscale). upscale is relative to the PACKED grid
         (8 for SyntheticBurst), whatever align.type is: the body always runs packed.

    Every MambaIRv2 arg (img_size, in_chans, embed_dim, d_state, depths, num_heads,
    window_size, inner_rank, num_tokens, convffn_kernel_size, mlp_ratio, upscale,
    img_range, upsampler, upsample_feat, resi_connection, use_checkpoint, ...) passes
    through **kwargs. The burst branch is configured by three dicts:

    align (KGTSAlign):  type ('bayer' | 'packed'), flow_feat, flow_in_chans, r,
                        num_frames, ref_idx
    token (TokenBank):  c (per-frame feature width = align's token_feat), d, k,
                        pos_freqs, norm, mark_ref, pin_ref
    kgts  (KGTS):       n, expand, heads, norm_s, out_gate, out_norm, dt_min, dt_max,
                        a_max, affinity, dt_norm

    KGTS's d is the token d and its ds is embed_dim -- both forced by the wiring.
    They may be written in the kgts dict for readability, but must then agree.
    """

    def __init__(self, out_chans=3, align=None, token=None, kgts=None, **kwargs):
        kwargs.setdefault('in_chans', 4)
        kwargs.setdefault('upsampler', 'pixelshuffledirect')
        super().__init__(out_chans=out_chans, **kwargs)  # builds body + head, runs _init_weights
        if self.upsampler not in ('pixelshuffle', 'pixelshuffledirect'):
            raise ValueError(f"KGTSMamba supports upsampler 'pixelshuffle' or 'pixelshuffledirect', "
                             f"got {self.upsampler!r}")
        self.use_checkpoint = kwargs.get('use_checkpoint', False)

        align = dict(align or {})
        token = dict(token or {})
        kgts = dict(kgts or {})
        token.setdefault('c', 64)
        token.setdefault('d', 16)
        for key, want in (('d', token['d']), ('ds', self.embed_dim)):
            got = kgts.pop(key, want)
            if got != want:
                src = 'token.d' if key == 'd' else 'embed_dim'
                raise ValueError(f'kgts.{key}={got} but it is fixed by {src}={want}')

        # Burst branch. Built AFTER super().__init__ on purpose: MambaIRv2 calls
        # self.apply(_init_weights), which would overwrite KGTS's zero-init W_c and
        # identity FiLM. Never call self.apply(...) on this model again.
        self.align = KGTSAlign(kwargs['in_chans'], token_feat=token['c'], **align)
        self.bank = TokenBank(**token)
        self.kgts = KGTS(self.embed_dim, token['d'], **kgts)
        self.n_inject = len(self.layers)

    def _kgts(self, x, cache):
        B = x.shape[0]
        s = rearrange(x, 'b l c -> (b l) c')
        if self.use_checkpoint and self.training:
            s = checkpoint(self.kgts, s, cache, use_reentrant=False)
        else:
            s = self.kgts(s, cache)
        return rearrange(s, '(b l) c -> b l c', b=B)

    def forward_features(self, x, params, cache):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)                                   # (B, H*W, C), row-major == (b h w)
        if self.ape:
            x = x + self.absolute_pos_embed
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, x_size, params, use_reentrant=False)
            else:
                x = layer(x, x_size, params)
            x = self._kgts(x, cache)
        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def forward(self, burst, return_aux=False):
        B, N, C, h_ori, w_ori = burst.shape
        ref = self.align.ref_idx
        ref = (N // 2 if ref is None else ref) % N                # same default as FlowAlign

        # pad the WHOLE burst to a multiple of window_size, so tokens and MambaIRv2 share one grid
        mod = self.window_size
        h, w = -(-h_ori // mod) * mod, -(-w_ori // mod) * mod
        burst = torch.cat([burst, burst.flip(-2)], -2)[..., :h, :]
        burst = torch.cat([burst, burst.flip(-1)], -1)[..., :w]
        burst = (burst - self.mean.type_as(burst).unsqueeze(1)) * self.img_range

        # burst branch: tokens on the packed grid, flow on the grid align.type picks
        feats, flow, flows = self.align(burst, ref)               # (B, N, c, h, w), packed px
        tok, valid = self.bank(feats, flow, ref)                  # (B*h*w, N*k*k, d)
        cache = self.kgts.precompute(tok, valid)

        # keyframe body
        params = {'attn_mask': self.calculate_mask([h, w]).to(burst.device),
                  'rpi_sa': self.relative_position_index_SA}
        x = self.conv_first(burst[:, ref])
        x = self.conv_after_body(self.forward_features(x, params, cache)) + x
        if self.upsampler == 'pixelshuffle':
            x = self.conv_last(self.upsample(self.conv_before_upsample(x)))
        else:
            x = self.upsample(x)

        x = x / self.img_range + self.mean.type_as(x)
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]
        if not return_aux:
            return x
        # MambaFusionModel's flow curriculum reads aux['flows']; crop off the padding
        ceil = lambda a, b: -(-a // b)
        flows = {lv: f[..., :ceil(f.shape[-2] * h_ori, h), :ceil(f.shape[-1] * w_ori, w)]
                 for lv, f in flows.items()}
        return x, {'flows': flows}


if __name__ == '__main__':
    model = KGTSMamba(upscale=4, embed_dim=48, d_state=8, depths=[5, 5, 5, 5], num_heads=[4, 4, 4, 4],
                      window_size=16, inner_rank=32, num_tokens=64, mlp_ratio=1., img_size=48,
                      kgts=dict(n=16, expand=2, heads=2)).cuda()
    print(f'params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    for name, m in (('kgts', model.kgts), ('bank', model.bank), ('align', model.align)):
        print(f'  {name}: {sum(p.numel() for p in m.parameters()) / 1e6:.3f}M')
    assert model.kgts.W_c.weight.abs().sum() == 0, 'W_c zero-init was overwritten'
    out = model(torch.randn(1, 14, 4, 40, 44).cuda())   # non-multiple of 16 on purpose: tests padding
    print('out', tuple(out.shape))                      # expect (1, 3, 160, 176)
