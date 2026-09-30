import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from burstISP.utils.registry import ARCH_REGISTRY
from .mambairv2_arch import MambaIRv2
from .kgts_arch import KGTS, TokenBank, TokenRefine
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
                        num_frames, ref_idx, global_motion, gm_margin, lk, local,
                        token_blocks
    token (TokenBank):  c (per-frame feature width = align's token_feat), d, k,
                        pos_freqs, norm, mark_ref, pin_ref, tap_pos
    kgts  (KGTS):       n, expand, heads, norm_s, out_gate, out_norm, dt_min, dt_max,
                        a_max, affinity, dt_norm, roles, a_max_sel, depth_embed, untie_out

    KGTS's d is the token d, its ds is embed_dim and its n_calls the number of
    injections -- all forced by the wiring. They may be written in the kgts dict
    for readability, but must then agree.

    inject_first: one more (tied) KGTS call on the embedded keyframe, BEFORE the
        first ASSB. Otherwise the first ASSB -- 1/len(depths) of the body -- works
        on the single noisy keyframe, and the burst first arrives after it.
    refine: None | dict(at=[call indices], hidden). After each listed call, the token
        bank is re-expressed in the current keyframe state (TokenRefine: every tap of
        pixel p is conditioned on the body's state at p) and KGTS's token cache is
        rebuilt from it. The tokens then deepen with the body instead of staying the
        shallow encoder's output for every call. Calls are 0-based and count the
        inject_first call. Costs one TokenRefine + one KGTS.precompute per entry
        (~5 GMACs at M1's width, 48x48).
    aux_head: a linear burst-only reconstruction from the FIRST call's pooled states
        (1x1 conv to out_chans*upscale^2 + pixel shuffle), returned as aux['burst'] by
        forward(return_aux=True) for MambaFusionModel's train.aux_opt loss. W_c is
        zero-init, so without it nothing upstream of W_c (scan, tokens, encoder)
        receives any gradient until W_c has grown; with it the whole burst branch is
        supervised from step 0, and |aux PSNR| is a live readout of what the burst
        branch alone carries. Needs train.aux_opt (its parameters are otherwise unused).
    """

    def __init__(self, out_chans=3, align=None, token=None, kgts=None, inject_first=False, refine=None,
                 aux_head=False, **kwargs):
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
        n_calls = len(self.layers) + int(inject_first)
        for key, want in (('d', token['d']), ('ds', self.embed_dim), ('n_calls', n_calls)):
            got = kgts.pop(key, want)
            if got != want:
                src = {'d': 'token.d', 'ds': 'embed_dim', 'n_calls': 'len(depths) + inject_first'}[key]
                raise ValueError(f'kgts.{key}={got} but it is fixed by {src}={want}')
        refine = dict(refine or {})
        refine_at = sorted(set(refine.pop('at', [])))
        if any(not 0 <= i < n_calls - 1 for i in refine_at):
            raise ValueError(f'refine.at={refine_at}: call indices must be in [0, {n_calls - 2}] '
                             f'(a refinement after the last call would feed nothing)')

        # Burst branch. Built AFTER super().__init__ on purpose: MambaIRv2 calls
        # self.apply(_init_weights), which would overwrite KGTS's zero-init W_c and
        # identity FiLM. Never call self.apply(...) on this model again.
        self.align = KGTSAlign(kwargs['in_chans'], token_feat=token['c'], **align)
        self.bank = TokenBank(**token)
        self.kgts = KGTS(self.embed_dim, token['d'], n_calls=n_calls, **kgts)
        self.n_inject = n_calls
        self.inject_first, self.refine_at = inject_first, refine_at
        # built last, so with every option off the parameter RNG stream is the original one
        self.refiners = nn.ModuleList(TokenRefine(token['d'], self.embed_dim, **refine) for _ in refine_at) \
            if refine_at else None
        self.aux = nn.Sequential(nn.Conv2d(2 * self.kgts.di, out_chans * self.upscale ** 2, 1),
                                 nn.PixelShuffle(self.upscale)) if aux_head else None

    def _kgts(self, x, cache, idx=0, pooled=False):
        B = x.shape[0]
        s = rearrange(x, 'b l c -> (b l) c')
        if self.use_checkpoint and self.training:
            out = checkpoint(self.kgts, s, cache, idx, pooled, use_reentrant=False)
        else:
            out = self.kgts(s, cache, idx, pooled)
        s, y = out if pooled else (out, None)
        return rearrange(s, '(b l) c -> b l c', b=B), y

    def _refine(self, j, tok, valid, extras, x):
        tok = self.refiners[j](tok, valid, rearrange(x, 'b l c -> (b l) c'))
        return tok, self.kgts.precompute(tok, valid, extras)

    def forward_features(self, x, params, bank):
        """bank: TokenBank's output. Returns the body features and, with aux_head, the first
        call's pooled states (else None)."""
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)                                   # (B, H*W, C), row-major == (b h w)
        if self.ape:
            x = x + self.absolute_pos_embed
        tok, valid, extras = (tuple(bank) + (None,))[:3]
        cache = self.kgts.precompute(tok, valid, extras)
        ckpt = self.use_checkpoint and self.training
        want = self.aux is not None                               # the first call's pooled states
        first, call = None, 0
        if self.inject_first:
            x, first = self._kgts(x, cache, call, want)
            call += 1
        for layer in self.layers:
            if ckpt:
                x = checkpoint(layer, x, x_size, params, use_reentrant=False)
            else:
                x = layer(x, x_size, params)
            x, y = self._kgts(x, cache, call, want and first is None)
            first = y if first is None else first
            if call in self.refine_at:
                j = self.refine_at.index(call)
                tok, cache = (checkpoint(self._refine, j, tok, valid, extras, x, use_reentrant=False) if ckpt
                              else self._refine(j, tok, valid, extras, x))
            call += 1
        x = self.norm(x)
        return self.patch_unembed(x, x_size), first

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
        feats, flow, flows = self.align(burst, ref, size=(h_ori, w_ori))   # (B, N, c, h, w), packed px
        # (B*h*w, N*k*k, d) tokens, valid mask, and with roles the per-tap side information
        bank = self.bank(feats, flow, ref, extras=self.kgts.roles is not None)

        # keyframe body
        params = {'attn_mask': self.calculate_mask([h, w]).to(burst.device),
                  'rpi_sa': self.relative_position_index_SA}
        x = self.conv_first(burst[:, ref])
        feat, first = self.forward_features(x, params, bank)
        x = self.conv_after_body(feat) + x
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
        aux = {'flows': flows}
        if self.aux is not None:                                  # burst-only reconstruction
            y = self.aux(rearrange(first, '(b h w) c -> b c h w', b=B, h=h).to(x.dtype))
            aux['burst'] = (y / self.img_range + self.mean.type_as(y))[..., :h_ori * self.upscale, :w_ori * self.upscale]
        return x, aux


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
