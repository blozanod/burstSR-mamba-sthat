"""Budget check: params / GMACs / peak memory for candidate configs. Forward only, fits a 4 GB GPU."""
import re
import sys
import torch
from torch.utils.flop_counter import FlopCounterMode
from burstISP.archs.KGTSMamba.kgts_mamba_arch import KGTSMamba
 
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BURST = (1, 14, 4, 48, 48)      # benchmark input -- match what QMambaBSR / BurstMamba report at
BODY = dict(upscale=8, window_size=16, convffn_kernel_size=5, img_size=48)
CONFIGS = {
    'old-bayer64': dict(embed_dim=128, depths=[6] * 6, num_heads=[4] * 6, d_state=8, inner_rank=32,
                        num_tokens=64, mlp_ratio=2., d=16, n=8,
                        align=dict(type='bayer', feat=64)),
    'table-M1':    dict(embed_dim=180, depths=[4] * 6, num_heads=[6] * 6, d_state=64, inner_rank=64,
                        num_tokens=128, mlp_ratio=2., d=64, n=16, up_feat=64,
                        align=dict(type='packed', flow_feat=32, token_feat=64)),
}
if len(sys.argv) > 1:
    CONFIGS = {k: v for k, v in CONFIGS.items() if k in sys.argv[1:]}
 
PARTS = ('align', 'bank', 'kgts', 'conv_first', 'layers', 'conv_after_body',
         'conv_before_upsample', 'upsample', 'conv_last')
 
for name, cfg in CONFIGS.items():
    model = KGTSMamba(**BODY, **cfg).to(DEV).eval()
    x = torch.randn(*BURST, device=DEV)
    if DEV == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        out = model(x)
 
    counts = fc.get_flop_counts()
    B, N, _, H, W = BURST
    P = B * H * W
    def part(p):   # exact module key, else sum its direct ModuleList children (e.g. layers.0, layers.1)
        keys = [f'KGTSMamba.{p}'] if f'KGTSMamba.{p}' in counts else \
               [k for k in counts if re.fullmatch(rf'KGTSMamba\.{p}\.\d+', k)]
        return sum(sum(counts[k].values()) for k in keys) / 2e9
    gm = {p: part(p) for p in PARTS}
    gm['kgts'] += model.n_inject * 2 * P * (N * model.bank.k ** 2) * cfg['d'] * cfg['n'] * 3 / 1e9
    n_layers = sum(cfg['depths'])
    gm['layers'] += n_layers * P * cfg['embed_dim'] * cfg['mlp_ratio'] * cfg['d_state'] * 3 / 1e9
    pm = {p: sum(q.numel() for q in getattr(model, p).parameters()) / 1e6
          for p in PARTS if hasattr(model, p)}
 
    print(f"\n{name}: out {tuple(out.shape)}  "
          f"params {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M  GMACs {sum(gm.values()):.1f}"
          + (f"  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if DEV == 'cuda' else ''))
    for p in PARTS:
        if p in pm:
            print(f"  {p:22s} {gm[p]:7.2f} GMACs  {pm[p]:7.3f} M")
    del model, out
    if DEV == 'cuda':
        torch.cuda.empty_cache()