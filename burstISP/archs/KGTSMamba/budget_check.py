"""Budget check: params / GMACs / peak memory for candidate configs. Forward only, fits a 4 GB GPU.

    python -m burstISP.archs.KGTSMamba.budget_check [preset names...]
    python -m burstISP.archs.KGTSMamba.budget_check --config main/configs/M1_KGTSMamba.yml [--budget 90]

Always at the benchmark input (1, 14, 4, 48, 48), whatever the config's training crop.
With --budget, exits 1 if the total exceeds it (main/mamba_job.sh's pre-flight gate).

GMACs are only meaningful on CUDA: the selective scans are then custom kernels the
FlopCounter cannot see, and are added analytically below. On CPU mamba_ssm's
reference scan is einsum-based and gets counted too (double-counting `layers`).

The total is the FlopCounter's GLOBAL count (+ the analytic scans), not the sum of the
per-part rows: a part's row only holds ops run while that module's forward is on the
stack, and KGTS.precompute's token-side Linears (W_u, W_delta, W_B, W_k; plus the rebuilt
cache after each `refine` entry) are called from KGTSMamba.forward directly. They show up
as the `other` row -- 4.2 GMACs in 236dbba's M1, whose header said ~75.0 GMACs for what
this count puts at 78.1.
"""
import argparse
import re
import sys
import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode
from burstISP.archs.KGTSMamba.kgts_mamba_arch import KGTSMamba

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BURST = (1, 14, 4, 48, 48)      # benchmark input -- match what QMambaBSR / BurstMamba report at
BODY = dict(upscale=8, window_size=16, convffn_kernel_size=5, img_size=48)
M1_BODY = dict(embed_dim=180, depths=[4] * 6, num_heads=[6] * 6, d_state=64, inner_rank=64,
               num_tokens=128, mlp_ratio=2., upsampler='pixelshuffle', upsample_feat=64)
# the KGTS corrections switched off (see KGTS / TokenBank / KGTSAlign docstrings)
ORIG_TOKEN = dict(pos_freqs=(), norm=False, mark_ref=False, pin_ref=False, tap_pos='target')
ORIG_KGTS = dict(affinity=False, dt_norm=False)
ORIG_ALIGN = dict(global_motion=None)
CONFIGS = {
    # the original KGTS: d=16, n=8, one head -- 32 dims injected per call
    'old-bayer64': dict(embed_dim=128, depths=[6] * 6, num_heads=[4] * 6, d_state=8, inner_rank=32,
                        num_tokens=64, mlp_ratio=2., align=dict(type='bayer', flow_feat=64, **ORIG_ALIGN),
                        token=dict(c=64, d=16, **ORIG_TOKEN),
                        kgts=dict(n=8, a_max=8, **ORIG_KGTS)),
    'table-M1':    dict(**M1_BODY, align=dict(type='packed', flow_feat=32),
                        token=dict(c=64, d=64), kgts=dict(n=16)),
    'M1':          dict(**M1_BODY, align=dict(type='packed', flow_feat=32),
                        token=dict(c=64, d=64), kgts=dict(n=16, heads=4)),
    'M1-bayer':    dict(**M1_BODY, align=dict(type='bayer', flow_feat=48, flow_in_chans=48),
                        token=dict(c=64, d=64), kgts=dict(n=16, heads=4)),
    # main/configs/M1_KGTSMamba.yml (docs/KGTS_ADVERSARIAL_REVIEW.md); --config reads the file itself
    'M1-wide':     dict(**M1_BODY, inject_first=True, aux_head=True, refine=dict(at=[2], hidden=128),
                        align=dict(type='packed', flow_feat=32, global_motion='lk', token_blocks=2),
                        token=dict(c=64, d=64),
                        kgts=dict(n=16, heads=8, expand=2, roles=['int'] * 4 + ['geo'] * 2 + ['con'] * 2,
                                  out_norm='group', depth_embed=True, untie_out=True)),
    # M1_KGTSMamba.yml at 236dbba, before the review, for the cost of its changes
    'M1-wide-236dbba': dict(**M1_BODY, align=dict(type='packed', flow_feat=32, global_motion='affine'),
                            token=dict(c=64, d=64), kgts=dict(n=16, heads=8, expand=2)),
    # M1-wide before the KGTS corrections, for the cost of the fixes
    'M1-wide-orig': dict(**M1_BODY, align=dict(type='packed', flow_feat=32, **ORIG_ALIGN),
                         token=dict(c=64, d=64, **ORIG_TOKEN),
                         kgts=dict(n=16, heads=8, expand=2, a_max=16, **ORIG_KGTS)),
}

# refiners: TokenRefine (its follow-up KGTS.precompute lands in `other`); aux runs only with
# return_aux, so its ~0.11 GMACs (M1) are not in the total
PARTS = ('align', 'bank', 'kgts', 'refiners', 'aux', 'conv_first', 'layers', 'conv_after_body',
         'conv_before_upsample', 'upsample', 'conv_last')


def measure(name, cfg):
    """Prints the breakdown; returns total GMACs (FlopCounter global + analytic scans)."""
    model = KGTSMamba(**cfg).to(DEV).eval()
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
    gm['other'] = fc.get_total_flops() / 2e9 - sum(gm.values())
    # one scan call per injection over 2*d_inner channels (fwd + bwd stacked), L = N*k*k taps
    kg = model.kgts
    gm['kgts'] += model.n_inject * P * (N * model.bank.k ** 2) * 2 * kg.di * kg.n * 3 / 1e9
    gm['layers'] += sum(cfg['depths']) * P * cfg['embed_dim'] * cfg['mlp_ratio'] * cfg['d_state'] * 3 / 1e9
    pm = {p: sum(q.numel() for q in getattr(model, p).parameters()) / 1e6
          for p in PARTS if getattr(model, p, None) is not None}
    total = sum(gm.values())

    print(f"\n{name}: out {tuple(out.shape)}  "
          f"params {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M  GMACs {total:.1f}"
          + (f"  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if DEV == 'cuda' else ''))
    print(f"  KGTS: d={kg.d} d_inner={kg.di} n={kg.n} heads={kg.heads} -> "
          f"{2 * kg.di} pooled dims per injection, x{model.n_inject} injections")
    for p in PARTS + ('other',):
        if p in pm or (p == 'other' and abs(gm[p]) > 1e-3):
            print(f"  {p:22s} {gm[p]:7.2f} GMACs" + (f"  {pm[p]:7.3f} M" if p in pm else
                                                     '            (KGTS.precompute: token-side Linears)'))
    del model, out
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('presets', nargs='*', help=f'preset names ({", ".join(CONFIGS)}); default all')
    ap.add_argument('--config', help='a KGTSMamba YAML config: measure its network_g instead of presets')
    ap.add_argument('--budget', type=float, default=None, help='GMACs; exit 1 if a measured total exceeds it')
    args = ap.parse_args()
    if args.budget is not None and DEV != 'cuda':
        print('WARNING: on CPU the reference selective scans are counted too; the total overstates GMACs')
    if args.config:
        with open(args.config) as f:
            net = dict(yaml.safe_load(f)['network_g'])
        if net.pop('type') != 'KGTSMamba':
            raise SystemExit(f'{args.config}: network_g.type is not KGTSMamba')
        todo = {args.config: net}
    else:
        unknown = [p for p in args.presets if p not in CONFIGS]
        if unknown:
            raise SystemExit(f'unknown presets {unknown}; choose from {list(CONFIGS)}')
        todo = {k: {**BODY, **v} for k, v in CONFIGS.items() if not args.presets or k in args.presets}
    ok = True
    for name, cfg in todo.items():
        total = measure(name, cfg)
        if args.budget is not None:
            over = total > args.budget
            ok &= not over
            print(f"  budget {args.budget:.1f} GMACs: {'OVER' if over else 'OK'} ({total:.1f})")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
