import torch
from burstISP.archs.arch_util import DCNv4Block

torch.manual_seed(0)
dev = 'cuda'
B, C, H, W, G = 1, 64, 32, 32, 4

dcn = DCNv4Block(channels=C, kernel_size=3, pad=1, stride=1, groups=G).to(dev)
with torch.no_grad():
    dcn.value_proj.weight.copy_(torch.eye(C).view(C, C, 1, 1))
    dcn.value_proj.bias.zero_()
    dcn.output_proj.weight.copy_(torch.eye(C).view(C, C, 1, 1))
    dcn.output_proj.bias.zero_()

x = torch.randn(B, C, H, W, device=dev)

@torch.no_grad()
def peak(om):
    return dcn(x, om).abs().max().item()

# --- sanity rungs, before any scanning ---
print('all zeros ->', peak(torch.zeros(B, 112, H, W, device=dev)))
print('all ones  ->', peak(torch.ones (B, 112, H, W, device=dev)))

# --- the scan ---
live = []
for c in range(112):
    om = torch.zeros(B, 112, H, W, device=dev)
    om[:, c] = 1.0
    if peak(om) > 1e-6:
        live.append(c)
print('live channels:', live)

@torch.no_grad()
def probe(**slots):
    om = torch.zeros(B, 112, H, W, device=dev)
    for k, v in slots.items():
        om[:, int(k[2:])] = v
    return dcn(x, om)

def best_shift(out):
    """Which (dy, dx) roll of x does this output match? Group 0 owns ch 0..15."""
    best = (1e9, None, None)
    for sy in range(-3, 4):
        for sx in range(-3, 4):
            ref = torch.roll(x, (sy, sx), (2, 3))
            e = (out[:, :16, 2:-2, 2:-2] - ref[:, :16, 2:-2, 2:-2]).abs().max().item()
            if e < best[0]:
                best = (e, sy, sx)
    return best

print('mask only      :', best_shift(probe(ch18=1.0)))
print('mask + ch0=+2  :', best_shift(probe(ch18=1.0, ch0=2.0)))
print('mask + ch1=+2  :', best_shift(probe(ch18=1.0, ch1=2.0)))