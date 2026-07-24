import torch

from burstISP.data.dbsr.image_quality import PSNR
from burstISP.utils.registry import METRIC_REGISTRY

# One PSNR module per boundary_ignore value (the module is stateless apart
# from its config, so caching avoids re-instantiation every val image).
_psnr_modules = {}


@METRIC_REGISTRY.register()
def calculate_psnr_synburst(img, img2, boundary_ignore=40, **kwargs):
    """Official SyntheticBurst PSNR, via the vendored DBSR evaluation path.

    Mirrors evaluation/synburst/compute_score.py from the DBSR toolkit (see
    burstISP/data/dbsr/README.md): the prediction is quantized to 14 bits
    (consistent with evaluating on saved 16-bit PNGs), then scored with the
    official PSNR(boundary_ignore=40) against the linear-RGB ground truth.

    Args:
        img (Tensor | ndarray): network output, [3, H, W] linear RGB.
        img2 (Tensor | ndarray): ground truth, [3, H, W] linear RGB in [0, 1].
        boundary_ignore (int): boundary pixels ignored by the official metric.
            Default 40 (the official evaluation setting).

    Returns:
        float: PSNR in dB.
    """
    if not torch.is_tensor(img):
        img = torch.from_numpy(img)
    if not torch.is_tensor(img2):
        img2 = torch.from_numpy(img2)

    img = img.detach().float().cpu()
    img2 = img2.detach().float().cpu()
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    # Official quantization step: consistent with scoring saved images
    pred_int = (img.clamp(0.0, 1.0) * 2 ** 14).short()
    pred = pred_int.float() / (2 ** 14)

    if boundary_ignore not in _psnr_modules:
        _psnr_modules[boundary_ignore] = PSNR(boundary_ignore=boundary_ignore)
    value = _psnr_modules[boundary_ignore](pred, img2)

    return value.item() if torch.is_tensor(value) else float(value)
