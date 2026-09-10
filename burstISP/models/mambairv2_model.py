from burstISP.utils.registry import MODEL_REGISTRY
from burstISP.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class MambaIRv2Model(SRModel):
    """MambaIRv2 model for image restoration."""

    def feed_data(self, data):
        # The shared burst datasets always emit lq as [B, N, C, H, W] (a
        # keyframe-only run sets num_frames: 1). MambaIRv2 is a plain conv/
        # attention net over [B, C, H, W], so collapse the singleton burst
        # axis here rather than teaching the datasets about non-burst models.
        lq = data['lq'].to(self.device)
        if lq.dim() == 5:
            assert lq.shape[1] == 1, (
                f'MambaIRv2Model takes a single keyframe; got burst dim {lq.shape[1]}. '
                'Set num_frames: 1 in the dataset config.')
            lq = lq.squeeze(1)
        self.lq = lq
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    # No test() override: the inherited SRModel.test() (a single full-frame
    # forward) is correct here and the tile-partitioned version this used to
    # be is not worth reinstating. It was written for same-channel-count
    # tasks (denoising, in_chans == out_chans) -- it allocated the merged
    # output buffer with the *input* channel count and never moved it to the
    # GPU, so it crashed on the first SR validation step with a channel-count
    # RuntimeError. Our crops (48x48 packed -> 384x384) are well under the
    # 200px threshold where its splitting logic would even activate, so
    # tiling was always a no-op here anyway.