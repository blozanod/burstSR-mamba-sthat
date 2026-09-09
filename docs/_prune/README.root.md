# **TODO Proper Readme File**

MambaFusion is a work-in-progress RAW burst image restoration model, currently built and trained for 4x image upscaling.

**Currently training MambaFusion (4th iteration)**

Results have been promising thus far, in that the model outputs upscaled images. However, they are quite blurry. This is likely due to misalignment and a cheap fusion mechanism.

---

# Planned/Implemented Changes

**New Transformer-Based Cross-Frame Fusion Plan: Done!**

    Center frame feature (frame 7): Query
    All frame features: key/values
    Use window paritioning for manageable compute (like in SwinIR)
    Output: Single fused feature map: [B, C, H, W]

**SR Head Modifications: Done!**

    Drop shallow feature extraction (first conv), as it has already been done by other steps (notably transformer)
    Do this by changing kernel to 1 from 3.

**New Alignment Loss:**

    Calculate alignment loss by returning aligned_burst variable to training function if it is training. Then remove the center frame from the burst and use it as a reference instead.
    Then add the alignment loss to the total loss. This way alignment loss gradients reach alignment module.
    For this, it is wise to scale the loss over time (0.5x -> 0.10x -> 0.0x)

**New Alignment: Done!**

    Swap DCNv2 for DCNv4, as well as interpolation for PixelShuffle for better upscaling in pyramid
    REMEMBER: REQUIRES COMPILING DCNV4 KERNELS AND CHANGING CONFIG.YML SO OFFSET GROUPS = 4, OFFSETS = 64

---

# **Future Improvements:**

One whole joint architecture (not 3-4 separate modules)

Supervise training with PWC-Net for greater alignment accuracy
