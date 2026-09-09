Dataset Structure:


```text
.
├── README.md
├── RealBSR_RAW_testpatch       # local mirror of the cluster val/test split
│   ├── xxx_xxxx_RAW
│   │   └── xxx_MFSR_Sony_xxxx_x1_01.png
│   │   └── xxx_MFSR_Sony_xxxx_x1_xx.png
│   │   └── xxx_MFSR_Sony_xxxx_x1_14.png
│   │   └── xxx_MFSR_Sony_xxxx_x4_rgb.png
│   │   └── MFSR_Sony_xxxx_x4.pkl
├── RealBSR_RAW_trainpatch      # small local sample of the train split, for
│   │                           # exercising the data pipeline locally (real
│   │                           # training runs on the cluster against the
│   │                           # full train split, not this local copy)
│   ├── xxx_xxxx_RAW
│   │   └── xxx_MFSR_Sony_xxxx_x1_01.png
│   │   └── xxx_MFSR_Sony_xxxx_x1_xx.png
│   │   └── xxx_MFSR_Sony_xxxx_x1_14.png
│   │   └── xxx_MFSR_Sony_xxxx_x4_rgb.png
│   │   └── MFSR_Sony_xxxx_x4.pkl
└── Inference_Set                # 10 curated test-set bursts (with GT) used
    │                             # as the standard local set for inference/
    │                             # visualization scripts
    ├── xxx_xxxx
    │   └── xxx_MFSR_Sony_xxxx_x1_01.png
    │   └── ...
    │   └── xxx_MFSR_Sony_xxxx_x4_rgb.png
    │   └── MFSR_Sony_xxxx_x4.pkl
```

`_archive/` (gitignored, not shown above) holds retired scratch data that predates this structure — single-burst leftovers from early development, kept for reference rather than deleted.