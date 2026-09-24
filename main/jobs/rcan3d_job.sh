#!/bin/bash

#$ -M blozanod@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 32        # Specify parallel environment and legal core size
#$ -q gpu@@crc_a10           # Specify queue
#$ -S /bin/bash      # This script uses bash syntax; do not inherit the login shell
#$ -N RCAN3D
#$ -l gpu_card=4
#$ -cwd

# RCAN3D end to end: pre-flight gates -> 100k training -> burst ablations.
#
# Usage: qsub [-N <job_name>] main/jobs/rcan3d_job.sh [config]
#   config defaults to R3D_RCAN3D_RAFT.yml; pass R3D_RCAN3D_NoAlign.yml for
#   the no-alignment control. Resolved like main/mamba_job.sh does (absolute,
#   repo-relative, or a bare name under main/configs/).
#
#   1. Prefetch the RAFT weights once, in one process, so the 4 DDP ranks
#      (and the val/ablation builds) hit the cache instead of racing a
#      download — and fail loudly here if the node is offline and nothing is
#      cached. Skipped for align: none.
#   2. count_macs.py --budget 80: refuse to train an over-budget config.
#   3. shape_check.py: one forward/backward at the config's batch, peak memory.
#   4. torchrun train.py (auto-resume, so a resubmit continues the run).
#   5. run_analysis.py (log dashboard), as in main/mamba_job.sh.
#   6. burst_ablation.py two_pass + frame_drop on the final checkpoint: does
#      the 3D trunk actually use the burst, and how does PSNR scale with N?

set -uo pipefail

REPO=/groups/rls/blozanod/MambaFusion
ARG="${1:-R3D_RCAN3D_RAFT.yml}"

CONFIG=""
for candidate in "$ARG" "$PWD/$ARG" "$REPO/$ARG" "$REPO/main/configs/$ARG" "$REPO/main/configs/$ARG.yml"; do
    if [ -f "$candidate" ]; then
        CONFIG="$(cd "$(dirname "$candidate")" && pwd)/$(basename "$candidate")"
        break
    fi
done
if [ -z "$CONFIG" ]; then
    echo "Error: config not found: $ARG"
    ls -1 "$REPO/main/configs/" 2>/dev/null | sed 's/^/    /'
    exit 1
fi

RUN_NAME="$(awk '/^name:/ {print $2; exit}' "$CONFIG")"
ALIGN="$(awk '/^  align:/ {print $2; exit}' "$CONFIG")"
EXP_DIR="$REPO/experiments/$RUN_NAME"
VAL_ROOT="$REPO/dataset/SyntheticBurstVal"

echo "======================================================================"
echo "  Config    : $CONFIG"
echo "  Run name  : ${RUN_NAME:-<unset>}   align: ${ALIGN:-<unset>}"
echo "  Host      : $(hostname)   Job ID: ${JOB_ID:-<none>}"
echo "  Commit    : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Started   : $(date)"
echo "======================================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

conda activate MambaTraining
cd "$REPO"

# --- 1. RAFT weights --------------------------------------------------------
if [ "$ALIGN" = "raft_large" ] || [ "$ALIGN" = "raft_small" ]; then
    python - "$ALIGN" <<'EOF' || { echo "ERROR: could not load RAFT weights (offline node? prefetch on a login node, see config header)"; exit 1; }
import sys
from torchvision.models import optical_flow as of
variant = sys.argv[1]
weights = {'raft_large': of.Raft_Large_Weights, 'raft_small': of.Raft_Small_Weights}[variant].DEFAULT
getattr(of, variant)(weights=weights)
print(f'RAFT weights ready: {variant} ({weights})')
EOF
fi

# --- 2-3. Pre-flight gates --------------------------------------------------
python analysis/count_macs.py "$CONFIG" --budget 80 || { echo "ERROR: over the 80 GMAC budget"; exit 1; }
python analysis/shape_check.py "$CONFIG" || { echo "ERROR: shape_check failed"; exit 1; }

# --- 4. Train ---------------------------------------------------------------
cd "$REPO/main"
torchrun --nproc_per_node=4 train.py -opt "$CONFIG" --launcher pytorch --auto_resume
STATUS=$?
cd "$REPO"
echo "  train.py exit status : $STATUS   ($(date))"

# --- 5. Log dashboard -------------------------------------------------------
python analysis/run_analysis.py --config "$CONFIG" --skip-progress

# --- 6. Burst ablations on the final checkpoint -----------------------------
CKPT="$EXP_DIR/models/net_g_latest.pth"
[ -f "$CKPT" ] || CKPT="$(ls -1 "$EXP_DIR"/models/net_g_*.pth 2>/dev/null | sort -V | tail -1)"
if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
    echo "=== Burst ablations on $CKPT ==="
    for MODE in two_pass frame_drop; do
        torchrun --nproc_per_node=4 analysis/burst_ablation.py \
            --model_path "$CKPT" --mode "$MODE" --drop_counts 1,2,5,9,14 \
            --config "$CONFIG" --dataset synburst --data_root "$VAL_ROOT" --num_frames 14 \
            --log_dir "$REPO/analysis/outputs/$RUN_NAME/ablation"
    done
else
    echo "WARN: no checkpoint under $EXP_DIR/models; skipping burst ablations."
fi

exit $STATUS
