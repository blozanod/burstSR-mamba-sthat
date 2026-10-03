#!/bin/bash

#$ -M blozanod@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 8         # CPU cores (data loading + per-image FFTs)
#$ -q gpu@@crc_a10   # Specify queue
#$ -S /bin/bash      # This script uses bash syntax; do not inherit the login shell
#$ -N KGTSDiag
#$ -l gpu_card=1
#$ -cwd

# KGTSMamba diagnostics on one checkpoint (analysis/diagnostics/README.md):
#   1. error_bands      error by frequency band, luma/chroma, edges, burst gain per band
#   2. colour_fit       per-image 3x3 + offset colour fit (and nested / tone variants)
#   3. oracle_geometry  generator flow instead of LK on generated validation bursts
#   4. burst_length     true-length burst curve, N = 1..14
# then SUMMARY.md under analysis/outputs/diagnostics/<run name>/<checkpoint>_ema/.
#
# Usage: qsub [-N <job_name>] main/jobs/diagnostics_job.sh <config.yml> [checkpoint] [run_all.py args...]
#   checkpoint: a path, an iteration number, or 'latest' (default) under experiments/<name>/models.
#   Extra args go to analysis/diagnostics/run_all.py, e.g. --draws 3, --only burst_length, --limit 20.
# One GPU is enough (~40 min on an A10 at M1's size, one burst_length draw).

set -uo pipefail

REPO=/groups/rls/blozanod/MambaFusion

if [ -z "${1:-}" ]; then
    echo "Usage: qsub main/jobs/diagnostics_job.sh <config.yml> [checkpoint|latest] [run_all.py args...]"
    exit 1
fi

CONFIG=""
for candidate in "$1" "$PWD/$1" "$REPO/$1" "$REPO/main/configs/$1" "$REPO/main/configs/$1.yml"; do
    if [ -f "$candidate" ]; then
        CONFIG="$(cd "$(dirname "$candidate")" && pwd)/$(basename "$candidate")"
        break
    fi
done
if [ -z "$CONFIG" ]; then
    echo "Error: config not found: $1"
    exit 1
fi
CKPT="${2:-latest}"
shift $(( $# >= 2 ? 2 : 1 ))

echo "======================================================================"
echo "  Config    : $CONFIG"
echo "  Checkpoint: $CKPT"
echo "  Host      : $(hostname)"
echo "  Commit    : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Started   : $(date)"
echo "======================================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

conda activate MambaTraining
cd "$REPO"

python analysis/diagnostics/run_all.py --config "$CONFIG" --ckpt "$CKPT" "$@"
STATUS=$?

echo "======================================================================"
echo "  run_all.py exit status : $STATUS"
echo "  Finished               : $(date)"
echo "======================================================================"
exit $STATUS
