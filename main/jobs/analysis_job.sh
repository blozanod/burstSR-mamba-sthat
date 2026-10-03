#!/bin/bash

#$ -M blozanod@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 8         # CPU cores
#$ -q gpu@@crc_a10   # Specify queue
#$ -S /bin/bash      # This script uses bash syntax; do not inherit the login shell
#$ -N RunAnalysis
#$ -l gpu_card=1
#$ -cwd

# analysis/run_analysis.py on one GPU, any time -- mid-run included: the log dashboard plus the
# checkpoint progress visualization (visualize_progress.py) over every checkpoint saved so far.
# The same thing main/mamba_job.sh runs after training. For SyntheticBurst configs the progress
# stage uses in-domain SyntheticBurstVal bursts (--source auto).
#
# Usage: qsub [-N <job_name>] main/jobs/analysis_job.sh <config.yml> [run_analysis.py args...]
#   e.g. ... M1_KGTSMamba.yml --source zurich    (bursts generated from Zurich test instead)
# Outputs: analysis/outputs/<run name>/ (dashboard) and .../progress/ (strips, curves.png).

set -uo pipefail

REPO=/groups/rls/blozanod/MambaFusion

if [ -z "${1:-}" ]; then
    echo "Usage: qsub main/jobs/analysis_job.sh <config.yml> [run_analysis.py args...]"
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
shift

echo "  Config : $CONFIG   Host: $(hostname)   Commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
conda activate MambaTraining
cd "$REPO"
python analysis/run_analysis.py --config "$CONFIG" "$@"
