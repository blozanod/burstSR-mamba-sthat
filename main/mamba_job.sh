#!/bin/bash

#$ -M blozanod@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 32        # Specify parallel environment and legal core size
#$ -q gpu@@crc_a10           # Specify queue
#$ -S /bin/bash      # This script uses bash syntax; do not inherit the login shell
#$ -N MambaTraining
#$ -l gpu_card=4
#$ -cwd

# Usage: qsub [-N <job_name>] main/mamba_job.sh <config.yml>
#
# The config argument is accepted as an absolute path, a path relative to the
# submission directory, a path relative to the repo root, or a bare filename
# under main/configs/. Do not assume the job starts in the repo: -cwd is not
# reliable here, so nothing below depends on the working directory. Override
# -N per run so concurrent jobs are distinguishable in qstat.

set -uo pipefail

REPO=/groups/rls/blozanod/MambaFusion

if [ -z "${1:-}" ]; then
    echo "Error: no config file specified."
    echo "Usage: qsub [-N <job_name>] main/mamba_job.sh <config.yml>"
    exit 1
fi

# Resolve the config without depending on the job's working directory.
CONFIG=""
for candidate in "$1" "$PWD/$1" "$REPO/$1" "$REPO/main/configs/$1" "$REPO/main/configs/$1.yml"; do
    if [ -f "$candidate" ]; then
        CONFIG="$(cd "$(dirname "$candidate")" && pwd)/$(basename "$candidate")"
        break
    fi
done

if [ -z "$CONFIG" ]; then
    echo "Error: config not found: $1"
    echo "  Tried, in order:"
    for candidate in "$1" "$PWD/$1" "$REPO/$1" "$REPO/main/configs/$1" "$REPO/main/configs/$1.yml"; do
        echo "    $candidate"
    done
    echo "  Working directory: $PWD"
    echo "  Available configs:"
    ls -1 "$REPO/main/configs/" 2>/dev/null | sed 's/^/    /'
    exit 1
fi

RUN_NAME="$(awk '/^name:/ {print $2; exit}' "$CONFIG")"

echo "======================================================================"
echo "  Config    : $CONFIG"
echo "  Run name  : ${RUN_NAME:-<unset>}"
echo "  Host      : $(hostname)"
echo "  Workdir   : $PWD"
echo "  Job ID    : ${JOB_ID:-<none>}"
echo "  Commit    : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Started   : $(date)"
echo "======================================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "======================================================================"

conda activate MambaTraining
cd "$REPO/main"

# KGTSMamba: ~1 min pre-flight on one GPU before committing all four --
# the GMAC budget, then CUDA scan vs reference (fwd + grads) and a training-step
# memory probe at the config's batch. A failure here means an over-budget model,
# silently wrong gradients or an OOM, so do not train. See
# burstISP/archs/KGTSMamba/budget_check.py and analysis/kgts_sanity.py.
#
# Budget: GMACs per burst at the benchmark input (1, 14, 4, 48, 48), counted like
# the comparison tables (FlopCounter + analytic selective scans). 90 leaves room for
# docs/KGTS_ADVERSARIAL_REVIEW.md's M1 (88.0; 78.1 before it).
MAX_GMACS=90
if grep -Eq '^[[:space:]]*type:[[:space:]]*KGTSMamba' "$CONFIG"; then
    echo "  KGTSMamba budget (budget_check.py --budget $MAX_GMACS GMACs)"
    if ! CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" python -m burstISP.archs.KGTSMamba.budget_check \
            --config "$CONFIG" --budget "$MAX_GMACS"; then
        echo "Over the $MAX_GMACS GMAC budget (or the count failed); not starting training."
        exit 1
    fi
    echo "  KGTSMamba pre-flight (analysis/kgts_sanity.py --skip overfit)"
    if ! CUDA_VISIBLE_DEVICES=0 python "$REPO/analysis/kgts_sanity.py" --config "$CONFIG" --skip overfit; then
        echo "Pre-flight FAILED; not starting training."
        exit 1
    fi
    echo "======================================================================"
fi

torchrun --nproc_per_node=4 train.py -opt "$CONFIG" --launcher pytorch --auto_resume
STATUS=$?

echo "======================================================================"
echo "  train.py exit status : $STATUS"
echo "  Finished             : $(date)"
echo "======================================================================"

# Post-training analysis: resolves the run's experiment folder from the
# config's `name:` field, finds its newest log, and writes the logfile
# dashboard/summary + checkpoint progress visualization under
# analysis/outputs/<name>/. See analysis/run_analysis.py. Runs even when
# training failed, so a crashed run still leaves a partial dashboard.
python "$REPO/analysis/run_analysis.py" --config "$CONFIG"

# Propagate the training status so a failed run is not reported as success.
exit $STATUS
