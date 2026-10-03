#!/bin/bash

#$ -M blozanod@nd.edu
#$ -m abe
#$ -pe smp 24
#$ -q gpu@@crc_a10
#$ -N GateAMotion
#$ -l gpu_card=3
#$ -cwd

# Gate-A: Inter-frame camera motion measurement (CPU-only, no GPU needed).
# Runs phase cross-correlation across all train + test bursts (~25k total)
# and writes gate_a_magnitudes.csv and gate_a_hist.png to analysis/.
#
# Usage (smoke test — first 200 bursts):
#   qsub gate_a_motion_job.sh --limit 200
#
# Usage (full dataset):
#   qsub gate_a_motion_job.sh
#
# The script parallelises across all allocated CPUs automatically.

conda activate MambaTraining
cd /groups/rls/blozanod/MambaFusion

python analysis/_archive/gate_a_motion.py \
    --config main/config.yml \
    --out-dir /groups/rls/blozanod/MambaFusion/analysis \
    "$@"
