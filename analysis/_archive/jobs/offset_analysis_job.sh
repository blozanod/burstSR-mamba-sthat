#!/bin/bash

#$ -M blozanod@nd.edu
#$ -m abe
#$ -pe smp 24
#$ -q gpu@@crc_a10
#$ -N OffsetAnalysis
#$ -l gpu_card=4
#$ -cwd

# Usage:
#   qsub offset_analysis_job.sh <path_to_models_dir>
#
# Example:
#   qsub offset_analysis_job.sh /groups/rls/blozanod/MambaFusion/experiments/STHAT_GW/models

if [ -z "$1" ]; then
    echo "Error: No models directory specified."
    echo "Usage: qsub offset_analysis_job.sh <path_to_models_dir>"
    exit 1
fi

conda activate MambaTraining
cd /groups/rls/blozanod/MambaFusion

torchrun --nproc_per_node=4 analysis/_archive/offset_analysis.py \
    --models_dir "$1" \
    --config main/config.yml \
    --data_root /groups/rls/blozanod/MambaFusion/dataset/RealBSR_RAW_testpatch \
    --seed 42 \
    --num_frames 5 \
    --log_dir /groups/rls/blozanod/MambaFusion/analysis/ablation_logs
