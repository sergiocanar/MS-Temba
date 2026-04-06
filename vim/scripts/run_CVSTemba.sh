#!/usr/bin/env bash
# run_CVSTemba.sh
# Single training run of CVSTemba. Update OUTPUT_DIR before running.

CVS_ROOT="/home/scanar/endovis/Datasets/SAGES_2024"
OUTPUT_DIR="/path/to/output/cvs_run1"
NUM_CLIPS=30   # window size in frames; keyframe is always the last frame

cd "$(dirname "$0")/.."   # run from vim/

conda run -n cvs_temba python CVSTemba_main.py \
    --gpu        0 \
    --epochs     50 \
    --batch_size 8 \
    --num_clips  ${NUM_CLIPS} \
    --skip       1 \
    --alpha_l    1.0 \
    --beta_l     0.05 \
    --model      cvstemba \
    --cvs_root   "${CVS_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --lr         5e-4 \
    --opt        adamw \
    --weight_decay  0.01 \
    --sched         cosine \
    --warmup_epochs 5 \
    --min_lr        1e-5 \
    --wandb_project cvstemba
