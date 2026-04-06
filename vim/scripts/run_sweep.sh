#!/usr/bin/env bash
# run_sweep.sh
#
# 1. Registers the sweep with wandb and prints the SWEEP_ID.
# 2. Starts one agent on this machine.
#
# Usage:
#   bash scripts/run_sweep.sh                     # single agent, GPU 0
#   GPU=1 bash scripts/run_sweep.sh               # single agent, GPU 1
#   bash scripts/run_sweep.sh --count 20          # limit to 20 runs
#
# To run agents in parallel (e.g. 2 GPUs):
#   SWEEP_ID=$(wandb sweep scripts/sweep_cvstemba.yaml 2>&1 | grep -oP '(?<=sweep ID: )\S+')
#   CUDA_VISIBLE_DEVICES=0 wandb agent $SWEEP_ID &
#   CUDA_VISIBLE_DEVICES=1 wandb agent $SWEEP_ID &

set -e
GPU=${GPU:-0}
COUNT=${1:-}   # optional --count N passed through to wandb agent

cd "$(dirname "$0")/.."   # run from vim/

# Register the sweep (only needs to happen once)
SWEEP_OUTPUT=$(conda run -n cvs_temba wandb sweep scripts/sweep_cvstemba.yaml 2>&1)
echo "$SWEEP_OUTPUT"
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP "(?<=wandb agent )\S+")

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: could not parse sweep ID from wandb output."
    exit 1
fi

echo ""
echo "Sweep ID: ${SWEEP_ID}"
echo "Starting agent on GPU ${GPU} ..."

CUDA_VISIBLE_DEVICES=${GPU} conda run -n cvs_temba \
    wandb agent ${COUNT:+--count $COUNT} "${SWEEP_ID}"
