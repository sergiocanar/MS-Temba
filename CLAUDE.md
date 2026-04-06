# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MS-Temba (Multi-Scale Temporal Mamba) is a Temporal Action Detection (TAD) model for untrimmed videos. It operates on **pre-extracted video features** (not raw video) and applies hierarchical dilated State Space Models (Mamba/SSM) for multi-scale temporal modeling. Target datasets: Charades (157 classes), Toyota Smarthome Untrimmed/TSU (51 classes), MultiTHUMOS.

## Environment Setup

The conda environment for this project is **`cvs_temba`**.

```bash
conda create -n cvs_temba python=3.10.13
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r vim/vim_requirements.txt
pip install wandb torchmetrics

# Install local Mamba/causal-conv1d (order matters — mamba must be --no-deps
# so it doesn't pull a newer causal_conv1d that breaks the API)
pip install --no-build-isolation causal-conv1d/
pip install --no-build-isolation --no-deps mamba-1p1p1/
```

After installing, two files from `mamba-1p1p1/` must be copied into the
installed `mamba_ssm` package, and the `causal_conv1d` call signatures patched
to match causal_conv1d 1.0.0 (which dropped the `seq_idx` argument):

```bash
export MAMBA_SITE=$(python -c "import mamba_ssm, os; print(os.path.dirname(mamba_ssm.__file__))")

# 1. Copy the getC custom modules (not included by the installer)
cp mamba-1p1p1/mamba_ssm/modules/mamba_simple_getC.py       $MAMBA_SITE/modules/
cp mamba-1p1p1/mamba_ssm/ops/selective_scan_interface_getC.py $MAMBA_SITE/ops/

# 2. Patch call signatures: remove seq_idx=None (incompatible with causal_conv1d 1.0.0)
python - <<'EOF'
import re, os
path = os.path.join(os.environ["MAMBA_SITE"], "ops/selective_scan_interface_getC.py")
src = open(path).read()
src = re.sub(r'(causal_conv1d_cuda\.causal_conv1d_fwd\(.+?),\s*None,\s*True\)',
             r'\1, True)', src)
src = re.sub(r'(causal_conv1d_cuda\.causal_conv1d_bwd\([^)]+?),\s*None,\s*(dx, True)',
             r'\1, \2', src)
open(path, 'w').write(src)
print("patched", path)
EOF
```

**Why this is needed**: `mamba-1p1p1`'s `selective_scan_interface_getC.py` was
written against a causal_conv1d version that added a `seq_idx` positional
argument. The local `causal-conv1d/` (v1.0.0) does not have it, causing a
`TypeError` on both forward and backward passes.

## Training Commands

All training is run from within the `vim/` directory:

```bash
cd vim

# Train on TSU dataset
bash scripts/run_MSTemba_TSU.sh

# Train on Charades dataset
bash scripts/run_MSTemba_Charades.sh
```

Key arguments for `MSTemba_main.py`:
- `-dataset`: `charades` or `tsu`
- `-backbone`: `i3d` (in_feat_dim=1024) or `clip` (in_feat_dim=768)
- `-rgb_root`: path to directory containing `.npy` feature files per video
- `-num_clips`: sequence length (256 for Charades, 2500 for TSU)
- `-unisize True`: pads all sequences to `-num_clips` length (required for batching)
- `-output_dir`: where checkpoints, logs, and per-epoch `.pkl` prediction files are saved
- `-alpha_l`, `-beta_l`: loss weights for final output and block auxiliary losses

## Architecture

All model/training code lives in `vim/`:

- **`models_MSTemba.py`**: Core model. `MSTemba` is the main class, registered with `timm` as `mstemba`. It stacks three hierarchical Temba blocks:
  - Block 1: single `VisionMamba` SSM on full sequence
  - Block 2: two parallel SSMs on even/odd-interleaved tokens (dilation factor 2)
  - Block 3: three parallel SSMs on 1-in-3 token groups (dilation factor 3)
  - A `MultiScaleMambaFuser` (interaction block) aggregates all three scales via another SSM
  - Each block has its own classification head (`block_heads`) for auxiliary supervision
  - `compute_c_state_diversity_loss_simple` computes pairwise cosine diversity loss between C-states of parallel SSMs to encourage differentiated representations

- **`MSTemba_main.py`**: Training loop. Uses `timm.create_model('mstemba', ...)` to instantiate the model. Loss = `alpha_l * (final_BCE + 0.3 * sum(block_BCEs)) + 100.0 * diversity_loss`. Saves best model checkpoint based on `val_map`.

- **`charades_dataloader.py`**: `Charades` dataset class handles both Charades and TSU (same format). Loads per-video `.npy` feature files, generates frame-level binary labels and Gaussian heatmaps centered at action midpoints.

- **`apmeter.py`**: Tracks per-class Average Precision during training/validation. `val_map` = mean over nonzero-AP classes.

- **`engine.py`**, **`utils.py`**: Utilities including `sampled_25` (samples 25 frames for evaluation), `mask_probs`, and `generate_gaussian`.

## Data Format

Input features are stored as `.npy` files named `<video_id>.npy` with shape `(T, feat_dim)` where T is number of frames. Dataset JSON files (in `data/`) map video IDs to action annotations with format `[class_idx, start_time, end_time]`.

## Output Structure

Training saves to `-output_dir`:
- `best_model.pth`: best checkpoint by val mAP
- `<epoch>.pkl`: prediction probabilities dict at best epoch
- `block_N/best_model.pth`: per-block best checkpoints
- `block_N/<epoch>.pkl`: per-block prediction probabilities
- `tensorboard_logs/`: TensorBoard event files
- `training.log`: full training log

## Mamba Dependencies

The custom Mamba layer at `mamba-1p1p1/mamba_ssm/modules/mamba_simple_getC.py` is a modified version that returns the C state (SSM selection matrix) alongside hidden states — this is non-standard and required for the diversity loss. The `causal-conv1d/` subdir is also a local install.
