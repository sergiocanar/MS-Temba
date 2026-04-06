"""
prepare_cvs_features.py

Converts the SAGES_2024 per-frame .pth feature files into:
  - Per-video .npy feature arrays  : {output_dir}/{split}/{video_name}.npy       shape (T, 1024) float32
  - Per-video .npy CA label arrays : {output_dir}/{split}/{video_name}_labels.npy shape (T, 3)    float32

Label strategy: confidence-aware (CA) labels are read from keyframe .pth files
(frame_id % 150 == 0). All other frames receive the label of their nearest keyframe
via nearest-neighbour propagation.

Usage:
    python prepare_cvs_features.py \
        --dataset_root /home/scanar/endovis/Datasets/SAGES_2024 \
        --output_dir   /path/to/cvs_npy \
        [--splits train val test]
"""

import os
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm

KEYFRAME_INTERVAL = 150   # every 5 s at 30 fps
FEAT_DIM          = 1024
NUM_CLASSES       = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_frame_paths(video_feat_dir: str):
    """Return list of (frame_id, path) sorted by frame_id."""
    paths = glob.glob(os.path.join(video_feat_dir, "frame_*.pth"))
    result = []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]   # "frame_XXXXXX"
        try:
            fid = int(base.split("_")[-1])
            result.append((fid, p))
        except ValueError:
            print(f"  Warning: skipping unexpected file {p}")
    return sorted(result, key=lambda x: x[0])


def _nearest_propagate(keyframe_ids: list, keyframe_labels: dict, T: int) -> np.ndarray:
    """
    Build a (T, NUM_CLASSES) label array by nearest-neighbour propagation
    from the annotated keyframes.

    keyframe_ids   : sorted list of annotated frame ids
    keyframe_labels: {frame_id: np.ndarray shape (NUM_CLASSES,)} of CA values
    T              : total number of frames in the video
    """
    labels = np.zeros((T, NUM_CLASSES), dtype=np.float32)
    if not keyframe_ids:
        return labels

    kf_arr = np.array(keyframe_ids, dtype=np.int64)   # sorted
    for f in range(T):
        idx = np.searchsorted(kf_arr, f)
        if idx == 0:
            nearest = kf_arr[0]
        elif idx >= len(kf_arr):
            nearest = kf_arr[-1]
        else:
            before = kf_arr[idx - 1]
            after  = kf_arr[idx]
            nearest = before if (f - before) <= (after - f) else after
        labels[f] = keyframe_labels[int(nearest)]

    return labels


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(video_feat_dir: str, output_video_dir: str, video_name: str):
    """
    Extract features and CA labels for one video.
    Writes {video_name}.npy and {video_name}_labels.npy into output_video_dir.
    Skips if both files already exist.
    """
    feat_out  = os.path.join(output_video_dir, f"{video_name}.npy")
    label_out = os.path.join(output_video_dir, f"{video_name}_labels.npy")

    if os.path.exists(feat_out) and os.path.exists(label_out):
        return  # already done

    frame_list = _sorted_frame_paths(video_feat_dir)
    if not frame_list:
        print(f"  Warning: no .pth files found in {video_feat_dir}, skipping.")
        return

    T = len(frame_list)
    features        = np.zeros((T, FEAT_DIM), dtype=np.float32)
    keyframe_ids    = []
    keyframe_labels = {}

    for i, (fid, fpath) in enumerate(frame_list):
        payload = torch.load(fpath, map_location="cpu", weights_only=False)

        # --- features (float16 → float32) ---
        emb = payload["image_embeddings"]
        features[i] = emb.float().numpy()

        # --- CA labels (only at keyframes) ---
        if fid % KEYFRAME_INTERVAL == 0:
            ca = payload.get("labels_ca")
            if ca is not None:
                keyframe_ids.append(fid)
                keyframe_labels[fid] = ca.float().numpy()

    # Propagate CA labels to all frames
    labels = _nearest_propagate(keyframe_ids, keyframe_labels, T)

    os.makedirs(output_video_dir, exist_ok=True)
    np.save(feat_out,  features)
    np.save(label_out, labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare CVS .npy features from SAGES_2024 .pth files")
    parser.add_argument("--dataset_root", required=True,
                        help="Root of SAGES_2024 dataset (contains train/, val/, test/ subdirs)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write the .npy files (mirrored split structure)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to process (default: train val test)")
    args = parser.parse_args()

    for split in args.splits:
        feat_root = os.path.join(args.dataset_root, split, "features")
        if not os.path.isdir(feat_root):
            print(f"[{split}] features dir not found: {feat_root}, skipping.")
            continue

        out_split_dir = os.path.join(args.output_dir, split)
        os.makedirs(out_split_dir, exist_ok=True)

        video_names = sorted(os.listdir(feat_root))
        print(f"[{split}] Processing {len(video_names)} videos ...")

        for vname in tqdm(video_names, desc=split):
            video_feat_dir = os.path.join(feat_root, vname)
            if not os.path.isdir(video_feat_dir):
                continue
            process_video(video_feat_dir, out_split_dir, vname)

        # Quick sanity check on the first video
        sample_feat = glob.glob(os.path.join(out_split_dir, "*.npy"))
        sample_feat = [p for p in sample_feat if "_labels" not in p]
        if sample_feat:
            f = np.load(sample_feat[0])
            l = np.load(sample_feat[0].replace(".npy", "_labels.npy"))
            vname_check = os.path.splitext(os.path.basename(sample_feat[0]))[0]
            print(f"  [{split}] Sample '{vname_check}': features {f.shape} {f.dtype}, "
                  f"labels {l.shape} {l.dtype}, "
                  f"label range [{l.min():.3f}, {l.max():.3f}], "
                  f"positive frames {(l > 0.5).any(axis=1).sum()}/{f.shape[0]}")

    print("Done.")


if __name__ == "__main__":
    main()
