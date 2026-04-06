"""
cvs_dataloader.py

PyTorch dataset for the CVS (Critical View of Safety) surgical dataset.

Each sample is a fixed-length window of frames ending at a keyframe (online /
causal).  The keyframe label (CA soft label, shape (3,)) is the supervision
signal; all other frames in the window are unlabelled context.

Expected directory layout (raw SAGES_2024 structure):
    {dataset_root}/
      train/
        features/
          {video_name}/
            frame_000000.pth   # keys: image_embeddings (1024,) float16
            frame_000001.pth   #       labels_ca (3,) float32  |  None
            ...
      val/
        features/
          {video_name}/
            ...

Only keyframe files (frame_id % 150 == 0) carry non-None labels_ca.
Windows shorter than num_clips (early keyframes) are zero-padded at the start.

Batch format returned by cvs_collate_fn:
    features  (B, C, T, 1, 1)   C=1024, T=num_clips
    labels    (B, 3)             CA label of the keyframe (last frame)
    metas     list of [video_name, kf_frame_id, 1]
"""

import os
import glob
import numpy as np
import torch
import torch.utils.data as data_utl

from utils import video_to_tensor   # (T,1,1,C) numpy → (C,T,1,1) tensor

_SPLIT_DIR = {"training": "train", "testing": "val"}

NUM_CLASSES       = 3
FEAT_DIM          = 1024
KEYFRAME_INTERVAL = 150


class CVSDataset(data_utl.Dataset):
    """
    Parameters
    ----------
    root : str
        Path to SAGES_2024 dataset root (contains train/, val/ subdirs).
    split : str
        ``"training"`` or ``"testing"``.
    num_clips : int
        Window size.  The keyframe is always the **last** frame of the window.
        If the keyframe is too early in the video, the window is zero-padded
        at the start (causal padding).
    skip : int
        Temporal stride (kept for API parity, currently unused).
    """

    def __init__(self, root: str, split: str, num_clips: int, skip: int = 1):
        assert split in _SPLIT_DIR, f"split must be 'training' or 'testing', got {split!r}"
        feat_root      = os.path.join(root, _SPLIT_DIR[split], "features")
        self.num_clips = num_clips

        # One entry per keyframe: (video_name, sorted_pth_paths, kf_index, kf_fid, kf_label)
        self.samples = []

        for vname in sorted(os.listdir(feat_root)):
            video_dir = os.path.join(feat_root, vname)
            if not os.path.isdir(video_dir):
                continue

            pth_files = sorted(
                glob.glob(os.path.join(video_dir, "frame_*.pth")),
                key=lambda p: int(
                    os.path.splitext(os.path.basename(p))[0].split("_")[-1]
                ),
            )
            if not pth_files:
                continue

            # Load only keyframe files to retrieve labels (avoids loading all ~2700)
            for idx, fpath in enumerate(pth_files):
                fid = int(
                    os.path.splitext(os.path.basename(fpath))[0].split("_")[-1]
                )
                if fid % KEYFRAME_INTERVAL != 0:
                    continue
                payload = torch.load(fpath, map_location="cpu", weights_only=False)
                ca = payload.get("labels_ca")
                if ca is None:
                    continue
                kf_label = ca.float().numpy().astype(np.float32)
                self.samples.append((vname, pth_files, idx, fid, kf_label))

        print(
            f"CVSDataset [{split}]: {len(self.samples)} keyframe windows "
            f"from {feat_root}  (window_size={num_clips})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        vname, pth_files, kf_idx, kf_fid, kf_label = self.samples[index]

        # Causal window: num_clips frames ending at kf_idx (inclusive)
        start_idx    = max(0, kf_idx - self.num_clips + 1)
        window_files = pth_files[start_idx: kf_idx + 1]
        T_actual     = len(window_files)

        features  = np.zeros((self.num_clips, FEAT_DIM), dtype=np.float32)
        pad_start = self.num_clips - T_actual   # zero-pad at start for early kfs

        for j, fpath in enumerate(window_files):
            payload = torch.load(fpath, map_location="cpu", weights_only=False)
            features[pad_start + j] = payload["image_embeddings"].float().numpy()

        # Reshape to (T, 1, 1, C) — expected by video_to_tensor
        features = features.reshape(self.num_clips, 1, 1, FEAT_DIM)

        meta = [vname, float(kf_fid), 1]
        return features, kf_label, meta


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def cvs_collate_fn(batch):
    """
    Collate a list of (features, label, meta) into tensors.

    Returns
    -------
    features  : torch.Tensor  (B, C, T, 1, 1)
    labels    : torch.Tensor  (B, 3)
    metas     : list of [video_name, kf_frame_id, 1]
    """
    features_list, labels_list, metas = zip(*batch)
    features_tensor = torch.stack(
        [video_to_tensor(f) for f in features_list]
    )                                                      # (B, C, T, 1, 1)
    labels_tensor = torch.from_numpy(
        np.stack(labels_list)
    )                                                      # (B, 3)
    return features_tensor, labels_tensor, list(metas)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="CVSDataset smoke test")
    parser.add_argument("--cvs_root",  required=True,
                        help="Path to SAGES_2024 dataset root")
    parser.add_argument("--num_clips", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    p = parser.parse_args()

    for split in ("training", "testing"):
        print(f"\n{'='*60}")
        ds = CVSDataset(p.cvs_root, split, p.num_clips)

        # Single sample
        feats, label, meta = ds[0]
        print(f"  Sample 0  video={meta[0]}  kf_frame={int(meta[1])}")
        print(f"  features : {feats.shape}  dtype={feats.dtype}"
              f"  range=[{feats.min():.3f}, {feats.max():.3f}]")
        print(f"  label    : {label}  (CA values in [0,1])")
        pad_frames = (feats.reshape(p.num_clips, -1).sum(-1) == 0).sum()
        print(f"  zero-pad : {int(pad_frames)} / {p.num_clips} frames")

        # Check last-frame (keyframe) is non-zero
        last_frame = feats.reshape(p.num_clips, -1)[-1]
        assert last_frame.sum() != 0, "Last frame (keyframe) must not be zero-padded!"

        # Batched DataLoader
        loader = DataLoader(ds, batch_size=p.batch_size, shuffle=False,
                            collate_fn=cvs_collate_fn, num_workers=0)
        batch_feats, batch_labels, batch_metas = next(iter(loader))
        print(f"\n  Batch features : {tuple(batch_feats.shape)}")
        print(f"  Batch labels   : {tuple(batch_labels.shape)}")
        print(f"  Label range    : [{batch_labels.min():.3f}, {batch_labels.max():.3f}]")
        print(f"  Positive kfs   : {(batch_labels > 0.5).any(dim=1).sum().item()} / {p.batch_size}")

    print("\nSmoke test passed.")
