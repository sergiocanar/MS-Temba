"""
CVSTemba_main.py

Training and evaluation script for CVSTemba — MS-Temba adapted to the CVS
(Critical View of Safety) surgical dataset with EVA02 features.

Design:
  - One sample = a causal window of num_clips frames ending at a keyframe.
  - Label = CA soft label (3,) of the keyframe (last frame in the window).
  - Loss and evaluation are computed on the last-frame prediction only.
    The rest of the window provides temporal context to the SSM.
  - BCE loss uses pos_weight=[6.80, 3.04, 5.44] for class imbalance.
  - Metrics logged to Weights & Biases (wandb).
"""

import time
import argparse
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import logging
from torch.autograd import Variable

import wandb
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import ModelEma

from torchmetrics.classification import MultilabelAveragePrecision
import models_cvs_temba          # registers 'cvstemba' with timm
from cvs_dataloader import CVSDataset, cvs_collate_fn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES    = 3
CVS_NAMES  = ["c1", "c2", "c3"]
POS_WEIGHT = torch.tensor([6.80, 3.04, 5.44])

# ---------------------------------------------------------------------------
# Argument parser
# NOTE: all args use -- and underscores so wandb sweep ${args} substitution works
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--mode",       type=str,   default="rgb")
parser.add_argument("--train",      type=str,   default="True")
parser.add_argument("--gpu",        type=str,   default="0")
parser.add_argument("--epochs",     type=int,   default=50)
parser.add_argument("--batch_size", type=int,   default=8)
parser.add_argument("--num_clips",  type=int,   default=30,
                    help="Window size in frames; keyframe is always the last frame")
parser.add_argument("--skip",       type=int,   default=1)
parser.add_argument("--alpha_l",    type=float, default=1.0)
parser.add_argument("--beta_l",     type=float, default=0.05)
parser.add_argument("--output_dir", type=str,   default="./output_cvs")
parser.add_argument("--cvs_root",   type=str,   required=True,
                    help="Path to SAGES_2024 dataset root (contains train/, val/)")
parser.add_argument("--comp_info",  type=str,   default="False")
parser.add_argument("--load_model", type=str,   default="False")

# Model
parser.add_argument("--model",          default="cvstemba", type=str)
parser.add_argument("--drop",           type=float, default=0.0)
parser.add_argument("--drop_path",      type=float, default=0.0,
                    dest="drop_path")
parser.add_argument("--model_ema",      action="store_true")
parser.add_argument("--no_model_ema",   action="store_false", dest="model_ema")
parser.set_defaults(model_ema=True)
parser.add_argument("--model_ema_decay",     type=float, default=0.99996)
parser.add_argument("--model_ema_force_cpu", action="store_true", default=False)

# Optimiser  (timm uses args.opt, args.weight_decay, etc.)
parser.add_argument("--opt",          default="adamw", type=str)
parser.add_argument("--opt_eps",      default=1e-8,   type=float)
parser.add_argument("--opt_betas",    default=None,   type=float, nargs="+")
parser.add_argument("--clip_grad",    type=float,     default=None)
parser.add_argument("--momentum",     type=float,     default=0.9)
parser.add_argument("--weight_decay", type=float,     default=0.01)

# LR schedule  (timm uses args.sched, args.warmup_epochs, args.min_lr, etc.)
parser.add_argument("--sched",           default="cosine", type=str)
parser.add_argument("--lr",              type=float, default=5e-4)
parser.add_argument("--lr_noise",        type=float, nargs="+", default=None)
parser.add_argument("--lr_noise_pct",    type=float, default=0.67)
parser.add_argument("--lr_noise_std",    type=float, default=1.0)
parser.add_argument("--warmup_lr",       type=float, default=1e-6)
parser.add_argument("--min_lr",          type=float, default=1e-5)
parser.add_argument("--decay_epochs",    type=float, default=30)
parser.add_argument("--warmup_epochs",   type=int,   default=5)
parser.add_argument("--cooldown_epochs", type=int,   default=10)
parser.add_argument("--patience_epochs", type=int,   default=10)
parser.add_argument("--decay_rate",      type=float, default=0.1)

# Wandb
parser.add_argument("--wandb_project", type=str, default="cvstemba")
parser.add_argument("--wandb_entity",  type=str, default=None)
parser.add_argument("--wandb_name",    type=str, default=None)
parser.add_argument("--no_wandb",      action="store_true", default=False)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    train_ds = CVSDataset(args.cvs_root, "training", args.num_clips, args.skip)
    val_ds   = CVSDataset(args.cvs_root, "testing",  args.num_clips, args.skip)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=cvs_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=cvs_collate_fn,
    )
    return {"train": train_loader, "val": val_loader}

# ---------------------------------------------------------------------------
# Network forward pass
# ---------------------------------------------------------------------------

def run_network(model, data, gpu, epoch=0):
    """
    data: (features (B,C,T,1,1),  labels (B,3),  metas)
    Supervision is on the last-frame prediction only (the keyframe).
    """
    inputs, labels, _ = data

    inputs = Variable(inputs.cuda(gpu))
    labels = Variable(labels.cuda(gpu))

    # (B, C, T, 1, 1) → (B, C, T)
    inputs = inputs.squeeze(3).squeeze(3)

    outputs_final, block_outputs, diversity_loss = model(inputs)
    pred_final = outputs_final[:, -1, :]   # (B, 3) — last frame = keyframe

    pw = POS_WEIGHT.to(pred_final.device)

    loss_f = F.binary_cross_entropy_with_logits(
        pred_final, labels, pos_weight=pw
    )

    block_losses = []
    block_preds  = []
    for block_out in block_outputs:
        bp = torch.sigmoid(block_out[:, -1, :])
        bl = F.binary_cross_entropy_with_logits(
            block_out[:, -1, :], labels, pos_weight=pw
        )
        block_losses.append(bl)
        block_preds.append(bp)

    block_loss_weight     = 0.3
    diversity_loss_weight = 100.0

    total_block_loss = sum(block_losses) * block_loss_weight
    loss = args.alpha_l * (loss_f + total_block_loss) + diversity_loss_weight * diversity_loss

    probs_final = torch.sigmoid(pred_final)   # (B, 3)
    return probs_final, loss, diversity_loss, block_preds, block_losses

# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _make_map_metric():
    return MultilabelAveragePrecision(num_labels=CLASSES, average=None)


def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss     = 0.0
    tot_div_loss = 0.0
    num_iter     = 0.0
    metric       = _make_map_metric()
    block_metrics = [_make_map_metric() for _ in range(3)]

    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        probs, loss, div_loss, block_preds, _ = run_network(model, data, gpu, epoch)

        labels_bin = (data[1] > 0.5).int()   # CA soft → binary for AP metric
        metric.update(probs.detach().cpu(), labels_bin)
        for i, bp in enumerate(block_preds):
            block_metrics[i].update(bp.detach().cpu(), labels_bin)

        tot_loss     += loss.item()
        tot_div_loss += div_loss.item()
        loss.backward()
        optimizer.step()

    train_map = float(100 * metric.compute().mean())
    logging.info(f"Epoch {epoch}  train-mAP: {train_map:.4f}")

    block_maps = []
    for i, bm in enumerate(block_metrics):
        bmap = float(100 * bm.compute().mean())
        block_maps.append(bmap)
        logging.info(f"Epoch {epoch}  Block {i+1} train-mAP: {bmap:.4f}")

    avg_div_loss = tot_div_loss / num_iter
    logging.info(f"Epoch {epoch}  diversity loss: {avg_div_loss:.6f}")

    return train_map, tot_loss / num_iter, block_maps, avg_div_loss

# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    metric        = _make_map_metric()
    block_metrics = [_make_map_metric() for _ in range(3)]

    tot_loss = 0.0
    num_iter = 0.0
    full_probs       = {}
    block_full_probs = [{} for _ in range(3)]

    for i in range(3):
        os.makedirs(os.path.join(args.output_dir, f"block_{i+1}"), exist_ok=True)

    with torch.no_grad():
        for data in dataloader:
            num_iter += 1
            metas = data[2]

            probs, loss, _, block_preds, _ = run_network(model, data, gpu, epoch)

            labels_bin = (data[1] > 0.5).int()   # CA soft → binary for AP metric
            metric.update(probs.cpu(), labels_bin)
            for i, bp in enumerate(block_preds):
                block_metrics[i].update(bp.cpu(), labels_bin)

            tot_loss += loss.item()

            for b in range(probs.shape[0]):
                key = f"{metas[b][0]}_f{int(metas[b][1])}"
                full_probs[key] = probs[b].cpu().numpy()
                for i, bp in enumerate(block_preds):
                    block_full_probs[i][key] = bp[b].cpu().numpy()

    ap_per_class = 100 * metric.compute()          # (3,) per-class AP
    val_map      = float(ap_per_class.mean())

    for ci, cname in enumerate(CVS_NAMES):
        logging.info(f"Epoch {epoch}  AP/{cname}: {float(ap_per_class[ci]):.4f}")

    block_val_maps = []
    for i in range(3):
        bap  = 100 * block_metrics[i].compute()
        bmap = float(bap.mean())
        block_val_maps.append(bmap)
        logging.info(f"Epoch {epoch}  Block {i+1} val-mAP: {bmap:.4f}")
        block_dir = os.path.join(args.output_dir, f"block_{i+1}")
        pickle.dump(block_full_probs[i],
                    open(os.path.join(block_dir, f"{epoch}.pkl"), "wb"),
                    pickle.HIGHEST_PROTOCOL)

    logging.info(f"Epoch {epoch}  val-mAP: {val_map:.4f}")

    return full_probs, tot_loss / num_iter, val_map, block_val_maps, ap_per_class

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run(model, gpu, dataloaders, optimizer, sched, num_epochs=50):
    since        = time.time()
    Best_val_map = 0.0
    Best_block_val_maps = [0.0, 0.0, 0.0]

    for epoch in range(num_epochs):
        since1 = time.time()
        logging.info(f"Epoch {epoch}/{num_epochs - 1}")
        logging.info("-" * 10)

        train_map, train_loss, block_train_maps, avg_div_loss = \
            train_step(model, gpu, optimizer, dataloaders["train"], epoch)

        prob_val, val_loss, val_map, block_val_maps, ap_per_class = \
            val_step(model, gpu, dataloaders["val"], epoch)

        sched.step(val_loss)
        epoch_time = time.time() - since1

        # ---- wandb logging ----
        log_dict = {
            "epoch":          epoch,
            "train/loss":     train_loss,
            "train/mAP":      train_map,
            "val/loss":       val_loss,
            "val/mAP":        val_map,
            "train/div_loss": avg_div_loss,
            "lr":             optimizer.param_groups[0]["lr"],
            "time/epoch_s":   epoch_time,
        }
        for ci, cname in enumerate(CVS_NAMES):
            log_dict[f"val/AP_{cname}"] = float(ap_per_class[ci])
        for i in range(3):
            log_dict[f"block{i+1}/train_mAP"] = block_train_maps[i]
            log_dict[f"block{i+1}/val_mAP"]   = block_val_maps[i]

        wandb.log(log_dict, step=epoch)

        logging.info(
            f"Epoch {epoch}  epoch_time: {epoch_time:.1f}s  "
            f"total_time: {time.time() - since:.1f}s"
        )

        # ---- save best model ----
        if Best_val_map < val_map:
            Best_val_map = val_map
            logging.info(f"Epoch {epoch}  *** New best val-mAP: {Best_val_map:.4f}")
            pickle.dump(prob_val,
                        open(os.path.join(args.output_dir, f"{epoch}.pkl"), "wb"),
                        pickle.HIGHEST_PROTOCOL)
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_model.pth"))
            wandb.summary["best_val_mAP"]   = Best_val_map
            wandb.summary["best_val_epoch"] = epoch

        for i in range(3):
            if Best_block_val_maps[i] < block_val_maps[i]:
                Best_block_val_maps[i] = block_val_maps[i]
                block_dir = os.path.join(args.output_dir, f"block_{i+1}")
                torch.save(model.state_dict(),
                           os.path.join(block_dir, "best_model.pth"))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str):
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- wandb init (must happen before output_dir is used, so sweep can set it) ----
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )
        # Sweep agents inject overrides via wandb.config — apply them to args
        for key, val in wandb.config.items():
            if hasattr(args, key) and key not in ("wandb_project", "wandb_entity",
                                                   "wandb_name", "no_wandb"):
                setattr(args, key, type(getattr(args, key))(val))
        # Each sweep run gets its own subdirectory
        args.output_dir = os.path.join(args.output_dir, wandb.run.id)
    else:
        wandb.init(mode="disabled")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    logging.info(f"Arguments: {args}")

    dataloaders = load_data()

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=CLASSES,
        in_feat_dim=1024,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    model.cuda(int(args.gpu))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model: {args.model}  |  Parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    optimizer    = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.model_ema:
        ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    run(model, int(args.gpu), dataloaders, optimizer, lr_scheduler,
        num_epochs=args.epochs)

    wandb.finish()
