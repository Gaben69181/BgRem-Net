"""backend/train.py

Training script for U-Net foreground/background segmentation.
Features:
- MLflow Experiment Tracking (SQLite backend)
- TQDM Progress Bars
- Automatic Mixed Precision (AMP) for faster training
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm  # Progress bar
from torch.cuda.amp import GradScaler, autocast # Mixed Precision

import mlflow

# Handle imports whether run as script or module
if __name__ == "__main__" and __package__ is None:
    from model import build_unet
    from dataset import create_dataloaders
    from metrics import BCEDiceLoss, evaluate_batch
else:
    from .model import build_unet
    from .dataset import create_dataloaders
    from .metrics import BCEDiceLoss, evaluate_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net for background removal segmentation.")
    parser.add_argument("--data-root", type=str, default="data", help="Root folder containing train/ and val/ subfolders.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size.")
    parser.add_argument("--img-size", type=int, default=512, help="Input image size (square).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker processes.")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="Where to save model checkpoints.")
    parser.add_argument("--pretrained-checkpoint", type=str, default="", help="Optional path to a pretrained checkpoint to fine-tune from.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'.")
    parser.add_argument("--timelapse", action="store_true", help="If set, save per-epoch prediction frames for timelapse.")
    parser.add_argument("--timelapse-samples", type=int, default=4, help="Number of validation images to track for timelapse.")
    parser.add_argument("--timelapse-dir", type=str, default="outputs/timelapse", help="Directory to store timelapse frames.")
    return parser.parse_args()


def prepare_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_iou: float,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_iou": best_val_iou,
        },
        path,
    )


def load_pretrained_if_available(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading pretrained checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict, strict=False)
    elif checkpoint_path:
        print(f"WARNING: Pretrained checkpoint {checkpoint_path} not found. Training from scratch.")


def infer_batch(model: nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        logits = model(images)
    return logits.cpu()


def maybe_init_timelapse_samples(
    val_loader: DataLoader,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0 for timelapse.")

    images, masks = next(iter(val_loader))
    if images.size(0) > num_samples:
        images = images[:num_samples]
        masks = masks[:num_samples]
    return images, masks


def save_timelapse_frames(
    epoch: int,
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    images_np = images.clone()
    gt_np = gt_masks.clone()
    pred_np = pred_masks.clone()

    images_np = torch.clamp(images_np, 0.0, 1.0)
    gt_np = torch.clamp(gt_np, 0.0, 1.0)
    pred_np = torch.clamp(pred_np, 0.0, 1.0)

    b = images_np.size(0)
    for i in range(b):
        sample_dir = os.path.join(out_dir, f"sample_{i:02d}")
        os.makedirs(sample_dir, exist_ok=True)

        gt_3ch = gt_np[i].repeat(3, 1, 1)
        pred_3ch = pred_np[i].repeat(3, 1, 1)
        triplet = torch.cat([images_np[i], gt_3ch, pred_3ch], dim=2)

        frame_path = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
        save_image(triplet, frame_path)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    # Initialize Scaler for Mixed Precision
    scaler = GradScaler()
    
    # TQDM Progress Bar for Training
    pbar = tqdm(loader, desc="Training", leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        # Cast operations to mixed precision
        with autocast():
            logits = model(images)
            loss = criterion(logits, masks)
        
        # Scale loss and backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    metric_sums = {"iou": 0.0, "dice": 0.0, "pixel_acc": 0.0}
    total_samples = 0

    # TQDM Progress Bar for Validation
    pbar = tqdm(loader, desc="Validating", leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        batch_metrics = evaluate_batch(logits, masks)
        for k in metric_sums:
            metric_sums[k] += batch_metrics[k] * batch_size

    avg_loss = running_loss / max(total_samples, 1)
    avg_metrics = {k: v / max(total_samples, 1) for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


def main() -> None:
    args = parse_args()
    device = prepare_device(args.device)

    print(f"Using device: {device}")
    
    # --- MLFLOW SETUP ---
    # Fix for Future Warning: Use local SQLite database instead of file system
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("UNet-Background-Removal")
    
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        train_loader, val_loader = create_dataloaders(
            data_root=args.data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        model = build_unet(n_channels=3, n_classes=1)
        model.to(device)

        load_pretrained_if_available(model, args.pretrained_checkpoint, device=device)

        criterion = BCEDiceLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_val_iou = 0.0
        fixed_images: torch.Tensor | None = None
        fixed_masks: torch.Tensor | None = None

        if args.timelapse:
            fixed_images, fixed_masks = maybe_init_timelapse_samples(val_loader, args.timelapse_samples)

        print(f"Starting training for {args.epochs} epochs...")
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

            elapsed = time.time() - start_time
            msg = (
                f"Epoch {epoch:03d}/{args.epochs:03d} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val IoU: {val_metrics['iou']:.4f} "
                f"| Time: {elapsed:.1f}s"
            )
            print(msg)

            # MLflow Logging
            metrics_to_log = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_metrics["iou"],
                "val_dice": val_metrics["dice"],
                "val_pixel_acc": val_metrics["pixel_acc"],
            }
            mlflow.log_metrics(metrics_to_log, step=epoch)

            # Save best model
            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                save_checkpoint(model, optimizer, epoch, best_val_iou, ckpt_path)
                print(f"  >>> New Best IoU! Saved: {ckpt_path}")
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

            # Timelapse
            if args.timelapse and fixed_images is not None and fixed_masks is not None:
                logits = infer_batch(model, fixed_images, device)
                probs = torch.sigmoid(logits)
                pred_masks = (probs >= 0.5).float()
                save_timelapse_frames(
                    epoch=epoch,
                    images=fixed_images,
                    gt_masks=fixed_masks,
                    pred_masks=pred_masks,
                    out_dir=args.timelapse_dir,
                )
        
        # Log timelapse outputs if they exist
        if args.timelapse and os.path.exists(args.timelapse_dir):
            print("Uploading timelapse frames to MLflow...")
            mlflow.log_artifacts(args.timelapse_dir, artifact_path="timelapse_frames")

    print("Training complete.")


if __name__ == "__main__":
    main()