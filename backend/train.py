"""backend/train.py

Training script for U-Net foreground/background segmentation.

This module can be run as a script:

    python backend/train.py --data-root data --epochs 20

It trains a U-Net model on a paired image/mask dataset and reports
IoU, Dice, and pixel accuracy. Optionally, it can export per-epoch
prediction frames for later timelapse visualization.
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
import matplotlib.pyplot as plt

from model import build_unet, build_u2net, build_u2net_lite
from dataset import create_dataloaders
from metrics import BCEDiceLoss, EdgeAwareLoss, evaluate_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net for background removal segmentation.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data_p3m",
        help="Root folder containing train/ and val/ subfolders (e.g. data_p3m).",
    )
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
    """
    Load weights from an existing checkpoint for fine-tuning.

    This allows using pretrained weights while still performing
    further training on the target dataset.
    """
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading pretrained checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict, strict=False)
    elif checkpoint_path:
        print(f"WARNING: Pretrained checkpoint {checkpoint_path} not found. Training from scratch.")


def infer_batch(model: nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run a forward pass and return logits on CPU."""
    model.eval()
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        logits = model(images)
    return logits.cpu()


def maybe_init_timelapse_samples(
    val_loader: DataLoader,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Grab a fixed mini-batch from the validation loader for timelapse visualization.

    Returns:
        (images, masks) on CPU with shape (N, C, H, W) and (N, 1, H, W).
    """
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
    """
    Save side-by-side image/ground-truth/predicted mask triplets for timelapse.

    Frames are stored as:
        {out_dir}/sample_{i}/epoch_{epoch:03d}.png
    """
    os.makedirs(out_dir, exist_ok=True)

    images_np = images.clone()
    gt_np = gt_masks.clone()
    pred_np = pred_masks.clone()

    # Ensure in [0,1]
    images_np = torch.clamp(images_np, 0.0, 1.0)
    gt_np = torch.clamp(gt_np, 0.0, 1.0)
    pred_np = torch.clamp(pred_np, 0.0, 1.0)

    b = images_np.size(0)
    for i in range(b):
        sample_dir = os.path.join(out_dir, f"sample_{i:02d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Convert single-channel masks to 3-channel for visualization
        gt_3ch = gt_np[i].repeat(3, 1, 1)
        pred_3ch = pred_np[i].repeat(3, 1, 1)

        # Concatenate horizontally: [image | gt | pred]
        triplet = torch.cat([images_np[i], gt_3ch, pred_3ch], dim=2)

        frame_path = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
        save_image(triplet, frame_path)


def plot_training_curves(
    history: Dict[str, list[float]],
    out_dir: str = "outputs/plots",
    filename: str = "training_curves.png",
) -> None:
    """
    Plot training/validation loss and validation metrics over epochs.

    Saves a PNG file containing:
        - train_loss vs. val_loss
        - val IoU, Dice, and Pixel Accuracy
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    if not epochs:
        print("No history to plot; skipping training curve plotting.")
        return

    plt.figure(figsize=(12, 8))

    # Loss curves
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Metric curves
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history["val_iou"], label="Val IoU")
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.plot(epochs, history["val_pixel_acc"], label="Val PixelAcc")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved training curves to {out_path}")


@torch.no_grad()
def plot_sample_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
    num_samples: int = 4,
) -> None:
    """
    Plot a few sample predictions from the validation set.

    For each sample, show:
        - input image
        - ground-truth mask
        - predicted soft mask (sigmoid output)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.eval()

    try:
        images, masks = next(iter(loader))
    except StopIteration:
        print("Validation loader is empty; skipping sample prediction plotting.")
        return

    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)

    logits = model(images)
    probs = torch.sigmoid(logits)

    b = min(num_samples, images.size(0))
    images = images[:b].cpu()
    masks = masks[:b].cpu()
    probs = probs[:b].cpu()

    fig, axes = plt.subplots(b, 4, figsize=(12, 3 * b))
    if b == 1:
        axes = axes[None, :]  # ensure 2D indexing

    for i in range(b):
        img = images[i].permute(1, 2, 0).clamp(0.0, 1.0).numpy()
        gt = masks[i, 0].numpy()
        pred = probs[i, 0].numpy()
        foreground = img * pred[..., None]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("GT Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Pred Mask (soft)")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(foreground)
        axes[i, 3].set_title("Foreground")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved sample predictions to {out_path}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

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

    for images, masks in loader:
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
    print(f"Data root  : {args.data_root}")

    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_u2net(n_channels=3, n_classes=1)
    model.to(device)
 
    load_pretrained_if_available(model, args.pretrained_checkpoint, device=device)
 
    # Use edge-aware loss: BCE + Dice + Laplacian boundary loss
    # This sharpens object boundaries, which is important for background removal.
    criterion = EdgeAwareLoss(bce_weight=0.5, dice_weight=0.5, edge_weight=0.3)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # For plotting training curves
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "val_pixel_acc": [],
    }

    best_val_iou = 0.0
    fixed_images: torch.Tensor | None = None
    fixed_masks: torch.Tensor | None = None

    if args.timelapse:
        fixed_images, fixed_masks = maybe_init_timelapse_samples(val_loader, args.timelapse_samples)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        # Store history for plotting
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_pixel_acc"].append(val_metrics["pixel_acc"])

        elapsed = time.time() - start_time
        msg = (
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- IoU: {val_metrics['iou']:.4f} "
            f"- Dice: {val_metrics['dice']:.4f} "
            f"- PixelAcc: {val_metrics['pixel_acc']:.4f} "
            f"- time: {elapsed:.1f}s"
        )
        print(msg)

        # Save best checkpoint based on IoU
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, best_val_iou, ckpt_path)
            print(f"  - New best IoU: {best_val_iou:.4f} (checkpoint saved to {ckpt_path})")

        # Optionally export timelapse frames
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

    # After training, generate plotting artifacts
    plots_dir = os.path.join("outputs", "plots")
    plot_training_curves(history, out_dir=plots_dir, filename="training_curves.png")

    sample_out_path = os.path.join(plots_dir, "val_samples.png")
    plot_sample_predictions(
        model,
        val_loader,
        device,
        out_path=sample_out_path,
        num_samples=4,
    )

    print("Training complete.")


if __name__ == "__main__":
    main()