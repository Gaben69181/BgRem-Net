"""
backend/metrics.py

Loss functions and evaluation metrics for binary image segmentation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Works directly on logits (before sigmoid) for numerical stability.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model outputs
            targets: (B, 1, H, W) binary masks in {0,1}
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy with Logits + Dice loss.

    This often yields better convergence and segmentation quality
    than either BCE or Dice alone.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss for binary segmentation.

    This combines:
        - BCE with logits
        - Dice loss
        - Laplacian-based boundary loss (on sigmoid probabilities)

    The boundary term encourages sharper, better-aligned edges between
    foreground and background, which is important for background removal.

    loss = bce_dice + edge_weight * edge_loss
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        edge_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.bce_dice = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
        self.edge_weight = edge_weight

        # 3x3 Laplacian kernel for edge detection
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        # Shape: (out_channels, in_channels, kH, kW) = (1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", kernel.view(1, 1, 3, 3))

    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian edges from a (B, 1, H, W) tensor.

        We clamp the result to keep edge magnitudes bounded, which
        stabilizes training when combined with BCE + Dice.
        """
        # Ensure kernel lives on the same device and dtype as input tensor `x`
        kernel = self.laplacian_kernel.to(x.device, dtype=x.dtype)
        edges = F.conv2d(x, kernel, padding=1)
        return edges

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Base BCE + Dice loss on logits
        base_loss = self.bce_dice(logits, targets)

        # Boundary loss on Laplacian of probabilities
        probs = torch.sigmoid(logits)
        pred_edges = self._compute_edges(probs)
        gt_edges = self._compute_edges(targets)

        edge_loss = F.mse_loss(pred_edges, gt_edges)

        return base_loss + self.edge_weight * edge_loss


@torch.no_grad()
def compute_confusion(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute confusion matrix terms (TP, FP, FN, TN) for a batch.

    Args:
        logits: (B, 1, H, W) raw outputs.
        targets: (B, 1, H, W) binary masks in {0,1}.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    return tp, fp, fn, tn


@torch.no_grad()
def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Intersection over Union (IoU) for binary segmentation.
    """
    tp, fp, fn, _ = compute_confusion(logits, targets, threshold=threshold)
    intersection = tp
    union = tp + fp + fn
    return (intersection + eps) / (union + eps)


@torch.no_grad()
def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Dice score for binary segmentation.
    """
    tp, fp, fn, _ = compute_confusion(logits, targets, threshold=threshold)
    numerator = 2 * tp
    denominator = 2 * tp + fp + fn
    return (numerator + eps) / (denominator + eps)


@torch.no_grad()
def pixel_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Pixel-wise accuracy for binary segmentation.
    """
    tp, fp, fn, tn = compute_confusion(logits, targets, threshold=threshold)
    correct = tp + tn
    total = tp + tn + fp + fn
    return (correct + eps) / (total + eps)


@torch.no_grad()
def evaluate_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute IoU, Dice, and Pixel Accuracy for a batch and return as Python floats.

    Intended for logging during validation.
    """
    iou = iou_score(logits, targets, threshold=threshold).item()
    dice = dice_score(logits, targets, threshold=threshold).item()
    acc = pixel_accuracy(logits, targets, threshold=threshold).item()
    return {"iou": iou, "dice": dice, "pixel_acc": acc}


class AverageMeter:
    """
    Tracks the running average of a scalar metric (e.g. loss, IoU) over batches.

    Example:
        meter = AverageMeter()
        for loss in batch_losses:
            meter.update(loss, n=batch_size)
        print(meter.avg)
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.avg: float = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.value = float(value)
        self.sum += float(value) * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0.0


class EpochMetricTracker:
    """
    Helper to monitor training/validation metrics within a single epoch.

    This is intended to be used inside the training loop to accumulate
    running averages and optionally print progress every few batches.

    Typical usage in a training loop:

        tracker = EpochMetricTracker()
        for batch_idx, (images, masks) in enumerate(loader, start=1):
            logits = model(images)
            loss = criterion(logits, masks)

            tracker.update(loss=loss, logits=logits, targets=masks, batch_size=images.size(0))

            if batch_idx % log_interval == 0:
                stats = tracker.as_dict()
                print(
                    f"Batch {batch_idx}/{len(loader)} "
                    f"- loss: {stats['loss']:.4f} "
                    f"- IoU: {stats['iou']:.4f} "
                    f"- Dice: {stats['dice']:.4f} "
                    f"- PixelAcc: {stats['pixel_acc']:.4f}"
                )

        epoch_stats = tracker.as_dict()

    Note:
        - `update` expects raw logits and target masks in (B, 1, H, W).
        - Metrics are computed on detached tensors to avoid interfering with autograd.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.loss_meter = AverageMeter()
        self.iou_meter = AverageMeter()
        self.dice_meter = AverageMeter()
        self.pixel_acc_meter = AverageMeter()

    def update(
        self,
        loss: torch.Tensor | float,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int | None = None,
    ) -> None:
        """
        Update running metrics with a new batch.

        Args:
            loss: Scalar tensor or float for this batch.
            logits: Raw model outputs (B, 1, H, W).
            targets: Ground-truth masks (B, 1, H, W).
            batch_size: Optional batch size; if None, inferred from logits.
        """
        if batch_size is None:
            batch_size = int(logits.size(0))

        # Loss
        loss_value = float(loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else loss)
        self.loss_meter.update(loss_value, n=batch_size)

        # Detach tensors before computing metrics
        batch_metrics = evaluate_batch(
            logits.detach(),
            targets.detach(),
            threshold=self.threshold,
        )
        self.iou_meter.update(batch_metrics["iou"], n=batch_size)
        self.dice_meter.update(batch_metrics["dice"], n=batch_size)
        self.pixel_acc_meter.update(batch_metrics["pixel_acc"], n=batch_size)

    def as_dict(self) -> Dict[str, float]:
        """
        Return current running averages as a plain dict:
            {"loss": ..., "iou": ..., "dice": ..., "pixel_acc": ...}
        """
        return {
            "loss": self.loss_meter.avg,
            "iou": self.iou_meter.avg,
            "dice": self.dice_meter.avg,
            "pixel_acc": self.pixel_acc_meter.avg,
        }


if __name__ == "__main__":
    # Smoke test
    logits = torch.randn(2, 1, 16, 16)
    targets = (torch.rand(2, 1, 16, 16) > 0.5).float()

    loss_fn = BCEDiceLoss()
    loss = loss_fn(logits, targets)

    metrics = evaluate_batch(logits, targets)
    print("Loss:", float(loss))
    print("Metrics:", metrics)

    # Test training-progress helpers
    tracker = EpochMetricTracker()
    for _ in range(5):
        # Simulate repeated batches with the same logits/targets
        loss = loss_fn(logits, targets)
        tracker.update(loss=loss, logits=logits, targets=targets, batch_size=logits.size(0))
    print("Epoch stats:", tracker.as_dict())