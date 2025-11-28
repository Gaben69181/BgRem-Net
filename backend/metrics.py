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


if __name__ == "__main__":
    # Smoke test
    logits = torch.randn(2, 1, 16, 16)
    targets = (torch.rand(2, 1, 16, 16) > 0.5).float()

    loss_fn = BCEDiceLoss()
    loss = loss_fn(logits, targets)

    metrics = evaluate_batch(logits, targets)
    print("Loss:", float(loss))
    print("Metrics:", metrics)