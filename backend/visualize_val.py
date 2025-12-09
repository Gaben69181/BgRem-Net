"""backend/visualize_val.py

Generate validation sample grid:
[Image | GT Mask | Pred Mask (soft) | Foreground]
using a trained checkpoint, without running training.
"""

from __future__ import annotations

import argparse
import os

import torch

from dataset import create_dataloaders
from model import build_u2net
from train import plot_sample_predictions, prepare_device


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Make sure training has produced a best_model.pth."
        )

    model = build_u2net(n_channels=3, n_classes=1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize validation samples with foreground using an existing checkpoint "
            "without running training."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data_p3m",
        help="Root folder containing train/ and val/ subfolders (e.g. data or data_p3m).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pth",
        help="Path to a trained checkpoint (.pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Input image size (square) used for the dataloader.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of validation samples to visualize (from the start of the dataset).",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="outputs/plots/val_samples.png",
        help="Where to save the output grid image.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = prepare_device(args.device)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Build dataloaders using the same helper as training
    _, val_loader = create_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=max(args.num_samples, 1),
        num_workers=args.num_workers,
    )

    # Load trained model
    model = load_model(args.checkpoint, device=device)

    # Reuse the training-time visualization helper, which already plots:
    # [Image | GT Mask | Pred Mask (soft) | Foreground]
    plot_sample_predictions(
        model=model,
        loader=val_loader,
        device=device,
        out_path=args.out_path,
        num_samples=args.num_samples,
    )

    print(f"Saved validation visualization to {args.out_path}")


if __name__ == "__main__":
    main()