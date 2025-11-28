"""backend/inference.py

Inference utilities for background removal segmentation.

Provides functions to:
- load a trained U-Net checkpoint
- run segmentation on arbitrary-resolution images
- generate foreground mask
- composite foreground over transparent or solid background
- optionally upscale mask back to original resolution
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from .model import build_unet


DEFAULT_IMG_SIZE = 512


@dataclass
class InferenceConfig:
    checkpoint_path: str = "outputs/checkpoints/best_model.pth"
    device: str = "cuda"
    img_size: int = DEFAULT_IMG_SIZE
    threshold: float = 0.5


def prepare_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(config: InferenceConfig) -> torch.nn.Module:
    """
    Load a U-Net model and weights from checkpoint.

    Returns a model in eval() mode, moved to the requested device.
    """
    device = prepare_device(config.device)
    model = build_unet(n_channels=3, n_classes=1)
    model.to(device)

    if not os.path.isfile(config.checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {config.checkpoint_path}. "
            "Train the model first with backend/train.py."
        )

    ckpt = torch.load(config.checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _preprocess_image(image: Image.Image, img_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Resize and normalize an arbitrary-resolution PIL image.

    Returns:
        tensor: (1, 3, H, W) float32 in [0,1]
        orig_size: (H_orig, W_orig)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_size = image.size[1], image.size[0]  # (H, W)

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    tensor = transform(image).unsqueeze(0)
    return tensor, orig_size


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image: Image.Image,
    img_size: int = DEFAULT_IMG_SIZE,
    threshold: float = 0.5,
) -> Tuple[Image.Image, Image.Image]:
    """
    Run segmentation on an input image and return:
        - binary mask resized to original resolution (P mode or L)
        - soft mask (grayscale, 0-255) resized to original resolution
    """
    device = next(model.parameters()).device

    tensor, orig_hw = _preprocess_image(image, img_size=img_size)
    tensor = tensor.to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits)[0, 0]  # (H, W) in [0,1]

    # Upsample to original resolution
    h_orig, w_orig = orig_hw
    probs = torch.nn.functional.interpolate(
        probs.unsqueeze(0).unsqueeze(0),
        size=(h_orig, w_orig),
        mode="bilinear",
        align_corners=True,
    )[0, 0]

    probs_np = probs.cpu().numpy().astype(np.float32)

    # Soft mask as grayscale image 0-255
    soft_mask_arr = (probs_np * 255.0).clip(0, 255).astype(np.uint8)
    soft_mask_img = Image.fromarray(soft_mask_arr, mode="L")

    # Binary mask
    bin_mask_arr = (probs_np >= threshold).astype(np.uint8) * 255
    bin_mask_img = Image.fromarray(bin_mask_arr, mode="L")

    return bin_mask_img, soft_mask_img


def compose_foreground(
    image: Image.Image,
    mask: Image.Image,
    background_color: Optional[Tuple[int, int, int]] = None,
) -> Image.Image:
    """
    Composite the foreground of `image` with transparency or a solid background.

    Args:
        image: Input RGB/RGBA image.
        mask: Binary or soft mask (mode "L"), same size as image.
        background_color: If None, return RGBA with transparent background.
            If a 3-tuple, composite over a solid background of this color.

    Returns:
        PIL Image in RGBA mode.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")

    image = image.resize(mask.size, resample=Image.BILINEAR)

    # Use mask as alpha channel
    rgba = image.copy()
    rgba.putalpha(mask)

    if background_color is None:
        return rgba

    bg = Image.new("RGBA", rgba.size, color=(*background_color, 255))
    out = Image.alpha_composite(bg, rgba)
    return out


def run_inference_on_path(
    model: torch.nn.Module,
    image_path: str,
    img_size: int = DEFAULT_IMG_SIZE,
    threshold: float = 0.5,
    background_color: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Convenience function for CLI / Streamlit use.

    Args:
        model: Loaded U-Net model.
        image_path: Path to input image.

    Returns:
        bin_mask: Binary mask (L).
        soft_mask: Soft mask (L).
        composed: RGBA image with transparent or solid background.
    """
    image = Image.open(image_path)
    bin_mask, soft_mask = predict_mask(model, image, img_size=img_size, threshold=threshold)
    composed = compose_foreground(image, bin_mask, background_color=background_color)
    return bin_mask, soft_mask, composed


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="Run background removal on a single image.")
    parser.add_argument("image_path", type=str, help="Path to input image.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--bg-color",
        type=str,
        default="",
        help="Optional background color as R,G,B (e.g. 255,255,255)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/images",
        help="Directory to save results.",
    )
    args = parser.parse_args()

    if args.bg_color:
        try:
            r, g, b = [int(c) for c in args.bg_color.split(",")]
            bg_color: Optional[Tuple[int, int, int]] = (r, g, b)
        except Exception as exc:  # noqa: F841
            raise ValueError("Invalid --bg-color format. Use R,G,B (e.g. 0,255,0)")
    else:
        bg_color = None

    cfg = InferenceConfig(
        checkpoint_path=args.checkpoint,
        device=args.device,
        img_size=args.img_size,
        threshold=args.threshold,
    )
    model = load_model(cfg)

    os.makedirs(args.out_dir, exist_ok=True)

    bin_mask, soft_mask, composed = run_inference_on_path(
        model,
        args.image_path,
        img_size=args.img_size,
        threshold=args.threshold,
        background_color=bg_color,
    )

    stem = os.path.splitext(os.path.basename(args.image_path))[0]
    bin_path = os.path.join(args.out_dir, f"{stem}_mask_binary.png")
    soft_path = os.path.join(args.out_dir, f"{stem}_mask_soft.png")
    comp_path = os.path.join(args.out_dir, f"{stem}_foreground.png")

    bin_mask.save(bin_path)
    soft_mask.save(soft_path)
    composed.save(comp_path)

    print("Saved:")
    print("  Binary mask :", bin_path)
    print("  Soft mask   :", soft_path)
    print("  Composited  :", comp_path)