"""
backend/dataset.py

Dataset utilities for paired image/mask segmentation datasets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


ImageType = Image.Image
MaskType = Image.Image


@dataclass
class SegmentationSample:
    """Container for a single segmentation sample (image path + mask path)."""

    image_path: str
    mask_path: str


class SegmentationDataset(Dataset):
    """
    Generic dataset for paired image/mask segmentation.

    Expected directory structure:

        root_dir/
            images/
                xxx.jpg
                yyy.png
            masks/
                xxx.png
                yyy.png

    Image and mask filenames must match (except for extension).
    """

    def __init__(
        self,
        root_dir: str,
        img_subdir: str = "images",
        mask_subdir: str = "masks",
        img_size: int = 512,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, img_subdir)
        self.mask_dir = os.path.join(root_dir, mask_subdir)
        self.img_size = img_size
        self.augment = augment

        self.resize_img = transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR)
        self.resize_mask = transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)
        self.to_tensor = transforms.ToTensor()

        self.samples: List[SegmentationSample] = self._scan_pairs()
        if len(self.samples) == 0:
            raise RuntimeError(f"No image/mask pairs found under {self.root_dir}")

    def _scan_pairs(self) -> List[SegmentationSample]:
        image_files = [
            f
            for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ]
        image_files.sort()

        samples: List[SegmentationSample] = []
        for img_name in image_files:
            stem, _ = os.path.splitext(img_name)
            # try several mask extensions
            mask_path = None
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                candidate = os.path.join(self.mask_dir, stem + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            if mask_path is None:
                continue
            samples.append(SegmentationSample(os.path.join(self.img_dir, img_name), mask_path))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> ImageType:
        img = Image.open(path).convert("RGB")
        img = self.resize_img(img)
        return img

    def _load_mask(self, path: str) -> MaskType:
        mask = Image.open(path).convert("L")
        mask = self.resize_mask(mask)
        return mask

    def _augment(self, image: ImageType, mask: MaskType) -> Tuple[ImageType, MaskType]:
        """Apply simple geometric augmentations (same for image and mask)."""
        # Note: we use torchvision transforms that operate on PIL images
        if np.random.rand() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if np.random.rand() < 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        # small random rotation
        if np.random.rand() < 0.3:
            angle = float(np.random.uniform(-10, 10))
            image = transforms.functional.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = transforms.functional.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = self._load_image(sample.image_path)
        mask = self._load_mask(sample.mask_path)

        if self.augment:
            image, mask = self._augment(image, mask)

        image_tensor = self.to_tensor(image)  # [0,1] float32

        # Convert mask to {0,1} float tensor
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        mask_np = (mask_np >= 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (1, H, W)

        return image_tensor, mask_tensor


def create_dataloaders(
    data_root: str = "data",
    img_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to create train/val dataloaders.

    Expects:

        data_root/train/images
        data_root/train/masks
        data_root/val/images
        data_root/val/masks
    """

    train_dataset = SegmentationDataset(
        root_dir=os.path.join(data_root, "train"),
        img_size=img_size,
        augment=True,
    )
    val_dataset = SegmentationDataset(
        root_dir=os.path.join(data_root, "val"),
        img_size=img_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick smoke test (will fail if dataset folders are empty)
    try:
        train_loader, val_loader = create_dataloaders(
            data_root="data",
            batch_size=2,
            img_size=512,
            num_workers=0,
        )
        images, masks = next(iter(train_loader))
        print("Images shape:", images.shape)
        print("Masks shape :", masks.shape)
    except Exception as e:
        print("Dataset smoke test failed:", e)