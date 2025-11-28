"""
download_dataset.py

Download and prepare the AISegment.com Matting Human Datasets from Kaggle.

This script downloads the dataset using kagglehub and organizes it into
the expected structure for training:

    data/train/images/
    data/train/masks/
    data/val/images/
    data/val/masks/

Usage:
    python download_dataset.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import kagglehub


def download_dataset() -> str:
    """Download the AISegment.com Matting Human Datasets."""
    print("Downloading AISegment.com Matting Human Datasets from Kaggle...")
    path = kagglehub.dataset_download("laurentmih/aisegmentcom-matting-human-datasets")
    print(f"Dataset downloaded to: {path}")
    return path


def organize_dataset(download_path: str, data_root: str = "data") -> None:
    """
    Organize the downloaded dataset into train/val splits.

    The dataset structure is:
    - matting_human_half/clip_img/[video_id]/clip_[seq]/*.jpg (images)
    - matting_human_half/matting/[video_id]/matting_[seq]/*.png (masks)

    We split the video folders into train/val (80/20).
    """
    download_path = Path(download_path)
    data_root = Path(data_root)

    # Create directories
    train_images = data_root / "train" / "images"
    train_masks = data_root / "train" / "masks"
    val_images = data_root / "val" / "images"
    val_masks = data_root / "val" / "masks"

    for dir_path in [train_images, train_masks, val_images, val_masks]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("Organizing dataset...")

    # Get all video folders
    clip_img_path = download_path / "matting_human_half" / "clip_img"
    matting_path = download_path / "matting_human_half" / "matting"

    if not clip_img_path.exists() or not matting_path.exists():
        print("Dataset structure not as expected. Please check the download.")
        return

    video_folders = [f.name for f in clip_img_path.iterdir() if f.is_dir()]
    video_folders.sort()  # Ensure consistent order

    # Split into train/val (80/20)
    split_idx = int(0.8 * len(video_folders))
    train_videos = video_folders[:split_idx]
    val_videos = video_folders[split_idx:]

    print(f"Total videos: {len(video_folders)}, Train: {len(train_videos)}, Val: {len(val_videos)}")

    def copy_files(video_list, dest_images, dest_masks, src_img_base, src_mask_base):
        count_images = 0
        count_masks = 0
        for video in video_list:
            # Copy images
            img_src = src_img_base / video
            if img_src.exists():
                for clip_dir in img_src.glob("clip_*"):
                    for img_file in clip_dir.glob("*.jpg"):
                        shutil.copy(img_file, dest_images)
                        count_images += 1

            # Copy masks
            mask_src = src_mask_base / video
            if mask_src.exists():
                for matting_dir in mask_src.glob("matting_*"):
                    for mask_file in matting_dir.glob("*.png"):
                        shutil.copy(mask_file, dest_masks)
                        count_masks += 1

        return count_images, count_masks

    # Copy train files
    train_img_count, train_mask_count = copy_files(train_videos, train_images, train_masks, clip_img_path, matting_path)
    print(f"Copied {train_img_count} train images, {train_mask_count} train masks")

    # Copy val files
    val_img_count, val_mask_count = copy_files(val_videos, val_images, val_masks, clip_img_path, matting_path)
    print(f"Copied {val_img_count} val images, {val_mask_count} val masks")

    print("Dataset organization complete.")
    print(f"Train images: {len(list(train_images.glob('*')))}")
    print(f"Train masks: {len(list(train_masks.glob('*')))}")
    print(f"Val images: {len(list(val_images.glob('*')))}")
    print(f"Val masks: {len(list(val_masks.glob('*')))}")


def main() -> None:
    download_path = download_dataset()
    organize_dataset(download_path)


if __name__ == "__main__":
    main()