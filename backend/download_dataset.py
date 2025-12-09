"""
backend/download_dataset.py

Download and prepare the P3M-10k dataset from Kaggle for training the U-Net
background removal model.

This script expects the Kaggle dataset:

    rahulbhalley/p3m-10k

which mirrors the official P3M-10k structure:

    P3M-10k/
        train/
            alpha/
            merged/
            ...
        validation/
            alpha/
            merged/
            ...

We reorganize it into the structure expected by SegmentationDataset:

    <data_root>/
        train/
            images/
            masks/
        val/
            images/
            masks/

Usage (from project root):

    python backend/download_dataset.py \
        --data-root data_p3m \
        --max-train 2000 \
        --max-val 400
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import kagglehub


@dataclass
class SplitConfig:
    name: str  # "train" or "validation"
    max_samples: int | None  # None = no limit


def download_p3m10k() -> Path:
    """Download the P3M-10k dataset via kagglehub and return the root path."""
    print("Downloading P3M-10k dataset from Kaggle (rahulbhalley/p3m-10k)...")
    path_str = kagglehub.dataset_download("rahulbhalley/p3m-10k")
    root = Path(path_str) / "P3M-10k"
    if not root.exists():
        raise FileNotFoundError(
            f"Expected 'P3M-10k' directory under {path_str}, but it was not found."
        )
    print(f"P3M-10k root directory: {root}")
    return root


def _collect_pairs_from_split(
    split_dir: Path,
    max_samples: int | None = None,
) -> List[Tuple[Path, Path]]:
    """
    Collect (image_path, mask_path) pairs from a P3M-10k split directory.

    P3M-10k (mirror on Kaggle) uses two slightly different layouts:

    1) For the main train split:

        split_dir/
            blurred_image/   - RGB images
            mask/            - alpha mattes

    2) For the validation split:

        split_dir/
            P3M-500-NP/
                blurred_image/
                mask/
            P3M-500-P/
                blurred_image/
                mask/

    This helper handles both cases. Pairing is based on filename stem
    (without extension).
    """

    def collect_from_pair_dirs(img_dir: Path, msk_dir: Path) -> List[Tuple[Path, Path]]:
        """Collect (image, mask) pairs from a single (blurred_image, mask) pair."""
        if not img_dir.exists() or not msk_dir.exists():
            return []

        image_files = sorted(
            [
                p
                for p in img_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            ]
        )

        pairs_local: List[Tuple[Path, Path]] = []
        missing_local = 0

        for img_path in image_files:
            stem = img_path.stem
            candidate = msk_dir / f"{stem}.png"
            if not candidate.exists():
                missing_local += 1
                continue
            pairs_local.append((img_path, candidate))

        if missing_local > 0:
            print(
                f"[WARN] {missing_local} images in {img_dir} had no matching alpha PNG "
                f"in {msk_dir} and were skipped."
            )

        return pairs_local

    pairs: List[Tuple[Path, Path]] = []

    # Case 1: split_dir itself has blurred_image/ and mask/ (train)
    merged_dir = split_dir / "blurred_image"
    alpha_dir = split_dir / "mask"

    if merged_dir.exists() and alpha_dir.exists():
        pairs.extend(collect_from_pair_dirs(merged_dir, alpha_dir))
    else:
        # Case 2: split_dir has subfolders (e.g. P3M-500-NP, P3M-500-P) each
        # containing blurred_image/ and mask/.
        subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not subdirs:
            raise RuntimeError(
                "P3M-10k directory structure not as expected.\n"
                f"Expected either 'blurred_image'+'mask' or subfolders under: {split_dir}\n"
                f"Found: {[p.name for p in split_dir.iterdir()]}"
            )

        for sub in sorted(subdirs):
            img_dir = sub / "blurred_image"
            msk_dir = sub / "mask"
            if img_dir.exists() and msk_dir.exists():
                print(f"Collecting pairs from nested split: {sub.name}")
                pairs.extend(collect_from_pair_dirs(img_dir, msk_dir))
            else:
                print(
                    f"[INFO] Skipping subdir {sub} (missing blurred_image/mask); "
                    f"contents: {[p.name for p in sub.iterdir()]}"
                )

        if not pairs:
            raise RuntimeError(
                "P3M-10k validation structure not as expected.\n"
                f"Tried all subfolders under: {split_dir} but found no usable "
                "blurred_image/mask directories."
            )

    # Apply global max_samples limit if requested
    if max_samples is not None and max_samples > 0 and len(pairs) > max_samples:
        pairs = pairs[:max_samples]

    print(
        f"Collected {len(pairs)} image/mask pairs from split '{split_dir.name}' "
        f"(requested max={max_samples!r})."
    )
    return pairs


def _copy_pairs_to_dest(
    pairs: Iterable[Tuple[Path, Path]],
    images_dest: Path,
    masks_dest: Path,
) -> None:
    """Copy image/mask pairs into flat destination folders."""
    images_dest.mkdir(parents=True, exist_ok=True)
    masks_dest.mkdir(parents=True, exist_ok=True)

    num_copied = 0
    for img_path, mask_path in pairs:
        # Keep original filename to preserve pairing by stem
        img_dest = images_dest / img_path.name
        mask_dest = masks_dest / mask_path.name

        shutil.copy2(img_path, img_dest)
        shutil.copy2(mask_path, mask_dest)
        num_copied += 1

    print(
        f"Copied {num_copied} pairs to:\n"
        f"  images: {images_dest}\n"
        f"  masks : {masks_dest}"
    )


def organize_p3m10k(
    p3m_root: Path,
    data_root: Path,
    train_cfg: SplitConfig,
    val_cfg: SplitConfig,
) -> None:
    """
    Organize P3M-10k into <data_root>/train|val/images|masks.

    Args:
        p3m_root: Root path of P3M-10k (containing 'train' and 'validation').
        data_root: Destination root inside this project.
        train_cfg: SplitConfig for training set.
        val_cfg: SplitConfig for validation set.
    """
    print(f"Organizing P3M-10k into project data root: {data_root}")
    train_src = p3m_root / "train"
    val_src = p3m_root / "validation"

    if not train_src.exists() or not val_src.exists():
        raise FileNotFoundError(
            "Expected 'train' and 'validation' directories inside P3M-10k root.\n"
            f"P3M-10k root: {p3m_root}\n"
            f"Found: {[p.name for p in p3m_root.iterdir()]}"
        )

    train_pairs = _collect_pairs_from_split(train_src, max_samples=train_cfg.max_samples)
    val_pairs = _collect_pairs_from_split(val_src, max_samples=val_cfg.max_samples)

    train_images = data_root / "train" / "images"
    train_masks = data_root / "train" / "masks"
    val_images = data_root / "val" / "images"
    val_masks = data_root / "val" / "masks"

    _copy_pairs_to_dest(train_pairs, train_images, train_masks)
    _copy_pairs_to_dest(val_pairs, val_images, val_masks)

    print("P3M-10k organization complete.")
    print(f"Train images: {len(list(train_images.glob('*')))}")
    print(f"Train masks : {len(list(train_masks.glob('*')))}")
    print(f"Val images  : {len(list(val_images.glob('*')))}")
    print(f"Val masks   : {len(list(val_masks.glob('*')))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and organize the P3M-10k dataset for training."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data_p3m",
        help="Destination root directory for organized dataset "
        "(will contain train/ and val/).",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=-1,
        help="Maximum number of training samples to copy (-1 = no limit).",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=-1,
        help="Maximum number of validation samples to copy (-1 = no limit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    max_train = None if args.max_train is None or args.max_train <= 0 else args.max_train
    max_val = None if args.max_val is None or args.max_val <= 0 else args.max_val

    p3m_root = download_p3m10k()

    train_cfg = SplitConfig(name="train", max_samples=max_train)
    val_cfg = SplitConfig(name="validation", max_samples=max_val)

    organize_p3m10k(p3m_root, data_root, train_cfg, val_cfg)


if __name__ == "__main__":
    main()