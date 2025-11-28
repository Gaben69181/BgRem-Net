"""backend/timelapse.py

Timelapse generation from per-epoch segmentation prediction frames.

This module expects that `backend/train.py` has been run with the
`--timelapse` flag, which stores frames in the following structure:

    outputs/timelapse/
        sample_00/
            epoch_001.png
            epoch_002.png
            ...
        sample_01/
            epoch_001.png
            ...

Each PNG contains a horizontal triplet:

    [ input | ground-truth mask | predicted mask ]

This script can then stitch the frames of each `sample_xx` directory
into an animated GIF or MP4 video to visualize how the segmentation
evolves over training epochs.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import imageio.v2 as imageio


def list_sample_dirs(frames_root: str) -> List[str]:
    """Return sorted list of `sample_xx` subdirectories under frames_root."""
    if not os.path.isdir(frames_root):
        raise FileNotFoundError(f"Frames root directory not found: {frames_root}")

    sample_dirs = [
        os.path.join(frames_root, d)
        for d in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, d)) and d.startswith("sample_")
    ]
    sample_dirs.sort()
    if not sample_dirs:
        raise RuntimeError(f"No sample_* directories found under {frames_root}")
    return sample_dirs


def list_frames(sample_dir: str) -> List[str]:
    """
    List frame images for a given sample directory.

    Expected filenames: epoch_001.png, epoch_002.png, ...
    """
    frames = [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.lower().endswith(".png")
    ]
    frames.sort()
    if not frames:
        raise RuntimeError(f"No PNG frames found in {sample_dir}")
    return frames


def make_gif(frames: List[str], out_path: str, fps: int = 3) -> None:
    """Create an animated GIF from a list of frame image paths."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    images = [imageio.imread(p) for p in frames]
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(out_path, images, duration=duration, loop=0)
    print(f"Saved GIF timelapse: {out_path}")


def make_mp4(frames: List[str], out_path: str, fps: int = 10) -> None:
    """
    Create an MP4 timelapse using imageio's ffmpeg writer.

    Note: Requires ffmpeg to be installed and available on PATH.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps)
    try:
        for p in frames:
            frame = imageio.imread(p)
            writer.append_data(frame)
    finally:
        writer.close()
    print(f"Saved MP4 timelapse: {out_path}")


def build_timelapse_for_sample(
    sample_dir: str,
    out_dir: str,
    fmt: str = "gif",
    fps: int = 5,
) -> str:
    """
    Build a timelapse video for a single sample_* directory.

    Args:
        sample_dir: Path to `sample_xx` directory containing epoch_*.png.
        out_dir: Output directory for resulting video.
        fmt: "gif" or "mp4".
        fps: Frames per second.

    Returns:
        Path to the generated video file.
    """
    frames = list_frames(sample_dir)
    sample_name = os.path.basename(sample_dir.rstrip("/\\"))
    os.makedirs(out_dir, exist_ok=True)

    if fmt.lower() == "gif":
        out_path = os.path.join(out_dir, f"{sample_name}.gif")
        make_gif(frames, out_path, fps=fps)
    elif fmt.lower() == "mp4":
        out_path = os.path.join(out_dir, f"{sample_name}.mp4")
        make_mp4(frames, out_path, fps=fps)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'gif' or 'mp4'.")
    return out_path


def build_all_timelapses(
    frames_root: str = "outputs/timelapse",
    out_dir: str = "outputs/videos",
    fmt: str = "gif",
    fps: int = 5,
) -> None:
    """
    Build timelapse videos for all samples under frames_root.

    Typical usage:

        python backend/timelapse.py --frames-root outputs/timelapse --out-dir outputs/videos --format gif
    """
    sample_dirs = list_sample_dirs(frames_root)
    print(f"Found {len(sample_dirs)} sample_* directories in {frames_root}")

    for sd in sample_dirs:
        try:
            build_timelapse_for_sample(sd, out_dir=out_dir, fmt=fmt, fps=fps)
        except Exception as e:
            print(f"WARNING: Failed to build timelapse for {sd}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build segmentation timelapse videos from saved frames.")
    parser.add_argument(
        "--frames-root",
        type=str,
        default="outputs/timelapse",
        help="Root directory where sample_*/epoch_XXX.png frames are stored.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/videos",
        help="Target directory to store generated timelapse videos.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gif",
        choices=["gif", "mp4"],
        help="Output video format.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_all_timelapses(
        frames_root=args.frames_root,
        out_dir=args.out_dir,
        fmt=args.format,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()