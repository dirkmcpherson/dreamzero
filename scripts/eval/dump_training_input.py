"""Dump the post-transform video the model sees during training.

Replicates the gen3_lite data pipeline:
  1. Read per-camera mp4s
  2. CenterCrop(scale=0.95)            (eval-mode of VideoCrop)
  3. Resize to (image_resolution_height, image_resolution_width)  default (176, 320)
  4. Tile into the 2x2 grid used by _prepare_video for non-DROID embodiments:
       [wrist,    black]
       [exterior, black]
  5. Save as mp4 so you can eyeball what the model actually saw.

Usage:
    python scripts/eval/dump_training_input.py \
        --data-root ./kinova_gen3_lite_smoke_v2 \
        --episode 0 \
        --out ./debug_training_input.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


CROP_SCALE = 0.95
TARGET_H = 176
TARGET_W = 320


def load_all_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)  # (T, H, W, 3)


def center_crop(frames: np.ndarray, scale: float) -> np.ndarray:
    """Replicates VideoCrop with mode='eval' (CenterCrop)."""
    _, h, w, _ = frames.shape
    new_h, new_w = int(h * scale), int(w * scale)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return frames[:, top:top + new_h, left:left + new_w, :]


def resize_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    out = np.empty((frames.shape[0], height, width, 3), dtype=frames.dtype)
    for i in range(frames.shape[0]):
        out[i] = cv2.resize(frames[i], (width, height), interpolation=cv2.INTER_LINEAR)
    return out


def tile_2x2(wrist: np.ndarray, exterior: np.ndarray) -> np.ndarray:
    """Mirrors dreamzero_cotrain._prepare_video for non-DROID 2-view embodiments.

    Layout:
        [wrist,    black]
        [exterior, black]
    (view 0 = wrist  -> top-left, view 1 = exterior -> bottom-left)
    """
    t, h, w, c = wrist.shape
    assert exterior.shape == wrist.shape, (wrist.shape, exterior.shape)
    out = np.zeros((t, 2 * h, 2 * w, c), dtype=wrist.dtype)
    out[:, :h, :w, :] = wrist
    out[:, h:, :w, :] = exterior
    return out


def write_mp4(frames: np.ndarray, out_path: Path, fps: int) -> None:
    t, h, w, _ = frames.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for i in range(t):
        writer.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="LeRobot v2 dataset root")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", default="./debug_training_input.mp4")
    p.add_argument("--height", type=int, default=TARGET_H)
    p.add_argument("--width", type=int, default=TARGET_W)
    p.add_argument("--crop-scale", type=float, default=CROP_SCALE)
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    root = Path(args.data_root).resolve()
    chunk = args.episode // 1000
    chunk_dir = root / "videos" / f"chunk-{chunk:03d}"

    wrist_mp4 = chunk_dir / "observation.images.wrist" / f"episode_{args.episode:06d}.mp4"
    ext_mp4 = chunk_dir / "observation.images.exterior" / f"episode_{args.episode:06d}.mp4"
    for f in (wrist_mp4, ext_mp4):
        if not f.exists():
            raise FileNotFoundError(f)

    wrist = load_all_frames(wrist_mp4)
    ext = load_all_frames(ext_mp4)
    print(f"Loaded wrist   {wrist.shape}, exterior {ext.shape}")

    wrist_c = center_crop(wrist, args.crop_scale)
    ext_c = center_crop(ext, args.crop_scale)
    print(f"After crop    : wrist {wrist_c.shape}, exterior {ext_c.shape}")

    wrist_r = resize_frames(wrist_c, args.height, args.width)
    ext_r = resize_frames(ext_c, args.height, args.width)
    print(f"After resize  : wrist {wrist_r.shape}, exterior {ext_r.shape}")

    t_min = min(wrist_r.shape[0], ext_r.shape[0])
    tiled = tile_2x2(wrist_r[:t_min], ext_r[:t_min])
    print(f"Tiled (2x2)   : {tiled.shape}  -> right half should be black")

    out_path = Path(args.out).resolve()
    write_mp4(tiled, out_path, fps=args.fps)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
