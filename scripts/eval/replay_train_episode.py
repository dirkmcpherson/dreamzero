#!/usr/bin/env python3
"""
Replay a training episode against a running DreamZero inference server,
then compare predicted actions/video vs ground truth.

Usage (server already running on localhost:8000):
  python scripts/eval/replay_train_episode.py \
      --data-root ./data/gen3_lite_lerobot \
      --episode 0 \
      --host localhost --port 8000 \
      --out-dir ./eval_replay

What it does:
  1. Loads episode <episode> from a LeRobot-format dataset
  2. Reads the language instruction from meta/episodes.jsonl
  3. Reads ground truth actions from data/chunk-XXX/episode_XXXXXX.parquet
  4. Streams real video frames to the server (same 4-frame schedule as test_client_AR.py)
  5. Logs predicted action chunks
  6. Writes a side-by-side numpy comparison (predicted vs GT actions) to <out_dir>/
  7. The server itself saves predicted videos to its checkpoint dir
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

# Make repo root importable so we can use the shipped client.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval_utils.policy_client import WebsocketClientPolicy  # noqa: E402
from eval_utils import policy_server  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Same anchor schedule as test_client_AR.py
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24


# ---------- LeRobot v2 readers ----------

def load_episode_language(data_root: Path, episode_idx: int) -> str:
    """Read the language instruction for an episode from meta/episodes.jsonl + tasks.jsonl."""
    episodes_path = data_root / "meta" / "episodes.jsonl"
    tasks_path = data_root / "meta" / "tasks.jsonl"

    task_idx = None
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            if ep.get("episode_index") == episode_idx:
                tasks = ep.get("tasks") or []
                if isinstance(tasks, list) and tasks:
                    # tasks may be a list of strings already, or task indices
                    if isinstance(tasks[0], str):
                        return tasks[0]
                    task_idx = tasks[0]
                break

    if task_idx is None:
        raise RuntimeError(f"No task found for episode {episode_idx} in {episodes_path}")

    with open(tasks_path) as f:
        for line in f:
            t = json.loads(line)
            if t.get("task_index") == task_idx:
                return t["task"]
    raise RuntimeError(f"task_index={task_idx} missing from {tasks_path}")


def find_episode_videos(data_root: Path, episode_idx: int) -> dict[str, Path]:
    """Find per-camera mp4 paths for an episode. Returns {camera_name: path}."""
    videos_dir = data_root / "videos"
    chunk_idx = episode_idx // 1000
    chunk_dir = videos_dir / f"chunk-{chunk_idx:03d}"
    if not chunk_dir.exists():
        # Some datasets use a single chunk dir
        chunks = list(videos_dir.glob("chunk-*"))
        if not chunks:
            raise RuntimeError(f"No video chunks under {videos_dir}")
        chunk_dir = chunks[0]

    cam_files: dict[str, Path] = {}
    for cam_dir in chunk_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        cam_name = cam_dir.name.replace("observation.images.", "")
        ep_file = cam_dir / f"episode_{episode_idx:06d}.mp4"
        if ep_file.exists():
            cam_files[cam_name] = ep_file
    if not cam_files:
        raise RuntimeError(f"No mp4s for episode {episode_idx} in {chunk_dir}")
    return cam_files


def load_episode_actions(data_root: Path, episode_idx: int) -> np.ndarray | None:
    """Load ground truth actions from the parquet shard. Returns (T, A) or None on failure."""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        logging.warning("pyarrow not installed — skipping GT action loading")
        return None

    chunk_idx = episode_idx // 1000
    pq_path = data_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
    if not pq_path.exists():
        candidates = list((data_root / "data").rglob(f"episode_{episode_idx:06d}.parquet"))
        if not candidates:
            logging.warning(f"No parquet for episode {episode_idx}")
            return None
        pq_path = candidates[0]

    table = pq.read_table(pq_path)
    cols = table.column_names
    action_col = next((c for c in cols if c == "action" or c.endswith(".action")), None)
    if action_col is None:
        logging.warning(f"No 'action' column in {pq_path}; cols={cols}")
        return None
    arr = np.asarray(table[action_col].to_pylist(), dtype=np.float32)
    return arr


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
    return np.stack(frames, axis=0)


# ---------- camera mapping (dataset -> server obs keys) ----------

def build_camera_mapping(
    dataset_cams: dict[str, Path],
    server_config: "policy_server.PolicyServerConfig",
) -> dict[str, Path]:
    """Map dataset camera names to the obs keys the server expects.

    Heuristic: any cam name containing 'wrist' -> wrist_image_left.
    Remaining cams fill exterior_image_{i}_left slots in sorted order.
    """
    n_ext = server_config.n_external_cameras
    needs_wrist = server_config.needs_wrist_camera

    wrist_cams = [name for name in dataset_cams if "wrist" in name.lower()]
    ext_cams = sorted(name for name in dataset_cams if "wrist" not in name.lower())

    obs_key_to_path: dict[str, Path] = {}
    if needs_wrist:
        if not wrist_cams:
            raise RuntimeError("Server wants wrist cam but none found in dataset")
        obs_key_to_path["observation/wrist_image_left"] = dataset_cams[wrist_cams[0]]

    if not ext_cams and n_ext > 0:
        raise RuntimeError(f"Server wants {n_ext} external cams, dataset has 0")
    for i in range(n_ext):
        # If dataset has fewer ext cams than server expects, cycle (duplicate) the available ones.
        src = ext_cams[i % len(ext_cams)]
        obs_key_to_path[f"observation/exterior_image_{i}_left"] = dataset_cams[src]
        if i >= len(ext_cams):
            logging.warning(f"Duplicating ext cam {src!r} into slot {i} (dataset has only {len(ext_cams)})")

    return obs_key_to_path


# ---------- main eval loop ----------

def build_obs(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    prompt: str,
    session_id: str,
    state_dim: int,
) -> dict:
    obs: dict = {}
    for cam_key, frames in camera_frames.items():
        selected = frames[frame_indices]
        if len(frame_indices) == 1:
            selected = selected[0]
        obs[cam_key] = selected
    obs["observation/joint_position"] = np.zeros(state_dim, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="LeRobot dataset root (e.g. ./data/gen3_lite_lerobot)")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--num-chunks", type=int, default=3)
    p.add_argument("--out-dir", default="./eval_replay")
    p.add_argument("--state-dim", type=int, default=7,
                   help="Length of observation/joint_position vector to send (7 = 6-dof + gripper-pad).")
    p.add_argument("--prompt-override", default=None,
                   help="Override the language prompt (default: use episode's task string).")
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Read episode metadata ----
    prompt = args.prompt_override or load_episode_language(data_root, args.episode)
    logging.info(f"Episode {args.episode} prompt: {prompt!r}")

    cam_paths = find_episode_videos(data_root, args.episode)
    logging.info(f"Dataset cameras: {list(cam_paths.keys())}")

    gt_actions = load_episode_actions(data_root, args.episode)
    if gt_actions is not None:
        logging.info(f"Ground truth actions: shape={gt_actions.shape}, "
                     f"range=[{gt_actions.min():.3f}, {gt_actions.max():.3f}]")

    # ---- 2. Connect to server ----
    logging.info(f"Connecting to {args.host}:{args.port}...")
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = client.get_server_metadata()
    server_config = policy_server.PolicyServerConfig(**metadata)
    logging.info(f"Server config: {server_config}")

    # ---- 3. Map cams + load frames ----
    obs_key_to_path = build_camera_mapping(cam_paths, server_config)
    logging.info("Camera mapping:")
    for k, v in obs_key_to_path.items():
        logging.info(f"  {k}  <-  {v.name}")

    camera_frames = {k: load_all_frames(v) for k, v in obs_key_to_path.items()}
    total_frames = min(v.shape[0] for v in camera_frames.values())
    logging.info(f"Episode length: {total_frames} frames")

    # ---- 4. Run frame schedule ----
    session_id = str(uuid.uuid4())

    predicted_actions: list[np.ndarray] = []
    anchor_frames: list[int] = []

    # Initial: single frame at index 0
    obs = build_obs(camera_frames, [0], prompt, session_id, args.state_dim)
    t0 = time.time()
    actions = client.infer(obs)
    logging.info(f"[init  ] frame 0 -> action {actions.shape} "
                 f"range=[{actions.min():.3f},{actions.max():.3f}] in {time.time()-t0:.1f}s")
    predicted_actions.append(np.asarray(actions))
    anchor_frames.append(0)

    current = 23
    for chunk_idx in range(args.num_chunks):
        indices = [max(current + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            logging.info(f"Hit end of episode at chunk {chunk_idx}, stopping")
            break
        obs = build_obs(camera_frames, indices, prompt, session_id, args.state_dim)
        t0 = time.time()
        actions = client.infer(obs)
        logging.info(f"[chunk {chunk_idx}] anchor={current} -> action {actions.shape} "
                     f"range=[{actions.min():.3f},{actions.max():.3f}] in {time.time()-t0:.1f}s")
        predicted_actions.append(np.asarray(actions))
        anchor_frames.append(current)
        current += ACTION_HORIZON

    # Reset triggers server-side video save
    client.reset({})

    # ---- 5. Save comparison ----
    pred = np.stack(predicted_actions, axis=0)  # (n_chunks, action_horizon, A)
    save_path = out_dir / f"episode_{args.episode:06d}_predicted.npz"
    save_kwargs = dict(
        predicted_actions=pred,
        anchor_frames=np.asarray(anchor_frames),
        prompt=prompt,
    )
    if gt_actions is not None:
        save_kwargs["gt_actions"] = gt_actions

        # Best-effort: extract GT slices aligned with each predicted chunk
        gt_slices = []
        for anchor in anchor_frames:
            end = min(anchor + ACTION_HORIZON, gt_actions.shape[0])
            slc = gt_actions[anchor:end]
            if slc.shape[0] < ACTION_HORIZON:
                pad = np.zeros((ACTION_HORIZON - slc.shape[0], gt_actions.shape[1]), dtype=slc.dtype)
                slc = np.concatenate([slc, pad], axis=0)
            gt_slices.append(slc)
        gt_aligned = np.stack(gt_slices, axis=0)
        save_kwargs["gt_actions_aligned"] = gt_aligned

        # Print L2 per chunk so the user gets feedback immediately
        a_dim = min(pred.shape[-1], gt_aligned.shape[-1])
        for i, anchor in enumerate(anchor_frames):
            diff = pred[i, :, :a_dim] - gt_aligned[i, :, :a_dim]
            l2 = float(np.sqrt((diff ** 2).mean()))
            logging.info(f"  chunk {i} (anchor={anchor}): RMSE vs GT = {l2:.4f}  "
                         f"(GT range [{gt_aligned[i,:,:a_dim].min():.3f},{gt_aligned[i,:,:a_dim].max():.3f}]  "
                         f"pred range [{pred[i,:,:a_dim].min():.3f},{pred[i,:,:a_dim].max():.3f}])")

    np.savez(save_path, **save_kwargs)
    logging.info(f"Saved predictions + GT to {save_path}")
    logging.info(f"Server should have saved predicted video to its --model-path's real_world_eval_gen_*/ dir")


if __name__ == "__main__":
    main()
