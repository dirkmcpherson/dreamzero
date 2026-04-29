# DreamZero: 6-DOF Single-Arm Finetuning Guide

## Overview

DreamZero is a 14B parameter World Action Model (WAM) built on a pretrained Wan2.1 image-to-video diffusion backbone. It jointly predicts future video frames and robot actions via flow matching, enabling zero-shot generalization to unseen tasks and few-shot adaptation to new embodiments. It is an NVIDIA GEAR Lab project.

## Why It Works for a 6-DOF Single Arm

1. **DROID (Franka single-arm) is a first-class embodiment.** The paper trains and evaluates DreamZero-DROID on the Franka single-arm robot. Single-arm is core, not an afterthought.

2. **DOF-agnostic architecture.** The action/state dimensions are set entirely by your data config. The `MultiEmbodimentActionEncoder` creates a per-embodiment MLP sized to whatever dimensions your data has. A 6-DOF arm with gripper gives you a 7-dim action space (vs DROID's 9-dim) -- no code changes needed.

3. **Fewer DOF is easier.** The paper notes that higher-DOF robots need more data because the implicit inverse dynamics mapping grows combinatorially. A 6-DOF arm is simpler than 7-DOF, so you may need less data.

4. **Few-shot embodiment adaptation is a headline result.** DreamZero pretrained on AgiBot transfers to a new robot (YAM) with only **30 minutes of play data** via LoRA finetuning.

## Hardware Requirements

### Training

The model is **14B parameters**. LoRA finetuning freezes most weights and uses DeepSpeed ZeRO-2, but you still need multi-GPU. With `per_device_train_batch_size=4` at 320x176 resolution, plan on **4-8x H100/A100 80GB**. ZeRO-3 with CPU offload could reduce this but will be slower.

### Inference

The paper achieves 7Hz on **2x GB200s** with all optimizations. On **2x H100s** without Blackwell-specific optimizations, expect ~3-5s per action chunk -- usable with asynchronous execution (robot executes previous chunk while next one computes).

### Smaller Option

The **Wan2.2-TI2V-5B** backbone (see `docs/WAN22_BACKBONE.md`) is ~3x smaller. The ablation shows 5B gets 21% vs 50% task progress for 14B -- a real drop, but potentially viable for a simple pick-and-place task, and much friendlier on hardware.

## Paper Results (Single-Arm Reference)

- **DROID-Franka seen tasks**: 75% success rate
- **DROID-Franka unseen tasks**: 49% task progress, 22.5% success rate
- **Few-shot adaptation** (AgiBot to YAM, 30 min data): retains zero-shot generalization

---

## Step-by-Step: Finetune and Run "Pick Up the Block"

### Prerequisites

- A 6-DOF arm with a parallel gripper and 2-3 cameras (1 external + 1 wrist minimum)
- A teleoperation setup to record demonstrations (LeRobot v2 format)
- 4-8x H100/A100 80GB GPUs (training), 2x H100+ (inference)
- ~150GB disk for checkpoints

### Step 0: Install Dependencies and Download Checkpoints

```bash
# Install DreamZero
pip install -e .

# Download pretrained DreamZero-AgiBot checkpoint (~45GB)
# This is the base model you'll LoRA-finetune from
huggingface-cli download GEAR-Dreams/DreamZero-AgiBot \
    --local-dir ./checkpoints/DreamZero-AgiBot

# Download Wan2.1 backbone weights (auto-downloaded by training script too)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

# Download tokenizer
huggingface-cli download google/umt5-xxl \
    --local-dir ./checkpoints/umt5-xxl
```

### Step 1: Collect Teleoperation Data

Record 50-100 demonstrations of picking up blocks. More diversity in block position/color matters more than repetition. The paper showed results with as few as 55 trajectories (~30 min).

What to record:
- **Cameras**: 2-3 views as MP4 video files (see camera placement below)
- **State**: 6 joint positions + 1 gripper position per timestep
- **Actions**: 6 joint position targets + 1 gripper target per timestep
- **Language annotation**: "pick up the block" (or varied: "pick up the red block", "grab the block", etc.)
- **FPS**: 30Hz recommended (match to your robot's control frequency)

#### Camera Placement

DROID (the single-arm reference) uses 3 cameras: two fixed exterior views from different angles, plus one wrist-mounted camera. For a 2-camera setup, use:

1. **One fixed exterior camera** -- overhead or ~45-degree downward angle from the side/front, covering the full workspace (table, block, and arm). The block and gripper should both be visible throughout the entire reach-grasp-lift trajectory.
2. **One wrist-mounted camera** -- attached to the end-effector, looking toward the gripper. This gives close-up spatial information for the final approach and grasp, which is critical for manipulation accuracy.

If you add a third camera, place it as a second exterior view from a substantially different angle (e.g., front-left and front-right) to help the model resolve depth ambiguity.

Practical tips:
- Keep cameras fixed between episodes -- the model treats camera placement as part of the environment
- Resolution doesn't need to be high; everything gets resized to 320x176 for training
- Avoid views where the block is frequently occluded by the arm
- Consistent, even lighting helps; the pipeline applies color jitter augmentation but large shadows still hurt

#### LeRobot Version Note

**DreamZero requires LeRobot v2.0 format**, but the latest LeRobot (>= 0.4.0) records in **v3.0 format** by default. The key difference is that v2 stores one episode per file while v3 packs many episodes into larger shard files.

**Recommended approach**: Collect data with the latest LeRobot tooling (v3), then convert to v2 before running the GEAR converter:

```bash
# After collecting data with lerobot >= 0.4.0 (outputs v3 format):
python -m lerobot.datasets.v30.convert_dataset_v30_to_v21 --repo-id=<your-dataset>
```

Alternatively, pin `lerobot < 0.4.0` to record directly in v2 format and skip the conversion step.

#### Expected v2 Directory Structure

After conversion (or if recording directly in v2), your dataset should look like:

```
data/myarm/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # columns: observation.state, action, timestamp, ...
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_exterior/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       └── observation.images.cam_wrist/
│           ├── episode_000000.mp4
│           └── ...
└── meta/
    └── info.json                    # must contain: features, total_episodes, fps
```

Each parquet file should have columns for `observation.state` (7-dim: 6 joints + 1 gripper), `action` (7-dim), and `annotation.task` (string).

### Step 2: Convert Dataset to GEAR Format

```bash
python scripts/data/convert_lerobot_to_gear.py \
    --dataset-path ./data/myarm \
    --embodiment-tag myarm \
    --state-keys '{"joint_position": [0, 6], "gripper_position": [6, 7]}' \
    --action-keys '{"joint_position": [0, 6], "gripper_position": [6, 7]}' \
    --relative-action-keys joint_position \
    --task-key annotation.task \
    --force
```

This creates metadata files under `data/myarm/meta/` without modifying your parquet or video files:
- `modality.json` -- maps state/action/video keys with index ranges
- `embodiment.json` -- `{"embodiment_tag": "myarm"}`
- `stats.json` -- per-feature normalization statistics (mean, std, q01, q99)
- `relative_stats_dreamzero.json` -- relative action stats (action minus state)
- `tasks.jsonl` -- unique task descriptions
- `episodes.jsonl` -- per-episode metadata

### Step 3: Register the Embodiment Tag

Add your arm to `groot/vla/data/schema/embodiment_tags.py`:

```python
class EmbodimentTag(Enum):
    ...
    MYARM = "myarm"
    """
    6-DOF single arm with parallel gripper.
    """
```

And add `"myarm"` to the `VALID_EMBODIMENT_TAGS` list in `scripts/data/convert_lerobot_to_gear.py`.

### Step 4: Add Modality Config and Transforms

Edit `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`.

Add the modality config for your arm (adjust camera names to match your `modality.json`):

```yaml
modality_config_myarm:
  video:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    eval_delta_indices: [0]
    modality_keys:
      - video.cam_exterior
      - video.cam_wrist
  state:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:
      - state.joint_position
      - state.gripper_position
  action:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    modality_keys:
      - action.joint_position
      - action.gripper_position
  language:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:
      - annotation.task
```

#### Understanding the Transforms

The transform pipeline processes raw data into the format the model expects. You define it as a list of steps applied in order. Here's what each one does:

**Video transforms** (applied to all camera streams):

| Transform | What it does | Config |
|---|---|---|
| `VideoToTensor` | Converts raw uint8 frames to float tensors | No params |
| `VideoCrop` | Random crop at 95% scale during training (data augmentation to handle slight camera shifts) | `scale: 0.95`, `mode: random` |
| `VideoResize` | Resizes to training resolution | `height: 176`, `width: 320` (set by training script) |
| `VideoColorJitter` | Random brightness/contrast/saturation/hue perturbation (augmentation for lighting variation) | `brightness: 0.3, contrast: 0.4, saturation: 0.5, hue: 0.08` |
| `VideoToNumpy` | Converts back to numpy for the data collator | No params |

**State/Action transforms** (applied to joint + gripper channels):

| Transform | What it does | Config |
|---|---|---|
| `StateActionToTensor` | Converts raw arrays to tensors | No params |
| `StateActionTransform` | Normalizes values to [-1, 1] using dataset statistics | `normalization_modes` per key |

The `normalization_modes` field tells the normalizer which strategy to use per key. Use **`q99`** for everything -- it clips to the 1st/99th percentile from `stats.json`, which is robust to outliers. Every state and action key must appear here or training will error.

**Structural transforms** (order matters):

| Transform | What it does |
|---|---|
| `ConcatTransform` | Concatenates multi-camera frames into a single image and multi-key state/action into single vectors. The `_concat_order` lists control the ordering. |
| `${model_specific_transform}` | Internal model tokenization (required, don't change) |

You should not need to modify any of the default values. The only things to customize are the `apply_to` keys (your camera/state/action names) and the `normalization_modes` keys, which must exactly match your `modality.json`.

Add the transform block:

```yaml
transform_myarm:
  _target_: groot.vla.data.transform.ComposedModalityTransform
  transforms:
    # Video
    - <<: *totensor_cfg
      apply_to: ${modality_config_myarm.video.modality_keys}
    - <<: *crop_cfg
      apply_to: ${modality_config_myarm.video.modality_keys}
    - <<: *resize_cfg
      apply_to: ${modality_config_myarm.video.modality_keys}
    - <<: *color_jitter_cfg
      apply_to: ${modality_config_myarm.video.modality_keys}
    - <<: *to_numpy_cfg
      apply_to: ${modality_config_myarm.video.modality_keys}

    # State
    - _target_: groot.vla.data.transform.StateActionToTensor
      apply_to: ${modality_config_myarm.state.modality_keys}
    - _target_: groot.vla.data.transform.StateActionTransform
      apply_to: ${modality_config_myarm.state.modality_keys}
      normalization_modes:
        state.joint_position: q99
        state.gripper_position: q99

    # Action
    - _target_: groot.vla.data.transform.StateActionToTensor
      apply_to: ${modality_config_myarm.action.modality_keys}
    - _target_: groot.vla.data.transform.StateActionTransform
      apply_to: ${modality_config_myarm.action.modality_keys}
      normalization_modes:
        action.joint_position: q99
        action.gripper_position: q99

    # Concat
    - _target_: groot.vla.data.transform.ConcatTransform
      video_concat_order: ${modality_config_myarm.video.modality_keys}
      state_concat_order: ${modality_config_myarm.state.modality_keys}
      action_concat_order: ${modality_config_myarm.action.modality_keys}

    # Model-specific (required, don't change)
    - ${model_specific_transform}
```

Register in the four global maps at the bottom of the same file:

```yaml
modality_configs:
  ...
  myarm: ${modality_config_myarm}

transforms:
  ...
  myarm: ${transform_myarm}

metadata_versions:
  ...
  myarm: '0221'

fps:
  ...
  myarm: 30
```

**Important**: The `modality_keys` values (e.g., `state.joint_position`) must exactly match the keys in your generated `modality.json`.

### Step 5: Create a Dataset YAML

Create `groot/vla/configs/data/dreamzero/myarm_relative.yaml`:

```yaml
# @package _global_

defaults:
  - dreamzero/base_48_wan_fine_aug_relative
  - _self_

max_state_dim: 64
use_global_metadata: false
relative_action: true
relative_action_per_horizon: false
relative_action_keys:
  - joint_position
max_chunk_size: 5
dataset_shard_sampling_rate: 0.1
mixture_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotMixtureDataset.from_mixture_spec
single_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotSubLangSingleActionChunkDatasetDROID

myarm_data_root: ???

train_dataset:
  _target_: ${mixture_dataset_cls}
  _convert_: object
  mixture_spec:
    - dataset_path:
        myarm:
          - ${myarm_data_root}
      dataset_weight: 1.0
      distribute_weights: true

  dataset_class: ${single_dataset_cls}
  all_modality_configs: ${modality_configs}
  all_transforms: ${transforms}
  metadata_versions: ${metadata_versions}
  fps: ${fps}
  dataset_kwargs:
    video_backend: decord
    use_global_metadata: ${use_global_metadata}
    max_chunk_size: ${max_chunk_size}
    relative_action: ${relative_action}
    relative_action_keys: ${relative_action_keys}
    relative_action_per_horizon: ${relative_action_per_horizon}
  mixture_kwargs:
    training: true
    balance_dataset_weights: false
    seed: 42
    shard_sampling_rate: ${dataset_shard_sampling_rate}
```

### Step 6: Create a Training Script

Create `scripts/train/myarm_training.sh`:

```bash
#!/bin/bash
export HYDRA_FULL_ERROR=1

# ============ CONFIGURATION ============
DATA_ROOT=${DATA_ROOT:?"Set DATA_ROOT to your GEAR-converted dataset path"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_myarm_lora"}

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
# =======================================

# Auto-download weights if missing
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [ ! -f "$DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: meta/embodiment.json missing -- run convert_lerobot_to_gear.py first"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/myarm_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=2500 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=5000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    myarm_data_root=$DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=./checkpoints/DreamZero-AgiBot \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
```

Key differences from the template for your 6-DOF arm:
- `num_views=2` (2 cameras instead of 3)
- `max_steps=5000` (small dataset, LoRA converges fast -- the AgiBot script uses 5k too)
- `per_device_train_batch_size=1` (safe for GPU memory)

### Step 7: Launch Training

```bash
DATA_ROOT=./data/myarm bash scripts/train/myarm_training.sh

# Or with overrides:
DATA_ROOT=./data/myarm NUM_GPUS=4 OUTPUT_DIR=./checkpoints/myarm_run1 \
    bash scripts/train/myarm_training.sh
```

Training will save LoRA checkpoints to `OUTPUT_DIR` every 2500 steps. At 5000 steps with a small dataset, this should take a few hours on 8x H100s.

### Step 8: Deploy Inference -- Pick Up the Block

The architecture is a **WebSocket server** (on the cluster GPU) and a **lightweight client** (on the lab machine connected to the robot). The codebase provides both.

#### GPU Requirements

| Setup | VRAM needed | Latency per chunk | Use case |
|---|---|---|---|
| 1x L40S (48GB) | ~28GB (14B bf16) | ~10-16s | Prototyping and smoke tests |
| 2x L40S | ~14GB each | ~5-8s | Better latency via CFG parallelism |
| 2x H100 (80GB) | ~14GB each | ~3s | Production |

**A single L40S is enough for prototyping.** The 14B model fits in 28GB bf16. Inference is slower (~10-16s per 24-action chunk) because the two CFG passes run sequentially instead of in parallel, but it works. The robot executes each 24-action chunk in 0.8s at 30Hz, so for a block pick (3-5 chunks) you're looking at ~1 minute total with pauses between chunks.

#### On the cluster: Launch the inference server

```bash
# Request a single L40S via SLURM (adjust for your cluster)
srun --gres=gpu:1 --mem=64G --time=4:00:00 --pty bash

# Single-GPU launch (no torchrun needed)
CUDA_VISIBLE_DEVICES=0 python socket_test_optimized_AR.py \
    --port 8000 \
    --model-path ./checkpoints/dreamzero_myarm_lora/checkpoint-5000

# For 2-GPU (faster, if available):
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
#     --standalone --nproc_per_node=2 \
#     socket_test_optimized_AR.py \
#     --port 8000 --enable-dit-cache \
#     --model-path ./checkpoints/dreamzero_myarm_lora/checkpoint-5000
```

First inference takes a few minutes (model warmup + torch.compile). Subsequent calls are steady-state.

#### Network: SSH tunnel from lab to cluster

```bash
# From your lab machine:
ssh -L 8000:compute-node:8000 your-cluster-login-node

# For long sessions, use autossh to keep the tunnel alive:
autossh -M 0 -L 8000:compute-node:8000 your-cluster-login-node
```

Now `localhost:8000` on the lab machine routes to the inference server.

#### On the lab machine: Robot client

You'll need to adapt `socket_test_optimized_AR.py`'s `ARDroidRoboarenaPolicy` class to map your arm's observation keys (it's written for DROID's 7-DOF / 3-camera layout). The two methods to change are `_convert_observation()` and `_convert_action()`.

Then the client side uses `eval_utils/policy_client.py`:

```python
from eval_utils.policy_client import WebsocketClientPolicy
import numpy as np

client = WebsocketClientPolicy(host="localhost", port=8000)

# First call: single frame per camera
obs = {
    "observation/exterior_image_0_left": img_exterior,  # (H, W, 3) uint8
    "observation/wrist_image_left": img_wrist,           # (H, W, 3) uint8
    "observation/joint_position": joint_pos,              # (6,) float32
    "observation/gripper_position": gripper_pos,          # (1,) float32
    "prompt": "pick up the block",
    "session_id": "episode_001",
}
actions = client.infer(obs)  # Returns (24, 7) action chunk

# Execute the chunk on robot at 30Hz
for action in actions:
    robot.command_joints(action[:6], action[6])
    robot.wait_for_step()  # 1/30s

# Subsequent calls: 4 frames per camera
obs_next = {
    "observation/exterior_image_0_left": np.stack([f1, f2, f3, f4]),  # (4, H, W, 3)
    "observation/wrist_image_left": np.stack([w1, w2, w3, w4]),
    "observation/joint_position": joint_pos,
    "observation/gripper_position": gripper_pos,
    "prompt": "pick up the block",
    "session_id": "episode_001",
}
actions = client.infer(obs_next)

# New episode: reset
client.reset({"session_id": "episode_002"})
```

The server handles all normalization, inference, and action unnormalization internally. The client just sends raw observations and gets back absolute joint position targets.

Key details under the hood:
1. Images are resized to (180, 320) and normalized to [-1, 1]
2. State is normalized using q99 statistics from `stats.json`
3. Model outputs normalized relative actions
4. Actions are unnormalized and converted back to absolute: `absolute = relative + current_state`

### Step 9: Iterate

If pick-up success is low:
- **Collect more diverse data**: vary block position, color, lighting, table height
- **Train longer**: increase `max_steps` to 10k-20k
- **Add cameras**: a third viewpoint helps with spatial reasoning
- **Check action smoothness**: the model applies Savitzky-Golay filtering on action chunks to suppress high-frequency noise

---

## Pre-Training Checklist

- [ ] `meta/embodiment.json` exists and has `"myarm"` tag
- [ ] `meta/modality.json` has correct state/action/video/annotation keys
- [ ] `meta/stats.json` and `meta/relative_stats_dreamzero.json` exist
- [ ] `meta/tasks.jsonl` and `meta/episodes.jsonl` exist
- [ ] Embodiment tag `myarm` matches keys in `modality_configs` / `transforms` / `metadata_versions` / `fps`
- [ ] YAML `modality_keys` match `modality.json` keys exactly (with `state.`/`action.`/`video.`/`annotation.` prefix)
- [ ] Every state and action key appears in `normalization_modes` in the transform block
- [ ] `relative_action_keys` lists sub-key names that exist in both state and action
- [ ] Wan2.1-I2V-14B-480P and umt5-xxl weights downloaded
- [ ] DreamZero-AgiBot checkpoint downloaded to `./checkpoints/DreamZero-AgiBot`

## Key Considerations

- **Cameras**: 2-3 cameras. System concatenates multi-view frames into a single image. One external + one wrist camera is the minimum.
- **Task instructions**: Model is conditioned on language. Even simple annotations like "pick up the block" work. Varying the phrasing can help.
- **Sub-centimeter precision**: The paper notes limitations on tasks requiring very fine precision (key insertion, fine assembly). Block picking should be well within capability.

### Action Space: Joint Positions, Not EEF

DreamZero outputs **absolute joint position targets**, not end-effector deltas or velocity commands. The action space is `joint_position` -- for a 6-DOF arm, the model outputs 6 joint angle targets + 1 gripper target per timestep.

Internally, training uses **relative actions** (action - current_state) for better normalization, but at inference the server converts back to absolute: `absolute_joint_target = relative_prediction + current_joint_state`.

The control loop is:
1. You send: camera images + current joint positions + language instruction
2. Model returns: 24 absolute joint position targets `(24, 7)` -- a trajectory
3. Your robot executes those targets in sequence at 30Hz

If your robot or sim uses EEF/Cartesian control, you need either:
- Switch to joint-space position control (recommended, simpler)
- Add an IK layer between DreamZero's joint outputs and your EEF interface

### L40S Notes

- 1x L40S (48GB) is enough for prototyping inference. 14B bf16 uses ~28GB. Inference is ~10-16s per chunk (vs ~3s on 2x H100) since CFG passes run sequentially.
- 2x L40S works for inference with CFG parallelism, but disable `torch.compile` (`TORCH_COMPILE_DISABLE=1`) to avoid OOM from Triton autotuning overhead.
- Training (LoRA) on L40S requires 4-8 GPUs with ZeRO-2 + CPU offload.

---

## Ideal AWS GPU Setup ($50k Budget)

### Training: p4de.24xlarge (8x A100 80GB) -- ~$40/hr on-demand

This is the sweet spot for LoRA finetuning. 80GB per GPU gives comfortable headroom for the 14B model with ZeRO-2 (no CPU offload needed), and 8 GPUs means fast training.

| Task | Time | Cost |
|---|---|---|
| LoRA finetune, 5k steps | ~3-4 hours | ~$160 |
| LoRA finetune, 100k steps | ~60-80 hours | ~$3,200 |
| Full finetune from scratch, 100k steps | ~80-100 hours | ~$4,000 |
| 10 experimental runs (hyperparameter sweeps) | ~40 hours | ~$1,600 |

**Training subtotal: ~$5k-10k** depending on iteration.

Alternatively, **p5.48xlarge (8x H100 80GB)** at ~$98/hr is ~2x faster but ~2.5x the cost. Worth it if you're iterating rapidly and time matters more than money.

**Spot instances** can cut costs by 60-70% for training (training is fault-tolerant -- checkpoints every 2500 steps, so preemption just means resuming).

### Inference: p5.2xlarge (1x H100 80GB) or p5.48xlarge (8x H100 80GB)

For inference, you want low latency more than throughput:

| Instance | GPUs | Latency/chunk | Cost/hr | Use case |
|---|---|---|---|---|
| g6e.12xlarge | 4x L40S 48GB | ~5-8s (use 2) | ~$8 | Budget prototyping |
| p4de.24xlarge | 8x A100 80GB | ~4-5s (use 2) | ~$40 | Good balance |
| p5.48xlarge | 8x H100 80GB | ~3s (use 2) | ~$98 | Fastest available |

For real-time-ish control (target <5s chunks), **2x H100** is the practical minimum. Use a p5.48xlarge and only use 2 of the 8 GPUs for inference -- wasteful, but AWS doesn't offer 2x H100 instances.

**Inference subtotal: ~$5k-15k** for 100-200 hours of inference sessions.

### Recommended Budget Allocation

| Category | Spend | What |
|---|---|---|
| Training (spot) | ~$3k-5k | 10-20 LoRA runs on p4de spot instances |
| Inference (on-demand) | ~$5k-10k | 100+ hours on p5.48xlarge for eval sessions |
| Data storage (S3) | ~$500 | Checkpoints (~150GB each), datasets, videos |
| Buffer | ~$30k+ | Future experiments, full finetune, scale up |

**Total: ~$10k-20k** for a solid development cycle, leaving $30k+ for scaling up once you have a working pipeline.

### Cost-Saving Tips

- **Use spot for training**: 60-70% cheaper, resume from checkpoints on preemption
- **Stop instances between sessions**: Don't leave inference servers running overnight
- **Start with the 5B model** (Wan2.2-TI2V-5B): Fits on cheaper instances (g5.12xlarge, 4x A10G 24GB, ~$5/hr), good enough to validate the pipeline before committing to 14B
- **Use your school L40S cluster for iteration**: Save AWS credits for final training runs and real-time inference sessions that need H100s
- **Run 4 parallel inference servers on p5.48xlarge**: Use all 8 GPUs (4 × 2-GPU servers on ports 8000-8003) for 4x evaluation throughput at the same hourly cost

---

## Experimental Design: 12 Tasks × 3 Conditions

### Reference: Paper's Evaluation Protocol

The paper provides these benchmarks for calibrating our design:

**Training (paper, Section 4.1):**
- Pretraining: 100K steps, global batch size 128, on 500 hrs of data
- Post-training: 50K steps per task (shirt folding, fruit packing, table bussing)
- Few-shot adaptation (AgiBot → YAM): 55 trajectories (~30 min), LoRA, 5K steps
- Cross-embodiment transfer: 10K steps, 1:1 mix with pretraining data
- Ablations: 50K steps, batch size 32

**Inference (paper, Table 1):**
- Baseline (no optimizations): 5.7s per action chunk
- + CFG parallelism (2 GPUs): 1.9x → ~3.0s
- + DiT caching: 5.5x → ~1.0s
- + torch.compile + CUDA graphs: 8.9x → ~0.64s (H100)
- + DreamZero-Flash (1 denoising step): 150ms (GB200 only)

**Evaluation (paper, Section 4):**
- AgiBot seen tasks: 10 tasks × 8 rollouts × 4 robots = 80 rollouts per checkpoint
- AgiBot unseen tasks: 10 tasks × 8 rollouts × 4 robots = 80 rollouts per checkpoint
- DROID: 40 tasks × 2 rollouts = 80 rollouts per checkpoint
- Post-training: 10 rollouts per task
- Results reported as mean task progress ± standard error

### Proposed Design

**Factors:**
- 12 tasks (simulated manipulation tasks in HilGym)
- 3 conditions (e.g., zero-shot pretrained, LoRA 5K steps, LoRA 20K steps)
- 3 training seeds per condition (different LoRA init / data shuffling)
- 20 rollouts per (task, condition, seed) cell

Each rollout uses a different diffusion noise seed inherently (the denoising process samples fresh noise each call), so the 20 rollouts capture inference stochasticity. The 3 training seeds capture training stochasticity.

**Total rollouts:** 12 tasks × 3 conditions × 3 seeds × 20 rollouts = **2,160 rollouts**

### Phase 1: Training

| Condition | Steps | Runs (3 seeds) | Hours per run (8x H100) | Total hours |
|---|---|---|---|---|
| Zero-shot pretrained | 0 | 0 | 0 | 0 |
| LoRA 5K steps | 5,000 | 3 | ~3 hrs | 9 hrs |
| LoRA 20K steps | 20,000 | 3 | ~12 hrs | 36 hrs |
| **Training total** | | **6 runs** | | **45 hrs** |

Reference: The paper trains LoRA for 5K steps on AgiBot (agibot_training.sh). Full pretraining is 100K steps at batch 128. Post-training is 50K steps per task.

**Cost on p4de.24xlarge (8x A100 80GB):**
- On-demand ($40/hr): 45 hrs × $40 = **$1,800**
- Spot (~$14/hr): 45 hrs × $14 = **$630**

### Phase 2: Inference / Evaluation

**Per rollout timing (2x H100, DiT caching, no torch.compile):**
- ~5 action chunks per manipulation task (reach → approach → grasp → lift → place)
- ~1.0s per chunk with DiT caching (paper Table 1: 5.5x speedup)
- + sim stepping / observation capture: ~2s overhead
- **~7s per rollout**

**Total inference time:**
- 2,160 rollouts × 7s = ~4.2 hours serial
- With 4 parallel servers (8 GPUs): **~1.1 hours wall time**
- 9 checkpoints to evaluate (3 conditions × 3 seeds)
- Per-checkpoint: 720 rollouts ÷ 4 parallel = ~0.35 hours
- **Total: 9 × 0.35 = ~3.2 hours** (+ warmup ~0.5 hrs each = ~7.7 hours)

**Cost on p5.48xlarge (8x H100 80GB):**
- On-demand ($98/hr): 7.7 hrs × $98 = **$755**

### Phase 3: Checkpoint Selection + Final Eval (Optional)

If you want to pick the best checkpoint per (condition, seed) from multiple save points:
- 3 checkpoints per training run × 6 runs = 18 additional checkpoints
- 18 × 0.35 hrs = ~6.3 hours + warmup
- **~$1,000 additional**

### Total Budget

| Phase | Hours | Instance | Rate | Cost (spot/on-demand) |
|---|---|---|---|---|
| Training (6 runs) | 45 hrs | p4de.24xlarge | $14-40/hr | $630-1,800 |
| Eval (9 checkpoints) | 7.7 hrs | p5.48xlarge | $98/hr | $755 |
| Checkpoint sweep (18 extra) | 10 hrs | p5.48xlarge | $98/hr | $980 |
| Buffer (debugging, reruns) | ~20 hrs | mixed | | $1,000 |
| Storage (S3) | | | | $200 |
| **Total** | | | | **$3,565-4,735** |

~$4-5K of the $50K budget. Leaves $45K+ for scaling up, adding conditions, running real-robot experiments, or trying the 5B model variant.

### Statistical Analysis

With 3 training seeds × 20 rollouts per cell:
- Report: mean task progress ± SEM across seeds (matching paper's convention)
- Per-seed analysis: 20 rollouts gives ~10% SEM for a 50% success rate task
- Across-seed analysis: 3 seeds captures training variance, report as error bars
- Significance testing: permutation test or bootstrap across seeds, not individual rollouts (seeds are the independent unit)
- Detectable effect size: ~25-30 percentage points between conditions (power=0.80, α=0.05, n=3 seeds)
- For smaller effects, increase to 5 seeds (+$1,500 training, +$500 eval)
