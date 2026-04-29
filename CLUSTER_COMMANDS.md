# DreamZero Cluster Commands: DROID LoRA Finetune + Eval

Run these on your A100 node in order.

## Step 1: Download DROID subset (~10-15GB)

```bash
huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data \
    --repo-type dataset \
    --local-dir ./data/droid_lerobot \
    --include "meta/*" "data/chunk-000/*" "videos/chunk-000/*"
```

## Step 2: LoRA finetune (4x A100, 500 steps)

Make sure CUDA_HOME is set first:

```bash
module load cuda
export CUDA_HOME=$CUDA_ROOT
```

Then run training:

```bash
export HYDRA_FULL_ERROR=1

torchrun --nproc_per_node 4 --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/droid_relative \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=500 \
    training_args.warmup_ratio=0.05 \
    output_dir=./checkpoints/droid_lora_test \
    per_device_train_batch_size=1 \
    max_steps=500 \
    weight_decay=1e-5 \
    save_total_limit=5 \
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
    droid_data_root=./data/droid_lerobot \
    dit_version=./checkpoints/Wan2.1-I2V-14B-480P \
    text_encoder_pretrained_path=./checkpoints/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=./checkpoints/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=./checkpoints/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
    tokenizer_path=./checkpoints/umt5-xxl \
    pretrained_model_path=./checkpoints \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
```

## Step 3: Evaluate -- compare base vs finetuned

### 3a. Start inference server with FINETUNED checkpoint

```bash
TORCH_COMPILE_DISABLE=1 torchrun --standalone --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port 8000 \
    --enable-dit-cache \
    --model-path ./checkpoints/droid_lora_test/checkpoint-500
```

### 3b. In another terminal, send test observations

```bash
python test_client_AR.py --host localhost --port 8000 --use-zero-images --num-chunks 3
```

Check the output: action shape, range, and timing.
Server saves predicted videos to `checkpoints/droid_lora_test/real_world_eval_gen_*/`.

### 3c. Kill the finetuned server (Ctrl+C), start the BASE checkpoint

```bash
TORCH_COMPILE_DISABLE=1 torchrun --standalone --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port 8000 \
    --enable-dit-cache \
    --model-path ./checkpoints/
```

### 3d. Same test client again

```bash
python test_client_AR.py --host localhost --port 8000 --use-zero-images --num-chunks 3
```

Compare action ranges between base and finetuned.

## What you're validating

| Step | What it proves |
|---|---|
| Step 1 completes | HF download works on your cluster |
| Step 2 completes | Training pipeline runs: data loading, LoRA injection, forward/backward, checkpointing all work on 4x A100 |
| Step 3 action ranges differ | Finetuning changed the model's behavior (DROID actions differ from AgiBot) |
| Step 3 predicted videos | Visual sanity check -- does the model imagine plausible Franka arm motion? |

## SSH tunnel (from your local machine)

To run the test client from your local machine instead of the cluster:

```bash
ssh -L 8000:COMPUTE_NODE:8000 jstale02@login-p03.tufts.edu
```

Replace COMPUTE_NODE with whatever node SLURM assigned (check with `hostname` on the cluster).
