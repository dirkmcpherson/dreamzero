# Preparing a LeRobot dataset for DreamZero fine-tuning

DreamZero's `ShardedLeRobotMixtureDataset` expects **LeRobot v2 format**
(per-episode parquet + per-episode MP4) plus three **GEAR metadata files**
(`embodiment.json`, `modality.json`, `relative_stats_dreamzero.json`).

LeRobot 0.4.3 records datasets in **v3 format** (packed shards), and the
documented `convert_dataset_v30_to_v21` helper does NOT exist in 0.4.3 —
we wrote our own converter in the lerobot-ros repo.

## Full pipeline

```bash
# 1. Record (v3 format, packed shards)
python record_kinova_data_teleoperated.py --robot gen3_lite --cameras --camera-set wrist
# -> data/lerobot/<name>

# 2. Convert v3 -> v2 (per-episode parquet + sliced MP4s)
python convert_v3_to_v2.py <name>
# -> data/lerobot/<name>_v2

# 3. Add GEAR meta files
python ~/workspace/dreamzero/scripts/data/convert_lerobot_to_gear.py \
    --src data/lerobot/<name>_v2 \
    --embodiment-tag gen3_lite \
    --task-key annotation.task

# 4. Set the task description (auto-recorded datasets leave it empty)
echo '{"task_index": 0, "task": "<your description>"}' \
    > data/lerobot/<name>_v2/meta/tasks.jsonl

# 5. rsync to the cluster
rsync -azvP data/lerobot/<name>_v2 \
    jstale02@<host>:/cluster/tufts/shortlab/jstale02/lerobot_data/
```

## Notes

- Step 2 uses `imageio_ffmpeg.get_ffmpeg_exe()` (the bundled binary) rather
  than the system ffmpeg — system ffmpeg isn't installed on this machine.
- The v3→v2 converter writes a v2 `info.json` with
  `data_path: "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"`.
- Step 3's converter at `scripts/data/convert_lerobot_to_gear.py` is patched
  to set `modality.json`'s annotation `original_key` to `task_index` (the
  LeRobot v2 numeric column). The DreamZero loader auto-detects numeric
  columns and resolves them via `tasks.jsonl`.

## See also

- `ADDING_NEW_EMBODIMENT.md` for the source-code edits required to register
  a new embodiment.
