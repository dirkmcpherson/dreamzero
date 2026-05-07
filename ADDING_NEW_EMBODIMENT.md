# Adding a new embodiment to DreamZero

Walking checklist for adding a new robot embodiment for LoRA fine-tuning of
Wan2.1-I2V-14B. Discovered the hard way during gen3_lite integration on
2026-05-07; following this list top-to-bottom avoids playing whack-a-mole
with one error per launch.

## In this repo (`~/workspace/dreamzero/`)

1. **`groot/vla/data/schema/embodiment_tags.py`** — add the enum member
   ```python
   GEN3_LITE = "gen3_lite"
   """Kinova Gen3 Lite 6-DOF arm with integrated 2-finger gripper."""
   ```

2. **`groot/vla/configs/model/dreamzero/transform/base.yaml`** — add to
   `embodiment_tag_to_projector_index`. **Reuse an existing index from a
   similar embodiment** so the pretrained projector head loads (gen3_lite
   reuses `21` = `real_panda_single_arm`, both single-arm + parallel gripper).
   ```yaml
   gen3_lite: 21  # share projector with real_panda_single_arm
   ```

3. **`groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`** —
   add `modality_config_<name>` and `transform_<name>` blocks, then register
   them in `modality_configs`, `transforms`, `metadata_versions`, and `fps`.

4. **`groot/vla/configs/data/dreamzero/<name>_relative.yaml`** — new file
   containing the mixture spec pointing to `${<name>_data_root}`.

5. **`groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`** — add an
   `elif` branch in **both** the try block (~line 110) **and** the except
   block (~line 132) of the `collate` function, before each
   `else: raise ValueError`. The collate function has a hardcoded prompt-
   template tree per embodiment; missing branches raise
   `Embodiment ID X not supported`.
   ```python
   elif elem["embodiment_id"] == embodiment_tag_mapping[EmbodimentTag.GEN3_LITE.value]:
       processed_item = "A multi-view video shows that a robot " + processed_item.lower() + " The video is split into two views: ..."
   ```

6. **`scripts/data/convert_lerobot_to_gear.py`** — add the embodiment string
   to `VALID_EMBODIMENT_TAGS`.

7. **`scripts/train/<name>_training.sh`** — torchrun launcher modeled on
   `yam_training.sh` or `gen3_lite_training.sh`.

## Data-side prep (LeRobot v2 dataset)

The DreamZero loader expects v2 format with GEAR meta files. After running
`scripts/data/convert_lerobot_to_gear.py` on a v2 dataset, verify:

- **`meta/modality.json`** — annotation `original_key` MUST be `task_index`
  (LeRobot v2's numeric column), NOT `annotation.<something>` (which doesn't
  exist as a parquet column). The DreamZero loader auto-detects numeric
  columns and resolves them via `tasks.jsonl`. The converter has been
  patched to do this automatically as of 2026-05-07.
- **`meta/tasks.jsonl`** — must contain a real task description, not `""`.
  Auto-recorded datasets often leave this empty.
  ```bash
  echo '{"task_index": 0, "task": "<your task description>"}' > meta/tasks.jsonl
  ```

## Order of errors you avoid by following this

If you skip step 2: `KeyError: 'gen3_lite'` in `embodiment_tag_mapping`.
If you skip step 5: `ValueError: Embodiment ID 21 not supported` in collate.
If you skip data-side prep: `KeyError: 'annotation.task'` from pandas during
the dataloader's `get_language` call.
