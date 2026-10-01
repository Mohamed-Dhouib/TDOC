# Training `G_theta` (Crop Quality Network)

Trains `G_theta`, the crop quality network used to reject ill-defined text crops during generation.

## Step 1) Build `.pt` Training Data

This step reads preprocessed CSVs and writes `.pt` samples.

### Config: `training_crop_quality/prepare_data.yaml`

Required fields:

| Key | Meaning |
|-----|---------|
| `datasets_main_path` | Root prefix containing dataset subfolders |
| `datasets_upsample_factor` | Per-dataset sampling factor |
| `total_jobs` | Number of hash partitions for parallel preparation |
| `target_folder_name` | Output folder for `.pt` files |

- You must use the same folder path `target_folder_name` later in `training_crop_quality/fraud.yaml` as `dataset_name_or_path`.
- `datasets_main_path` is the parent folder containing preprocessed dataset folders.
- Each key in `datasets_upsample_factor` must be present as a folder under `datasets_main_path`.

Example:

```yaml
datasets_main_path: "/data/tdoc/"
datasets_upsample_factor:
  dataset_1: 1
  dataset_2: 3
target_folder_name: "/data/crop_quality_data"
```

### Run data preparation

```bash
TOTAL_JOBS=32  # set as high as your machine/cluster can handle
python training_crop_quality/prepare_data.py --mod 0
python training_crop_quality/prepare_data.py --mod 1
# ...
python training_crop_quality/prepare_data.py --mod $((TOTAL_JOBS-1))
```

Each `--mod` run writes its partition to `target_folder_name`.

## Step 2) Train `G_theta`

### Config: `training_crop_quality/fraud.yaml`

Main fields:

| Key | Meaning |
|-----|---------|
| `pretrained_model_name_or_path` | Optional initial checkpoint path (`""` for scratch) |
| `dataset_name_or_path` | Path to `.pt` folder from step 1 (same value as `target_folder_name`) |

Example:

```yaml
pretrained_model_name_or_path: ""
dataset_name_or_path: "/data/crop_quality_data"
```

Train:

```bash
python training_crop_quality/train.py --config training_crop_quality/fraud.yaml --exp_version my_run
```

Checkpoints are saved under `result/`.

## Use in Generation

Set `pretrained_model_name_or_path` in `training_crop_quality/fraud.yaml` to the trained checkpoint path.
Then set generation config `config_path_crop_quality` to that YAML file.
