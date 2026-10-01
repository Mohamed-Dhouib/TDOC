# Training `F_theta` (Crop Similarity Network)

Trains the contrastive similarity model `F_theta`.
Given text crops, `F_theta` helps generation choose similar source/target crop pairs.

Pretrained weights are available at `https://drive.google.com/drive/folders/1L1vPIlkrlWp-e3Xjb6tqj8Ql3NtDSvUO?usp=sharing`.
The pretrained model generalizes well across many document types and quality levels. Retraining is only needed for highly specific domains, such as mostly handwritten documents or non-Latin scripts.

## Required Dataset Format

Before running this training, you need to run preprocessing (see [`../generation_pipeline/preprocessing/README.md`](../generation_pipeline/preprocessing/README.md)).
This training reads CSV files produced by that preprocessing step.

Expected structure after preprocessing:

```text
<datasets_main_path>/
|-- dataset_1/
|   |-- images/
|   |-- merged_jsons/
|   `-- grouped_csvs/
|-- dataset_2/
|   |-- images/
|   |-- merged_jsons/
|   `-- grouped_csvs/
`-- dataset_3/
    |-- images/
    |-- merged_jsons/
    `-- grouped_csvs/
```

## Config (`training_contrastive/fraud.yaml`)

Main fields:

| Key | Meaning |
|-----|---------|
| `pretrained_model_name_or_path` | Optional initial checkpoint path (`""` for scratch) |
| `datasets_main_path` | Root prefix containing dataset subfolders |
| `datasets_upsample_factor` | Per-dataset sampling factor (float or int) |
| `small_datasets` | Dataset keys used for targeted negative-source sampling |

Path resolution for dataset loading:

- `datasets_main_path` is the parent folder containing all datasets.
- Each key in `datasets_upsample_factor` must be present as a folder under `datasets_main_path`.

Example:

```yaml
datasets_main_path: "/data/tdoc/"
datasets_upsample_factor:
  dataset_1: 1
  dataset_2: 3
```

Code behavior for `small_datasets`:

- During negative sampling, there is a 20% chance that sampling occurs only from indices belonging to these datasets. This aims to ensure variety of negative samples.
- If empty, negatives are sampled from the full pool only.

Data-specific parameters are in `training_contrastive/data.yaml`.

## Train

Train:

```bash
python training_contrastive/train.py --config training_contrastive/fraud.yaml --exp_version my_run
```

Checkpoints are saved under `result/`.

## Use in Generation

Set `pretrained_model_name_or_path` in `training_contrastive/fraud.yaml` to the trained checkpoint path.
Then set generation config `config_path_crop_embed` to that same YAML file.
