# Generation Pipeline

Generates synthetic document forgeries (copy-move, splicing, insertion, inpainting, coverage) guided by:

- `F_theta`: crop similarity model
- `G_theta`: crop quality model

## Prerequisites

1. Preprocessed datasets (see [`generation_pipeline/preprocessing/README.md`](preprocessing/README.md) first).
2. Trained weights for `F_theta` and `G_theta` (or use provided pretrained weights: [`F_theta`](https://drive.google.com/drive/folders/1L1vPIlkrlWp-e3Xjb6tqj8Ql3NtDSvUO?usp=sharing), [`G_theta`](https://drive.google.com/drive/folders/1r0YtzJt1D6o_6zzBLsZzKKwas-8L8jnm?usp=sharing)).

## Dataset Structure

`datasets_main_path` must point to a parent folder that contains multiple dataset folders.
This structure is the output of preprocessing (see [`generation_pipeline/preprocessing/README.md`](preprocessing/README.md)).

```text
<datasets_main_path>/
|-- dataset_1/
|   |-- images/
|   |-- images_black/      # optional
|   |-- merged_jsons/
|   `-- grouped_csvs/
|-- dataset_2/
|   |-- images/
|   |-- images_black/      # optional
|   |-- merged_jsons/
|   `-- grouped_csvs/
`-- dataset_3/
    |-- images/
    |-- images_black/      # optional
    |-- merged_jsons/
    `-- grouped_csvs/
```

Only dataset folder names present as keys in `datasets_upsample_factor` are used for generation.
See the `datasets_upsample_factor` explanation in the configuration section below.

## Configuration (`config.yaml`)

Main fields to set:

| Key | What to set |
|-----|-------------|
| `datasets_main_path` | Root folder containing all preprocessed dataset subfolders |
| `config_path_crop_embed` | Path to `training_contrastive/fraud.yaml` with `pretrained_model_name_or_path` set |
| `config_path_crop_quality` | Path to `training_crop_quality/fraud.yaml` with `pretrained_model_name_or_path` set |
| `output_folder` | Output directory for generated data |
| `datasets_upsample_factor` | Mapping `{dataset_key: upsampling_factor}` controlling generation volume per dataset |


- `datasets_main_path` is the parent folder containing all preprocessed dataset folders.
- Each key in `datasets_upsample_factor` must be present as a folder under `datasets_main_path`.

Example:

```yaml
datasets_main_path: "/data/tdoc/"
grouped_csvs_folder: "grouped_csvs"
datasets_upsample_factor:
  dataset_1: 1
  dataset_2: 3
```

Auxiliary model config paths:

- `config_path_crop_embed` must point to `training_contrastive/fraud.yaml` with `pretrained_model_name_or_path` set.
- `config_path_crop_quality` must point to `training_crop_quality/fraud.yaml` with `pretrained_model_name_or_path` set.

Useful optional fields:

| Key | Default | Description |
|-----|---------|-------------|
| `datasets_to_expand` | `[]` | Dataset keys that need bbox expansion |
| `threshold_crop_quality` | `0.5` | `G_theta` threshold |
| `threshold_crop_similarity` | `0.78` | `F_theta` threshold |
| `grouped_csvs_folder` | `grouped_csvs` | Folder containg grouped text segments (needs to be set the same as the preprocessing script)|
| `max_manipulations_per_doc` | `5` | Maximum manipulations per source sample |
| `save_in_a_single_folder` | `true` | Save all outputs under `output_folder/images` and `output_folder/masks` |

## Run Generation

Single process:

```bash
python generation_pipeline/generate_data.py --config generation_pipeline/config.yaml
```

Parallel run:

```bash
MAX_JOBS=32  # set as high as your machine/cluster can handle
python generation_pipeline/generate_data.py --config generation_pipeline/config.yaml --job_index 0 --max_jobs $MAX_JOBS
python generation_pipeline/generate_data.py --config generation_pipeline/config.yaml --job_index 1 --max_jobs $MAX_JOBS
# ...
python generation_pipeline/generate_data.py --config generation_pipeline/config.yaml --job_index $((MAX_JOBS-1)) --max_jobs $MAX_JOBS
```

`job_index` must be in `[0, MAX_JOBS-1]`.

## Validate Outputs and Regenerate Corrupted Samples

1. Check generated pairs:

```bash
python generation_pipeline/check_images.py --root <output_folder> --workers 16 --only-stats
```

2. If issues exist, run without `--only-stats` to delete invalid pairs:

```bash
python generation_pipeline/check_images.py --root <output_folder> --workers 16
```

3. Build resume map and regenerate missing items:

```bash
python generation_pipeline/create_resume_map.py --config generation_pipeline/config.yaml --out_all resume_map.json
python generation_pipeline/generate_data.py --config generation_pipeline/config.yaml --resume_map resume_map.json
```

## Output

With `save_in_a_single_folder: true`, generated files are in:

```text
<output_folder>/
|-- images/
`-- masks/
```

This output can be used directly for training detectors using scripts in `training/`.
