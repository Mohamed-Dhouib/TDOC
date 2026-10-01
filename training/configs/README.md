# Detector Config Reference

This document explains the YAML configuration files used by detector training, inference, and evaluation.

Config files are in `training/configs/`.

You can use the provided config files, or create your own configs for pretraining, finetuning, inference, and evaluation.
This document is the reference for custom configs.

## File Naming

Each model has one pretraining config and finetuning configs:

- `{model}_pretraining.yaml`
- `{model}_finetuning_findit.yaml`
- `{model}_finetuning_finditagain.yaml`
- `{model}_finetuning_rtm.yaml`

Supported model names:

- `DTD`
- `ffdn`
- `catnet`
- `ascformer`
- `psccnet`

## Required Keys for Training

Top-level keys used by `training/train.py` and `training/lightning_module.py`:

| Key | Type | Meaning |
|-----|------|---------|
| `model_name` | string | Detector architecture to build |
| `seed` | int | Random seed for training |
| `result_path` | string | Root output directory for logs and checkpoints |
| `pretrained_model_name_or_path` | string | Optional checkpoint to load before training (`""` means no extra load) |
| `dataset_name_or_path` | list | Dataset pairs format: `[[images_path, masks_path], ...]` |
| `train_batch_size` | int | Training batch size per process |
| `input_size` | list[int, int] | Input size `[H, W]` |
| `lr` | float | Optimizer learning rate |
| `warmup_steps` | int | Warmup steps for cosine scheduler |
| `scheduler_num_training_samples_per_cycle_and_epoch` | int | If `-1`, use full dataset size; otherwise force this value |
| `max_epochs` | int | Number of training epochs |
| `num_workers` | int | DataLoader worker processes |
| `precision` | string or int | Trainer precision (for example `bf16-mixed`, `32`) |
| `accumulate_grad_batches` | int | Gradient accumulation steps |
| `check_point_every_n_step` | bool | Enable periodic checkpoint callback |
| `steps_per_checkpoint` | int | Save period (steps) when periodic checkpoints are enabled |
| `compile_model` | bool | Enable `torch.compile` path in `train.py` |
| `loss_pos_weight` | float | Positive-class weight for segmentation focal loss |
| `class_coef` | float | Weight multiplier for image-level classification loss |

Coverage note:

- The table above covers all top-level keys currently present in `training/configs/*.yaml`.
- The optional and legacy sections below cover extra keys consumed by code paths even if not present in every YAML.

Optional keys:

| Key | Type | Meaning |
|-----|------|---------|
| `persistent_workers` | bool | Passed to DataLoader (`False` if omitted) |
| `finetuning` | bool | Metadata flag in current code path |
| `min_lr` | float | Optional minimum LR clamp for cosine scheduler |
| `resume_from_checkpoint_path` | string | Optional path passed to Lightning `fit(..., ckpt_path=...)` |

Legacy key present in configs:

| Key | Status |
|-----|--------|
| `gradient_clip_val` | Present in YAMLs but not currently used in `training/train.py` trainer arguments |

## Data Augmentation Block

`data_aug_config` is used by `training/model.py` during input preparation.

```yaml
data_aug_config:
  enabled: true
  prob_disable: 0.5
  probs:
    spatial: 0.25
    visual: 0.16
    flip: 0.05
  compression:
    max_num: 3
    quality_range: [75, 100]
```

Fields:

- `enabled`: master augmentation switch.
- `prob_disable`: probability to disable augmentation for a sample (optional).
- `probs.spatial`, `probs.visual`, `probs.flip`: probabilities for augmentation groups.
- `compression.max_num`, `compression.quality_range`: JPEG compression augmentation settings.

## Dataset Path Format

`dataset_name_or_path` must be either:

- One pair: `["/path/to/images", "/path/to/masks"]`
- Multiple pairs: `[[ "/path1/images", "/path1/masks" ], [ "/path2/images", "/path2/masks" ]]`

Example:

```yaml
dataset_name_or_path:
  - ["/data/findit/images", "/data/findit/masks"]
  - ["/data/finditagain/images", "/data/finditagain/masks"]
```

## Pretraining vs Finetuning

Typical pretraining settings:

- `pretrained_model_name_or_path: ""` (or a custom checkpoint if desired)
- larger `loss_pos_weight` in provided examples
- `finetuning` omitted or `false`

Typical finetuning settings:

- `pretrained_model_name_or_path` points to a pretrained checkpoint
- `finetuning: true`
- lower learning rate and shorter warmup in provided examples

## Eval and Infer Config Usage

`training/eval.py` and `training/infer.py` read these keys from the selected config:

- `model_name`
- `input_size`
- `pretrained_model_name_or_path`

Both scripts support `--weights-path` to override `pretrained_model_name_or_path` at runtime.
This means you can create lightweight custom eval/infer configs that contain these keys.

## Environment Variable Overrides (Train Only)

`training/train.py` also supports:

- `DATASET_NAME_OR_PATH` override for `dataset_name_or_path`
- `MODEL_NAME` override for `model_name`
- `PRETRAINED_PATH` override for `pretrained_model_name_or_path`
