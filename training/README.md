# Training / Inference / Evaluation for Document Tampering Detectors

Train, finetune, evaluate, and run inference with five models:
DTD, FFDN, CATNet, ASCFormer, and PSCC-Net.

## Config Files

Configs are in `training/configs/`:
Full config details: [`training/configs/README.md`](configs/README.md).

You can use the provided configs as-is, or create your own config files for pretraining, finetuning, inference, and evaluation.

We provide these config files:

| Stage | Config pattern | Example |
|-------|----------------|---------|
| Pretraining | `{model}_pretraining.yaml` | `dtd_pretraining.yaml` |
| Finetuning (FindIt) | `{model}_finetuning_findit.yaml` | `catnet_finetuning_findit.yaml` |
| Finetuning (FindItAgain) | `{model}_finetuning_finditagain.yaml` | `ffdn_finetuning_finditagain.yaml` |
| Finetuning (RTM) | `{model}_finetuning_rtm.yaml` | `psccnet_finetuning_rtm.yaml` |

Main fields to set:

- `pretrained_model_name_or_path`
  - For pretraining: this depends on your intent.
    - Use `""` for scratch/model-default initialization.
    - Set a path if you want to load weights before pretraining.
  - For finetuning: set the checkpoint path you want to start from.
- `dataset_name_or_path`
  - Format: list of `[images_folder, masks_folder]` pairs.

Example:

```yaml
dataset_name_or_path:
  - ["/data/findit/images", "/data/findit/masks"]
```

You can pass multiple datasets:

```yaml
dataset_name_or_path:
  - ["/data/dataset_1/images", "/data/dataset_1/masks"]
  - ["/data/dataset_2/images", "/data/dataset_2/masks"]
```

## Training

```bash
python training/train.py --config <path_to_config>.yaml --exp_version "my_run"
```

Checkpoints are saved under:

`result/<config_name>/<exp_version>/checkpoints/periodic.pth`

## Evaluation

```bash
python training/eval.py \
    --gt-images-path <path_to_test_images> \
    --gt-masks-path <path_to_test_masks> \
    --config-path <path_to_config>.yaml \
    --weights-path result/<config_name>/<exp_version>/checkpoints/periodic.pth
```

`--weights-path` is optional and overrides the config value if provided.

Optional:

- `--save-images --pred-save-dir <dir>` to save predicted masks.

## Inference

```bash
python training/infer.py \
    --input <image_or_directory> \
    --output-dir <output_directory> \
    --config-path <path_to_config>.yaml \
    --weights-path result/<config_name>/<exp_version>/checkpoints/periodic.pth \
    --threshold 0.6 \
    --save-binary
```

`--weights-path` is optional and overrides the config value if provided.

## Environment Variable Overrides

`training/train.py` can override config values through environment variables:

| Variable | Overrides | Example |
|----------|-----------|---------|
| `DATASET_NAME_OR_PATH` | `dataset_name_or_path` | `export DATASET_NAME_OR_PATH='[[\"/data/imgs\",\"/data/masks\"]]'` |
| `MODEL_NAME` | `model_name` | `export MODEL_NAME=\"dtd\"` |
| `PRETRAINED_PATH` | `pretrained_model_name_or_path` | `export PRETRAINED_PATH=\"/weights/periodic.pth\"` |

## `jpegio` Note

DTD, FFDN, and CATNet use `jpegio` for JPEG quantization-table extraction. The code includes a Pillow fallback intended to give equivalent QT extraction, but for exact paper reproduction install `jpegio==0.2.4`.

## See Also

[`../reproduction_guide.md`](../reproduction_guide.md) for how to reproduce the paper results.
