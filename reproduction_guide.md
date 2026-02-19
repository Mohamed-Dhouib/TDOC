# Reproducing the Paper Results

Step-by-step guide to reproduce the full training pipeline.
Our experiments were run with multi-GPU distributed training for pretraining and single-GPU finetuning.

## 0) Environment

```bash
conda env create -f environment.yaml
conda activate fraud
```

For exact paper reproduction, we recommend installing `jpegio==0.2.4`.

## 0.1) Download Provided Data

To reproduce results, download the data we provide before training:

- Generated pretraining data (Hugging Face): `https://huggingface.co/datasets/MohamedDhouib1/TDoc-2.8M`
- Main Drive folder (TDoc): `https://drive.google.com/drive/folders/1_xQ1ti6--e7QMY5apBUk7db1xX31qe-A?usp=sharing`
- Finetuning data root: `https://drive.google.com/drive/folders/1jtB9rmdUww_zJ6jGg5qrjujgRfLVv3vV?usp=sharing`
- Evaluation data root: `https://drive.google.com/drive/folders/1HkzytEz8BM7p2QxiyKZYQWSYtZbW_HMu?usp=sharing`

## 1) Model Initialization and Backbone Weights

To reproduce results, follow the initialization logic exactly where it is implemented in code.

| Model | Initialization location in code | Default local files used by model code |
|-------|----------------------------------|-----------------------------------------|
| ASCFormer | `training/models/ascformer/ascformer_model.py` | `training/models/ascformer/weights/mit_b2_20220624-66e8bf70.pth` |
| CATNet | `training/models/catnet/model.py` | `training/models/catnet/weights/CAT_full_v2.pth` |
| PSCC-Net | `training/models/psccnet/model.py` | `training/models/psccnet/checkpoint/HRNet_checkpoint/HRNet.pth`, `training/models/psccnet/checkpoint/NLCDetection_checkpoint/NLCDetection.pth`, `training/models/psccnet/checkpoint/DetectionHead_checkpoint/DetectionHead.pth` |
| DTD | `training/models/DTD/dtd.py` | `training/models/DTD/weights/vph_imagenet.pt`, `training/models/DTD/weights/swin_imagenet.ckpt` |
| FFDN | `training/models/ffdn/timm_backbone.py` | no automatic backbone load in current code (starts from scratch unless a checkpoint is explicitly loaded) |

Weights can be stored anywhere on disk.
If you want to initialize from a specific checkpoint, set `pretrained_model_name_or_path` in the config to that path.

Weights download links:

- All weights root: `https://drive.google.com/drive/folders/1txtCMr2ZGq1LYHclrMip3I3tOmeHN5IH?usp=sharing`
- Crop quality network pretrained weights (`G_theta`): `https://drive.google.com/drive/folders/1r0YtzJt1D6o_6zzBLsZzKKwas-8L8jnm?usp=sharing`
- Crop similarity network pretrained weights (`F_theta`): `https://drive.google.com/drive/folders/1L1vPIlkrlWp-e3Xjb6tqj8Ql3NtDSvUO?usp=sharing`
- Tampering localization model weights (pretrained and finetuned): `https://drive.google.com/drive/folders/14M5GsbJTIXIJ9jlhZkH4bSOa43Uj4cwp?usp=sharing`

## 2) Pretraining on Generated Data

Open the pretraining config for the chosen model (for example `training/configs/dtd_pretraining.yaml`).

Set:

- `dataset_name_or_path` as a list of `[images_folder, masks_folder]` pairs.
- `pretrained_model_name_or_path` depending on your intent:
  - `""` for scratch/model-default initialization.
  - a path if you want to load weights before pretraining.

Example:

```yaml
pretrained_model_name_or_path: ""
dataset_name_or_path:
  - ["/path/to/generated/images", "/path/to/generated/masks"]
```

Run:

```bash
python training/train.py --config training/configs/dtd_pretraining.yaml
python training/train.py --config training/configs/ffdn_pretraining.yaml
python training/train.py --config training/configs/catnet_pretraining.yaml
python training/train.py --config training/configs/ascformer_pretraining.yaml
python training/train.py --config training/configs/psccnet_pretraining.yaml
```

Training checkpoints are written under:

`result/<config_name>/<exp_version>/checkpoints/periodic.pth`

## 3) Finetuning

Finetuning datasets used in the paper:

| Dataset | Link |
|---------|------|
| FindIt | `https://drive.google.com/drive/folders/1xDDaylVs0eU-Phb293vW4G9Q2SeltLpO?usp=sharing` |
| FindItAgain | `https://drive.google.com/drive/folders/1EflNAPr8R0KjPsNLhCCCZocQdcbcljgO?usp=sharing` |
| RTM | `https://drive.google.com/drive/folders/1yPNjRCEGsJ8HksylrKfMr2-0DRtll_Ya?usp=sharing` |

For each finetuning config:

- Set `pretrained_model_name_or_path` to the checkpoint you want to start from (for example the `periodic.pth` from pretraining).
- Set `dataset_name_or_path` to the target dataset image/mask pair(s).

Example:

```yaml
pretrained_model_name_or_path: "result/dtd_pretraining/<version>/checkpoints/periodic.pth"
dataset_name_or_path:
  - ["/path/to/findit/images", "/path/to/findit/masks"]
```

Run the corresponding config:

```bash
python training/train.py --config training/configs/dtd_finetuning_findit.yaml
python training/train.py --config training/configs/ffdn_finetuning_findit.yaml
python training/train.py --config training/configs/catnet_finetuning_findit.yaml
python training/train.py --config training/configs/ascformer_finetuning_findit.yaml
python training/train.py --config training/configs/psccnet_finetuning_findit.yaml
```

Repeat with `*_finetuning_finditagain.yaml` and `*_finetuning_rtm.yaml` as needed.

## 4) Evaluation

Evaluation datasets used in the paper:

| Dataset | Link |
|---------|------|
| FindIt | `https://drive.google.com/drive/folders/15hSbwE6JmlJPy3XfcUgiJKWWbcmkaDA8?usp=sharing` |
| FindItAgain | `https://drive.google.com/drive/folders/1CPVQHGd3Lfv-YK4rCMdjwneWoApzB0ZM?usp=sharing` |
| RTM | `https://drive.google.com/drive/folders/1S9SbwblQ32SgDjPazCTQvq9UP3xAOK7Y?usp=sharing` |

```bash
python training/eval.py \
    --config-path training/configs/<model>_finetuning_<dataset>.yaml \
    --weights-path result/<config_name>/<version>/checkpoints/periodic.pth \
    --gt-images-path /path/to/test/images \
    --gt-masks-path /path/to/test/masks
```

`--weights-path` is optional; if omitted, the script uses the path from the config.

## 5) Inference on New Images

```bash
python training/infer.py \
    --input /path/to/images \
    --output-dir /path/to/output \
    --config-path training/configs/<model>_finetuning_<dataset>.yaml \
    --weights-path result/<config_name>/<version>/checkpoints/periodic.pth \
    --threshold 0.6 \
    --save-binary
```

## Notes

- `DATASET_NAME_OR_PATH`, `MODEL_NAME`, and `PRETRAINED_PATH` environment variables can override config values in `training/train.py`.
- To reproduce results faithfully, use the initialization locations listed in section 1.
