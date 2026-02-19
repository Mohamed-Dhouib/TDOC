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

- Generated pretraining data TDoc-2.8M (Hugging Face): `https://huggingface.co/datasets/MohamedDhouib1/TDoc-2.8M`
- Finetuning data root: `https://drive.google.com/drive/folders/1jtB9rmdUww_zJ6jGg5qrjujgRfLVv3vV?usp=sharing`
- Evaluation data root: `https://drive.google.com/drive/folders/1HkzytEz8BM7p2QxiyKZYQWSYtZbW_HMu?usp=sharing`

## 1) Model Initialization and Backbone Weights

To reproduce results in the paper, models needs to be initialized in the same way.
First download initialization weights from: `https://drive.google.com/drive/folders/1zEYzmVGcWopJhCOjeEosSqb5UL2ZumwD`

Then set `pretrained_model_name_or_path` in pretraining configs for DTD and FFDN to the corresponding downloaded checkpoint path.
For ASCFormer, CATNet, and PSCC-Net, initialization uses the fixed model-local files listed below.

| Model | Initialization location in code | Default local files used by model code |
|-------|----------------------------------|-----------------------------------------|
| ASCFormer | `training/models/ascformer/ascformer_model.py` | `training/models/ascformer/weights/mit_b2_20220624-66e8bf70.pth` |
| CATNet | `training/models/catnet/model.py` | `training/models/catnet/weights/CAT_full_v2.pth` |
| PSCC-Net | `training/models/psccnet/model.py` | `training/models/psccnet/checkpoint/HRNet_checkpoint/HRNet.pth`, `training/models/psccnet/checkpoint/NLCDetection_checkpoint/NLCDetection.pth`, `training/models/psccnet/checkpoint/DetectionHead_checkpoint/DetectionHead.pth` |


For all models, when `pretrained_model_name_or_path` is set in config, startup should print:

- `Loading weights from ...`
- `Weights loaded`

## 2) Pretraining on Generated Data

Open the pretraining config for the chosen model (for example `training/configs/dtd_pretraining.yaml`).

Set:

- `dataset_name_or_path` as a list of `[images_folder, masks_folder]` that point to the image and mask folders for the TDoc-2.8M dataset (`https://huggingface.co/datasets/MohamedDhouib1/TDoc-2.8M`). Note: run the script that decompresses the downloaded Hugging Face data (provided along the shards) so `images/` and `masks/` folders are created.
- `pretrained_model_name_or_path`:
  - DTD and FFDN: set this to the downloaded weights path from the Drive link above.
  - ASCFormer, CATNet, and PSCC-Net: keep this empty for backbone initialization and place weights at the fixed local paths shown in section 1.

Example:

```yaml
pretrained_model_name_or_path: "/path/to/downloaded/model_weights.pth"
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
- Paper results were produced with `jpegio==0.2.4` for JPEG quantization-table extraction, so to ensure 100 percent reproducibility, install `jpegio==0.2.4`.
- The code also provides a Pillow fallback as an easier alternative, which can be faster to run and should give same results. Jpegio is not included in the environment.yaml file, so you need to install it yourself.
