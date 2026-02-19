# Leveraging Contrastive Learning for a Similarity-Guided Tampered Document Data Generation Pipeline

Official repository for the paper "Leveraging Contrastive Learning for a Similarity-Guided Tampered Document Data Generation Pipeline".

## Resources

- Generated pretraining dataset TDoc-2.8M (Hugging Face): `https://huggingface.co/datasets/MohamedDhouib1/TDoc-2.8M`
- Pretrained and finetuned weights: `https://drive.google.com/drive/folders/1txtCMr2ZGq1LYHclrMip3I3tOmeHN5IH?usp=sharing`
- Finetuning datasets (FindIt, FindItAgain, RTM): `https://drive.google.com/drive/folders/1jtB9rmdUww_zJ6jGg5qrjujgRfLVv3vV?usp=sharing`
- Evaluation datasets (FindIt, FindItAgain, RTM): `https://drive.google.com/drive/folders/1HkzytEz8BM7p2QxiyKZYQWSYtZbW_HMu?usp=sharing`

## Overview

This repository provides tools to:

1. Train auxiliary models used during generation (`F_theta` and `G_theta`).
2. Preprocess raw document datasets (OCR, CSV conversion, grouping).
3. Generate synthetic document forgeries (copy-move, splicing, insertion, inpainting, coverage) guided by auxiliary models.
4. Train, finetune, infer, and evaluate five document-tampering detectors: DTD, FFDN, CATNet, ASCFormer, and PSCC-Net.

## Documentation

Each main folder has its own documentation:

| Folder | Document | What it covers |
|--------|----------|----------------|
| `generation_pipeline/preprocessing/` | [`README.md`](generation_pipeline/preprocessing/README.md) | Dataset preprocessing required before generation |
| `generation_pipeline/` | [`README.md`](generation_pipeline/README.md) | Running the synthetic tampering generation pipeline |
| `training_contrastive/` | [`README.md`](training_contrastive/README.md) | Training `F_theta` (crop similarity model) |
| `training_crop_quality/` | [`README.md`](training_crop_quality/README.md) | Training `G_theta` (crop quality model) |
| `training/` | [`README.md`](training/README.md) | Detectors training, finetuning, inference, and evaluation |
| `training/configs/` | [`README.md`](training/configs/README.md) | Detectors config documentation |
| repository root | [`reproduction_guide.md`](reproduction_guide.md) | End-to-end paper reproduction instructions |

## Repository Structure

```text
.
|-- environment.yaml
|-- generation_pipeline/
|   `-- preprocessing/
|-- training_contrastive/
|-- training_crop_quality/
`-- training/
    |-- configs/
    `-- models/
```

## Installation

```bash
conda env create -f environment.yaml
conda activate fraud
```

## End-to-End Workflow

This repository can be used in three ways:
- End-to-end pipeline: follow steps 1 to 5.
- Data generation only: follow steps 1 to 3 (step 2 is optional).
- Finetuning and evaluating document tampering detectors with the provided pretrained weights: follow steps 4 and 5.

### 1) Preprocess each dataset

Run preprocessing separately for each dataset folder so each one has the final format expected by generation.
See [`generation_pipeline/preprocessing/README.md`](generation_pipeline/preprocessing/README.md).

### 2) (Optional) Train auxiliary models

You can train:
- `F_theta` in [`training_contrastive/README.md`](training_contrastive/README.md)
- `G_theta` in [`training_crop_quality/README.md`](training_crop_quality/README.md)

We provide pretrained weights for both auxiliary networks: [F_theta crop similarity](https://drive.google.com/drive/folders/1L1vPIlkrlWp-e3Xjb6tqj8Ql3NtDSvUO?usp=sharing) and [G_theta crop quality](https://drive.google.com/drive/folders/1r0YtzJt1D6o_6zzBLsZzKKwas-8L8jnm?usp=sharing). They generalize well across many document types and quality levels. Retraining is usually only needed for highly specific domains, such as mostly handwritten documents or non-Latin scripts.
### 3) Generate synthetic forgeries

Run generation to produce manipulated images and masks.
See [`generation_pipeline/README.md`](generation_pipeline/README.md).

### 4) Train or finetune detectors

Training and finetuning instructions are in [`training/README.md`](training/README.md).
You can start from the provided pretrained detector weights in [tampering localization models](https://drive.google.com/drive/folders/14M5GsbJTIXIJ9jlhZkH4bSOa43Uj4cwp?usp=sharing), then finetune on your own dataset.

### 5) Evaluation

See [`training/README.md`](training/README.md) for eval and inference commands.

## Reproducing Paper Results

See [`reproduction_guide.md`](reproduction_guide.md) for how to reproduce the results in our paper.

## Note on `jpegio`

Paper results were produced with `jpegio==0.2.4` for JPEG quantization-table extraction.  
The code also provides a Pillow fallback as an easier alternative, which can be faster to run and should give same results. Jpegio is not included in the environment.yaml file, so you need to install it yourself.
