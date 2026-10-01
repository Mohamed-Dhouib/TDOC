# Preprocessing

This pipeline prepares each raw dataset in the format required for synthetic tampered document generation and for training the two auxiliary networks. Skip this step if you only want to finetune pretrained detector models on your own dataset (or on a public dataset).

## Input Format Per Dataset

You need to preprocess each dataset independently and keep all dataset folders under the same parent folder.

Initial structure before preprocessing:

```text
<datasets_main_path>/
|-- dataset_1/
|   `-- images/
|-- dataset_2/
|   `-- images/
`-- dataset_3/
    `-- images/
```

Each dataset folder must contain a single folder containing all images:

```text
<dataset_folder>/
`-- images/
```

## Steps

### 1) OCR extraction

Run script to apply OCR per image and write one JSON per image to `merged_jsons/`.

```bash
python generation_pipeline/preprocessing/ocr_pipeline_single_step.py \
    --image-dir <dataset_folder>/images \
    --output-dir <dataset_folder>/merged_jsons \
    --ocr-backend tesseract \
    --num-workers 4 \
    --skip-existing
```

OCR backends:

- `--ocr-backend google`: uses Google Cloud Vision.
  - The script does not hardcode a credential file path.
  - You can provide credentials with `GOOGLE_APPLICATION_CREDENTIALS` or any valid alternative.
- `--ocr-backend tesseract`: uses local Tesseract (`--tesseract-lang` supported, default `eng`).
  - If you use this backend, you need to install `pytesseract`.

### 2) Grayscale copies (Optional)

Run command to create gray copies of each color image (saved in `images_black/`).

```bash
python generation_pipeline/preprocessing/transform_to_black.py \
    --dataset_folder <dataset_folder>
```

This aims to add variability to the dataset.

### 3) Build unified OCR CSV format

Converts OCR JSONs into per-image CSV files (saved in `merged_jsons/`).

```bash
python generation_pipeline/preprocessing/create_cvs.py \
    --dataset_folder <dataset_folder>
```

### 4) Create parquet shards

`prepare_grouping.py` reads all `merged_jsons/*.csv`, concatenates them, then splits rows into `NUM_JOBS` parquet shards via `np.array_split`. These shard files are then used to group text segments by `len_text`, `Width`, and `Height` (necessary for splicing).

```bash
NUM_JOBS=32  # set as high as your machine/cluster can handle
python generation_pipeline/preprocessing/prepare_grouping.py \
    --dataset_folder <dataset_folder> \
    --num_jobs $NUM_JOBS
```

### 5) Process each shard into grouped CSV chunks

`parallel_process_groups.py`:

- loads `grouped_csvs/merged_<task_id>.parquet`
- groups rows by `(len_text, Width, Height)`
- keeps only groups that satisfy:
  - `len(group) >= min_el`
  - at least 2 distinct `filename`
  - at least 2 distinct `text`
- writes chunked CSVs into `grouped_csvs/<group_name>/`

Run once for each `TASK_ID` in `[0, NUM_JOBS-1]`:

```bash
NUM_JOBS=32
GROUP_WORKERS=8  # set as high as your machine/cluster can handle
python generation_pipeline/preprocessing/parallel_process_groups.py \
    --dataset_folder <dataset_folder> \
    --min_el 4 \
    --task_id 0 \
    --num_workers $GROUP_WORKERS
```

Repeat with `--task_id 1`, `--task_id 2`, ... up to `--task_id $((NUM_JOBS-1))`.

Optional flag: add `--delete_shard_after` to delete `grouped_csvs/merged_<task_id>.parquet` after successful processing. Default behavior keeps shard parquet files.

## Final Dataset Format

After preprocessing, each dataset folder should look like:

```text
<dataset_folder>/
|-- images/
|-- images_black/      # only if step 2 was run
|-- merged_jsons/      # OCR JSON + per-image CSV
`-- grouped_csvs/      # grouped CSV chunks (and parquet shards if not deleted)
```

This final format is the expected input for `generation_pipeline/generate_data.py` and for training the two auxiliary networks. The OCR jsons can be deleted but we recommend keeping them.
