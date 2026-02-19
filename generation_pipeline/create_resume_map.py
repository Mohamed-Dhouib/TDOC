#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict, Counter
import yaml
from tqdm import tqdm

def collect_csvs_nonrecursive(dirpath: str) -> set[str]:
    """Collect CSVs directly under dirpath (no recursion)."""
    out = set()
    if not os.path.isdir(dirpath):
        return out
    for fn in os.listdir(dirpath):
        if fn.endswith(".csv"):
            out.add(os.path.join(dirpath, fn))
    return out

def base_from_csv(csv_path: str) -> str:
    """basename without extension (used as the prefix for generated files)"""
    return os.path.splitext(os.path.basename(csv_path))[0]

def build_images_index(images_dir: str) -> dict[str, int]:
    """Count generated images by CSV base name."""
    idx = defaultdict(int)
    if not os.path.isdir(images_dir):
        return idx

    with os.scandir(images_dir) as it:
        for e in it:
            if not e.is_file():
                continue
            name = e.name
            if not name.endswith(".png"):
                continue
            uscore = name.rfind("_")
            if uscore <= 0:
                continue
            base = name[:uscore]
            idx[base] += 1
    return dict(idx)

def norm_tag(tag: str) -> str:
    """Normalize a dataset tag to a short lowercase name."""
    if tag is None:
        return ""
    tag = tag.strip().strip("/").split("/")[-1]
    return tag.lower()

def parse_args():
    """Parse command-line arguments for resume map creation."""
    ap = argparse.ArgumentParser(
        "Build a resume map for all datasets using the single-folder images index."
    )
    ap.add_argument("--config", required=True, type=str,
                    help="Path to the same config.yaml used by generation")
    ap.add_argument("--out_all", required=True, type=str,
                    help="Output JSON path for all datasets")
    return ap.parse_args()

def main():
    """Build resume maps for incomplete data generation."""
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    save_in_a_single_folder = bool(cfg.get("save_in_a_single_folder", True))
    if not save_in_a_single_folder:
        raise ValueError("This script expects save_in_a_single_folder=True in config.")

    output_folder = cfg.get("output_folder", "")
    if not output_folder:
        raise ValueError("Missing 'output_folder' in config.")
    images_dir = os.path.join(output_folder, "images")

    datasets_main_path = cfg.get("datasets_main_path", "")
    datasets_upsample_factor = cfg.get("datasets_upsample_factor", {})

    print(f"Indexing generated images in: {images_dir}")
    img_index = build_images_index(images_dir)

    tasks: list[tuple[str, str, int]] = []

    suffix = "/merged_jsons"
    for dataset_tag, factor in datasets_upsample_factor.items():
        factor = int(factor)
        if factor <= 0:
            continue
        dirpath = datasets_main_path + dataset_tag + suffix
        ds_csvs = collect_csvs_nonrecursive(dirpath)
        ds_name = norm_tag(dataset_tag)
        for csv_path in ds_csvs:
            tasks.append((csv_path, ds_name, factor))

    resume_all = {}
    per_dataset_left = Counter()
    total_left = 0

    print("Computing left counts...")
    for csv_path, ds_name, factor in tqdm(tasks, desc="Computing left", unit="csv"):
        base = base_from_csv(csv_path)
        already = img_index.get(base, 0)
        left = max(0, factor - already)
        if left <= 0:
            continue

        resume_all[csv_path] = left
        per_dataset_left[ds_name] += left
        total_left += left

    os.makedirs(os.path.dirname(os.path.abspath(args.out_all)), exist_ok=True)
    with open(args.out_all, "w") as f:
        json.dump(dict(sorted(resume_all.items())), f, indent=2)

    print("\n=== Per-dataset left ===")
    for ds in sorted(set(ds for _, ds, _ in tasks)):
        print(f"{ds}: {per_dataset_left[ds]}")

    print(f"\nTotal left (all): {total_left}")
    print(f"Wrote:\n  all -> {args.out_all}")

    per_ds_total_over = Counter()
    per_ds_max_item = {}
    all_ds_names = set(ds for _, ds, _ in tasks)

    for csv_path, ds_name, factor in tasks:
        base = base_from_csv(csv_path)
        already = img_index.get(base, 0)
        over = max(0, already - factor)
        if over <= 0:
            if ds_name not in per_ds_total_over:
                per_ds_total_over[ds_name] = 0
            continue
        per_ds_total_over[ds_name] += over
        prev = per_ds_max_item.get(ds_name)
        if (prev is None) or (over > prev[1]):
            per_ds_max_item[ds_name] = (csv_path, over)

    print("\n=== Per-dataset over-repeats (total + max item) ===")
    for ds in sorted(all_ds_names):
        total = per_ds_total_over[ds]
        info = per_ds_max_item.get(ds)
        if info is None:
            print(f"{ds}: total_over=0 | max_item=- (over=0)")
        else:
            csv_path, max_over = info
            print(f"{ds}: total_over={total} | max_item={base_from_csv(csv_path)} (over={max_over})")

if __name__ == "__main__":
    main()
