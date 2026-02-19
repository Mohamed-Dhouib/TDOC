#!/usr/bin/env python3
import os
import cv2
import argparse
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

REASON_OK = "ok"
REASON_MISSING_IMAGE = "missing_image"
REASON_MISSING_MASK = "missing_mask"
REASON_READ_FAIL_IMAGE = "read_fail_image"
REASON_READ_FAIL_MASK = "read_fail_mask"
REASON_TOO_SMALL_IMAGE = "too_small_image"  # < 2 KB
REASON_SHAPE_MISMATCH = "shape_mismatch"

MIN_IMAGE_SIZE_BYTES = 2048  # 2 KB


def build_pairs(root_dir):
    """Build list of (name, image_path, mask_path) tuples."""
    images_dir = os.path.join(root_dir, "images")
    masks_dir = os.path.join(root_dir, "masks")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"'images' folder not found under {root_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"'masks' folder not found under {root_dir}")

    image_names = set(os.listdir(images_dir))
    mask_names = set(os.listdir(masks_dir))

    all_names = image_names | mask_names
    pairs = []
    for name in sorted(all_names):
        img_path = os.path.join(images_dir, name) if name in image_names else None
        mask_path = os.path.join(masks_dir, name) if name in mask_names else None
        pairs.append((name, img_path, mask_path))
    return pairs


def stable_shard_idx(name: str, num_shards: int) -> int:
    """
    Deterministic, content-independent shard:
    every script run, on any node, will map the same `name` to the same shard.
    """
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(h, 16) % num_shards


def file_size_bytes(path):
    """Get file size in bytes, return -1 on error."""
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def process_one(item, only_stats=False):
    """Validate one image-mask pair and optionally delete invalid files."""
    name, img_path, mask_path = item

    if img_path is None or not os.path.exists(img_path):
        if not only_stats and mask_path and os.path.exists(mask_path):
            try:
                os.remove(mask_path)
            except OSError:
                pass
        return {
            "name": name,
            "reason": REASON_MISSING_IMAGE,
            "delete_image": False,
            "delete_mask": bool(mask_path and os.path.exists(mask_path)),
        }

    if mask_path is None or not os.path.exists(mask_path):
        if not only_stats and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except OSError:
                pass
        return {
            "name": name,
            "reason": REASON_MISSING_MASK,
            "delete_image": bool(os.path.exists(img_path)),
            "delete_mask": False,
        }

    img_size = file_size_bytes(img_path)
    if img_size < MIN_IMAGE_SIZE_BYTES:
        print(f"[too-small] {name} -> {img_size} bytes (< {MIN_IMAGE_SIZE_BYTES})")
        if not only_stats:
            try:
                os.remove(img_path)
            except OSError:
                pass
            if os.path.exists(mask_path):
                try:
                    os.remove(mask_path)
                except OSError:
                    pass
        return {
            "name": name,
            "reason": REASON_TOO_SMALL_IMAGE,
            "delete_image": True,
            "delete_mask": True,
        }

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        if not only_stats:
            try:
                os.remove(img_path)
            except OSError:
                pass
            if os.path.exists(mask_path):
                try:
                    os.remove(mask_path)
                except OSError:
                    pass
        return {
            "name": name,
            "reason": REASON_READ_FAIL_IMAGE,
            "delete_image": True,
            "delete_mask": True,
        }

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        if not only_stats:
            try:
                os.remove(img_path)
            except OSError:
                pass
            if os.path.exists(mask_path):
                try:
                    os.remove(mask_path)
                except OSError:
                    pass
        return {
            "name": name,
            "reason": REASON_READ_FAIL_MASK,
            "delete_image": True,
            "delete_mask": True,
        }

    h_img, w_img = img.shape[:2]
    h_msk, w_msk = mask.shape[:2]
    if h_img != h_msk or w_img != w_msk:
        if not only_stats:
            try:
                os.remove(img_path)
            except OSError:
                pass
            if os.path.exists(mask_path):
                try:
                    os.remove(mask_path)
                except OSError:
                    pass
        return {
            "name": name,
            "reason": REASON_SHAPE_MISMATCH,
            "delete_image": True,
            "delete_mask": True,
        }

    return {
        "name": name,
        "reason": REASON_OK,
        "delete_image": False,
        "delete_mask": False,
    }


def aggregate_and_print(results, only_stats=False, shard_idx=0, num_shards=1):
    """Summarize validation results and print statistics."""
    counts = {
        REASON_OK: 0,
        REASON_MISSING_IMAGE: 0,
        REASON_MISSING_MASK: 0,
        REASON_READ_FAIL_IMAGE: 0,
        REASON_READ_FAIL_MASK: 0,
        REASON_TOO_SMALL_IMAGE: 0,
        REASON_SHAPE_MISMATCH: 0,
    }

    to_delete = []

    for r in results:
        counts[r["reason"]] += 1
        if r["reason"] != REASON_OK:
            to_delete.append(r)

    total = len(results)
    deleted = total - counts[REASON_OK]

    print(f"\n=== Dataset cleaning report (shard {shard_idx}/{num_shards}) ===")
    print(f"Total entries inspected (this shard): {total}")
    print(f"Kept: {counts[REASON_OK]}")
    print(f"To delete (any reason): {deleted}\n")

    print("Breakdown by reason:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    if only_stats and to_delete:
        print("\n--- Files that WOULD be deleted (only-stats mode) ---")
        for r in to_delete[:200]:
            print(f"{r['name']}: {r['reason']}")
        if len(to_delete) > 200:
            print(f"... and {len(to_delete) - 200} more")

    print()
    if only_stats:
        print("Nothing was deleted (only-stats mode).")
    else:
        print("Deletion has been performed.")


def main():
    parser = argparse.ArgumentParser(
        description="Check image/mask pairs under a folder (images/ and masks/) with OpenCV."
    )
    parser.add_argument("--root", required=True,
                        help="Root folder containing 'images' and 'masks'")
    parser.add_argument("--only-stats", action="store_true",
                        help="Do NOT delete, only print what would be deleted.")
    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help="Number of worker processes (default: all CPUs).")
    parser.add_argument("--shard-idx", type=int, default=0,
                        help="Which shard am I? (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards (for multi-node / array jobs)")
    args = parser.parse_args()

    pairs = build_pairs(args.root)

    if args.num_shards > 1:
        my_pairs = [
            p for p in pairs
            if stable_shard_idx(p[0], args.num_shards) == args.shard_idx
        ]
    else:
        my_pairs = pairs

    worker = partial(process_one, only_stats=args.only_stats)

    results = []
    total = len(my_pairs)
    desc = f"shard {args.shard_idx}/{args.num_shards}" if args.num_shards > 1 else "checking"

    with Pool(processes=args.workers) as pool, tqdm(total=total, desc=desc) as pbar:
        for r in pool.imap_unordered(worker, my_pairs, chunksize=64):
            results.append(r)
            pbar.update(1)

    aggregate_and_print(results,
                        only_stats=args.only_stats,
                        shard_idx=args.shard_idx,
                        num_shards=args.num_shards)


if __name__ == "__main__":
    main()
