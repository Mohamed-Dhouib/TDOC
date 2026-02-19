import os
import argparse
import pandas as pd
from multiprocessing import Pool

def is_valid_group(grp: pd.DataFrame, min_el: int) -> bool:
    """Check if group has enough elements."""
    return (
        len(grp) >= min_el
        and grp["filename"].nunique() >= 2
        and grp["text"].nunique()     >= 2
    )


def process_group(args, chunk_size = 2000):
    """Process a single group and save as chunked CSVs."""
    name, df, dest, min_el, task_id = args
    if not is_valid_group(df, min_el):
        return

    group_name = "_".join(map(str, name))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)

    folder = os.path.join(dest, group_name)
    os.makedirs(folder, exist_ok=True)

    for i in range(max(num_chunks - 2, 0)):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        chunk.to_csv(os.path.join(folder, f"{group_name}_{i+1}.csv"), index=False)

    start_idx = max(num_chunks - 2, 0) * chunk_size
    chunk = df.iloc[start_idx:]
    chunk.to_csv(os.path.join(folder, f"{group_name}_{max(num_chunks-2,0)+1}_{task_id}.csv"), index=False)


def main(
    dataset_folder: str,
    min_el: int,
    task_id: int,
    num_workers: int,
    delete_shard_after: bool = False,
):
    """Split groups into chunks for parallel negative mining."""
    shard_path = f"{dataset_folder}/grouped_csvs/merged_{task_id}.parquet"
    merged_df  = pd.read_parquet(shard_path)

    dest = f"{dataset_folder}/grouped_csvs"
    os.makedirs(dest, exist_ok=True)

    grouped = merged_df.groupby(
        ["len_text","Width","Height"]
    )

    group_sizes = grouped.size()
    print("\nGroup size summary:")
    print(group_sizes.describe().to_string())

    args_iter = ((name, grp, dest, min_el, task_id) for name, grp in grouped)

    print(f"[Task {task_id}] -> {len(grouped)} groups | workers={num_workers}")

  

    with Pool(processes=num_workers) as pool:
        pool.map(process_group, args_iter)

    print(f"[Task {task_id}] done.")

    if delete_shard_after:
        try:
            os.remove(shard_path)
            print(f"[Task {task_id}] deleted shard: {shard_path}")
        except FileNotFoundError:
            print(f"[Task {task_id}] shard already missing, skip delete: {shard_path}")
        except Exception as e:
            print(f"[Task {task_id}] could not delete shard '{shard_path}': {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_folder", required=True)
    p.add_argument("--min_el",      required=True, type=int)
    p.add_argument("--task_id",     required=True, type=int)
    p.add_argument("--num_workers", required=True, type=int)
    p.add_argument(
        "--delete_shard_after",
        action="store_true",
        help="Delete grouped_csvs/merged_<task_id>.parquet after processing (default: false).",
    )
    args = p.parse_args()
    main(
        args.dataset_folder,
        args.min_el,
        args.task_id,
        args.num_workers,
        delete_shard_after=args.delete_shard_after,
    )
