import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

def build_csv_list(dataset_folder: str):
    """Get list of CSV files from merged_jsons folder."""
    csv_files = []
    dirpath = os.path.join(dataset_folder, "merged_jsons")
    csv_files = glob(os.path.join(dirpath, "*.csv"))

    return csv_files


def read_csv(path: str) -> pd.DataFrame:
    """Read CSV file with error handling."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read '{path}': {e}")
        return pd.DataFrame()


def main(dataset_folder: str, num_jobs: int,save_only_first=False):
    """Merge CSVs and split into shards for parallel processing."""
    csv_files = build_csv_list(dataset_folder)

    with Pool() as pool:
        dfs = list(tqdm(pool.imap(read_csv, csv_files), total=len(csv_files)))

    merged_df = pd.concat(dfs, ignore_index=True)

    for col in ("text", "filename"):
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype(str)

    out_dir = f"{dataset_folder}/grouped_csvs"
    os.makedirs(out_dir, exist_ok=True)

    for idx, shard in enumerate(np.array_split(merged_df, num_jobs)):
        save_path=os.path.join(out_dir, f"merged_{idx}.parquet")

        shard.to_parquet(save_path)
    
        print(f"[OK] Saved shard {idx+1}/{num_jobs} to {save_path}")
        if save_only_first:
            break
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", required=True)
    parser.add_argument("--num_jobs",    required=True, type=int)
    args = parser.parse_args()
    main(args.dataset_folder, args.num_jobs)
