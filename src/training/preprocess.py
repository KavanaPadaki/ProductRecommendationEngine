# src/training/preprocess.py
"""
preprocess.py
- Loads raw JSON-lines dataset (Amazon)
- Keeps necessary columns
- Maps user/item -> integer indices
- Saves processed CSVs and mappings
"""
import argparse
import os
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from .utils import save_mappings

tqdm.pandas()


def load_raw(path: str, user_col="reviewerID", item_col="asin", time_cols=("unixReviewTime", "reviewTime")) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    # pick timestamp if present
    if time_cols[0] in df.columns:
        df["timestamp"] = pd.to_datetime(df[time_cols[0]], unit="s")
    elif time_cols[1] in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df[time_cols[1]])
        except Exception:
            df["timestamp"] = pd.Timestamp.now()
    else:
        df["timestamp"] = pd.Timestamp.now()
    df = df[[user_col, item_col, "timestamp"]].dropna().rename(columns={user_col: "user_id", item_col: "item_id"})
    return df.reset_index(drop=True)


def build_mappings(df: pd.DataFrame) -> Tuple[dict, dict]:
    users = df["user_id"].unique()
    items = df["item_id"].unique()
    user_map = {u: int(i) for i, u in enumerate(users)}
    item_map = {m: int(i) for i, m in enumerate(items)}
    return user_map, item_map


def apply_mappings(df: pd.DataFrame, user_map: dict, item_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)
    # drop rows that couldn't be mapped (shouldn't happen)
    df = df.dropna(subset=["user_idx", "item_idx"])
    df["user_idx"] = df["user_idx"].astype(int)
    df["item_idx"] = df["item_idx"].astype(int)
    return df


def time_split_per_user(df: pd.DataFrame, k: int = 1, strategy: str = "last_k", test_frac: float = 0.2):
    df = df.sort_values(["user_id", "timestamp"])
    if strategy == "last_k":
        test_idx = df.groupby("user_id").tail(k).index
    elif strategy == "last_frac":
        def tail_frac(g):
            n = max(1, int(len(g) * test_frac))
            return g.tail(n).index
        test_idx = df.groupby("user_id").apply(tail_frac).explode().astype(int).values
    else:
        raise ValueError("Unknown split strategy")
    test_df = df.loc[test_idx].copy().reset_index(drop=True)
    train_df = df.drop(test_idx).copy().reset_index(drop=True)
    return train_df, test_df


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading raw data...")
    df = load_raw(args.input_path, user_col=args.user_col, item_col=args.item_col)
    print("Building mappings from union of dataset...")
    user_map, item_map = build_mappings(df)
    print(f"Users: {len(user_map)}, Items: {len(item_map)}, Rows: {len(df)}")

    print("Applying mappings...")
    df_mapped = apply_mappings(df, user_map, item_map)

    print("Splitting per-user (time-based)...")
    train_df, test_df = time_split_per_user(df_mapped, k=args.k, strategy=args.split_strategy, test_frac=args.test_frac)

    # save files
    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    mappings_path = os.path.join(args.output_dir, "mappings.json")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    save_mappings(user_map, item_map, mappings_path)

    print("Saved:")
    print(" -", train_path)
    print(" -", test_path)
    print(" -", mappings_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", default="../../data/")
    parser.add_argument("--user-col", default="reviewerID")
    parser.add_argument("--item-col", default="asin")
    parser.add_argument("--split-strategy", default="last_k", choices=["last_k", "last_frac"])
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
