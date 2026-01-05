# src/training/build_matrix.py
"""
Build CSR interaction matrices from processed CSVs.
Saves train/test CSR as compressed npz for quick reload.
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import sparse_save_csr, load_mappings


def build_interaction_matrix(df: pd.DataFrame, num_users: int, num_items: int, weight: float = 1.0):
    rows = df["user_idx"].astype(np.int64).values
    cols = df["item_idx"].astype(np.int64).values
    data = np.ones(len(rows), dtype=np.float64) * weight
    mat = sp.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    mat.eliminate_zeros()
    return mat


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    user_map, item_map = load_mappings(args.mappings_path)
    num_users = len(user_map)
    num_items = len(item_map)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    # Ensure user_idx/item_idx columns exist (they should from preprocess)
    if "user_idx" not in train_df.columns or "item_idx" not in train_df.columns:
        raise RuntimeError("train.csv missing user_idx/item_idx columns")
    if "user_idx" not in test_df.columns or "item_idx" not in test_df.columns:
        raise RuntimeError("test.csv missing user_idx/item_idx columns")

    train_mat = build_interaction_matrix(train_df, num_users, num_items, weight=args.train_confidence)
    test_mat = build_interaction_matrix(test_df, num_users, num_items, weight=1.0)

    sparse_save_csr(os.path.join(args.output_dir, "train"), train_mat)
    sparse_save_csr(os.path.join(args.output_dir, "test"), test_mat)

    print("Saved CSR matrices to", args.output_dir)
    print("Train nnz:", train_mat.nnz, "Test nnz:", test_mat.nnz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="../../data/train.csv")
    parser.add_argument("--test-csv", default="../../data/test.csv")
    parser.add_argument("--mappings-path", default="../../data/mappings.json")
    parser.add_argument("--output-dir", default="../../data/")
    parser.add_argument("--train-confidence", type=float, default=40.0)
    args = parser.parse_args()
    main(args)
