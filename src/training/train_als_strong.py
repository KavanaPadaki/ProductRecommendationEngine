# src/training/train_als_strong.py
"""
Strong ALS baseline with:
- BM25 confidence weighting
- Tuned hyperparameters
- Correct implicit usage

RUN:
    python -m src.training.train_als_strong
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

# ---------------- CONFIG ----------------
TRAIN_NPZ = "data/train.npz"
MAPPINGS  = "data/mappings.json"
MODEL_DIR = "models"

FACTORS = 128
ITERATIONS = 30
REG = 0.01
ALPHA = 1.0
# --------------------------------------


# ---------- load sparse matrix ----------
def load_csr_npz(path):
    z = np.load(path, allow_pickle=True)
    return sp.csr_matrix(
        (z["data"], z["indices"], z["indptr"]),
        shape=tuple(z["shape"])
    )


print("Loading train interactions...")
train_mat_user_item = load_csr_npz(TRAIN_NPZ)  # shape: (users, items)
num_users, num_items = train_mat_user_item.shape
print(f"Users={num_users}, Items={num_items}")

# ---------- BM25 weighting ----------
print("Applying BM25 confidence weighting...")
train_bm25 = bm25_weight(train_mat_user_item, K1=1.2, B=0.75)

# implicit expects (items x users)
train_bm25 = train_bm25.T.tocsr()

# ---------- train ALS ----------
print("Training ALS...")
model = AlternatingLeastSquares(
    factors=FACTORS,
    regularization=REG,
    iterations=ITERATIONS,
    alpha=ALPHA,
    use_gpu=False
)

model.fit(train_bm25)

# ---------- extract embeddings ----------
user_factors = model.item_factors    # (num_users, factors)
item_factors = model.user_factors    # (num_items, factors)

assert user_factors.shape[0] == num_users
assert item_factors.shape[0] == num_items

# ---------- save ----------
os.makedirs(MODEL_DIR, exist_ok=True)

np.save(f"{MODEL_DIR}/item_factors.npy", item_factors)
np.save(f"{MODEL_DIR}/user_factors.npy", user_factors)

# normalized items for FAISS / cosine
norm = np.linalg.norm(item_factors, axis=1, keepdims=True)
norm[norm == 0] = 1.0
item_norm = item_factors / norm
np.save(f"{MODEL_DIR}/item_vectors_normed.npy", item_norm)

print("Saved strong ALS embeddings to models/")
