"""
src/inference/model_loader.py

Responsibilities:
- Load saved artifacts from models/ directory:
    - item_factors.npy  (shape: num_items x f)   <- implicit.user_factors after fit(train.T)
    - user_factors.npy  (shape: num_users x f)   <- implicit.item_factors after fit(train.T)
    - mappings.json     (optional)
    - faiss.index       (optional)
    - item_vectors_normed.npy (optional; L2-normalized item vectors)
- Provide a simple ModelArtifacts dataclass for other modules to consume.
- Provide save/load helpers for FAISS index (optional).
"""

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import faiss
except Exception:
    faiss = None  # optional dependency


@dataclass
class ModelArtifacts:
    item_vectors: np.ndarray       # items x dim (not necessarily normalized)
    user_vectors: np.ndarray       # users x dim
    item_vectors_normed: Optional[np.ndarray]  # L2-normalized items x dim
    mappings: Optional[Dict[str, Dict[Any, int]]]  # user_map, item_map
    faiss_index: Optional[Any]     # faiss index object or None
    num_users: int = 0
    num_items: int = 0
    factors: int = 0


def load_mappings(path: str) -> Optional[Dict[str, Dict[Any, int]]]:
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_factors(model_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load item_factors.npy and user_factors.npy from model_dir.
    Returns (item_vectors, user_vectors)
    Note: implicit naming after fit(train.T):
        model.user_factors -> item_vectors (num_items x f)
        model.item_factors -> user_vectors (num_users x f)
    """
    item_path = os.path.join(model_dir, "item_factors.npy")
    user_path = os.path.join(model_dir, "user_factors.npy")
    if not os.path.exists(item_path) or not os.path.exists(user_path):
        raise FileNotFoundError(f"Missing factors in {model_dir}. Expected item_factors.npy and user_factors.npy")
    item_vectors = np.load(item_path)
    user_vectors = np.load(user_path)
    return item_vectors, user_vectors


def try_load_faiss_index(model_dir: str) -> Optional[Any]:
    """
    Try to load faiss.index from model_dir/faiss.index.
    Returns index or None.
    """
    idx_path = os.path.join(model_dir, "faiss.index")
    if faiss is None or not os.path.exists(idx_path):
        return None
    return faiss.read_index(idx_path)


def ensure_item_vectors_normed(item_vectors: np.ndarray, model_dir: str) -> np.ndarray:
    """
    Load item_vectors_normed if present, otherwise compute and optionally save it.
    """
    normed_path = os.path.join(model_dir, "item_vectors_normed.npy")
    if os.path.exists(normed_path):
        return np.load(normed_path)
    norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    item_normed = item_vectors / norms
    try:
        np.save(normed_path, item_normed)
    except Exception:
        pass
    return item_normed


def load_all(model_dir: str) -> ModelArtifacts:
    """
    Load all artifacts and return ModelArtifacts.
    This is the main entrypoint other modules should call.
    """
    item_vectors, user_vectors = load_factors(model_dir)
    item_vectors_normed = ensure_item_vectors_normed(item_vectors, model_dir)
    mappings = load_mappings(os.path.join(model_dir, "mappings.json"))
    faiss_index = try_load_faiss_index(model_dir)
    ma = ModelArtifacts(
        item_vectors=item_vectors,
        user_vectors=user_vectors,
        item_vectors_normed=item_vectors_normed,
        mappings=mappings,
        faiss_index=faiss_index,
        num_users=int(user_vectors.shape[0]),
        num_items=int(item_vectors.shape[0]),
        factors=int(item_vectors.shape[1])
    )
    return ma


def save_faiss_index(index, model_dir: str, filename: str = "faiss.index"):
    """
    Persist a FAISS index to model_dir.
    """
    if faiss is None:
        raise RuntimeError("faiss not installed")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    faiss.write_index(index, path)
    return path
