"""
src/inference/recommender.py

Provides the Recommender class with two primary modes:
 - ANN candidate + scoring (fast): query FAISS for candidate set then score candidates using user vector.
 - Full dense scoring (exact, slower): compute dot-product to all items and return top-K (useful for small catalogs or testing).

Features:
 - Exclude_seen: will remove items present in `user_seen` (set of item indices)
 - Optional fallback to dense scoring when FAISS missing
 - Simple CLI testing at bottom
"""

import os
from typing import List, Tuple, Optional, Set

import numpy as np

from .model_loader import load_all, ModelArtifacts, save_faiss_index
from .faiss_index import build_faiss_cpu_index, build_faiss_gpu_index

try:
    import faiss
except Exception:
    faiss = None


class Recommender:
    def __init__(self, artifacts: ModelArtifacts, faiss_index: Optional[any] = None, use_faiss: bool = True):
        self.artifacts = artifacts
        # item_vectors: items x f (raw)
        self.item_vectors = artifacts.item_vectors
        # user_vectors: users x f
        self.user_vectors = artifacts.user_vectors
        # normed item vectors (items x f) for FAISS (should be precomputed)
        self.item_vectors_normed = artifacts.item_vectors_normed
        self.faiss_index = faiss_index if faiss_index is not None else artifacts.faiss_index
        self.use_faiss = use_faiss and (self.faiss_index is not None)
        # basic sanity
        assert self.item_vectors is not None and self.user_vectors is not None
        assert self.item_vectors.shape[1] == self.user_vectors.shape[1]
        self.num_items = int(self.item_vectors.shape[0])
        self.num_users = int(self.user_vectors.shape[0])
        self.dim = int(self.item_vectors.shape[1])

    # ----------------------
    # Candidate generation
    # ----------------------
    def faiss_candidates(self, pos_item: int, n_candidates: int = 500, overshoot: int = 4) -> List[int]:
        """
        Query FAISS for nearest neighbors to pos_item using the normalized item vector.
        Returns up to n_candidates items (excludes nothing).
        """
        if not self.use_faiss:
            raise RuntimeError("FAISS not available. Build index or set use_faiss=False.")
        q = self.item_vectors_normed[int(pos_item)].reshape(1, -1).astype("float32")
        # ask for more than needed to allow filtering seen items upstream
        k = min(self.num_items, int(n_candidates * overshoot))
        D, I = self.faiss_index.search(q, k)
        return [int(x) for x in I[0] if x != -1]

    # ----------------------
    # Scoring
    # ----------------------
    def score_candidates(self, user_idx: int, candidates: List[int]) -> List[Tuple[int, float]]:
        """
        Score candidate items for user with dot-product: item_vecs.dot(user_vec)
        Returns list of (item_idx, score) sorted descending by score.
        """
        if user_idx < 0 or user_idx >= self.num_users:
            raise IndexError("user_idx out of range")
        uvec = self.user_vectors[user_idx]  # shape (f,)
        item_vecs = self.item_vectors[candidates]  # (n_candidates, f)
        scores = item_vecs.dot(uvec)
        # sort
        order = np.argsort(-scores)
        return [(int(candidates[i]), float(scores[i])) for i in order]

    def dense_topk(self, user_idx: int, k: int = 10, exclude: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        """
        Exact top-k by scoring all items (useful when num_items small or for testing).
        """
        uvec = self.user_vectors[user_idx]
        scores = self.item_vectors.dot(uvec)
        if exclude:
            for it in exclude:
                if 0 <= it < len(scores):
                    scores[int(it)] = -np.inf
        top_idx = np.argpartition(-scores, k)[:k]
        top_idx = sorted(top_idx, key=lambda i: -scores[i])
        return [(int(i), float(scores[i])) for i in top_idx]

    # ----------------------
    # High-level recommend
    # ----------------------
    def recommend(self,
                  user_idx: int,
                  K: int = 10,
                  exclude_seen: Optional[Set[int]] = None,
                  use_faiss_candidates: bool = True,
                  candidate_pool_size: int = 500,
                  overshoot: int = 4) -> Tuple[List[int], List[float]]:
        """
        Main entrypoint.
        - If use_faiss_candidates and FAISS available: get candidate pool, score and return top-K (fast).
        - Otherwise: dense top-k scoring over all items.

        exclude_seen: set of item indices to exclude (train items + any other)
        """
        if exclude_seen is None:
            exclude_seen = set()

        if use_faiss_candidates and self.use_faiss:
            # pick a seed candidate - best choice is to use user's last positive or multiple seeds.
            # here we use user's vector to get a near-neighbor seed? Simpler: call faiss with user's vector not supported if index built on items.
            # So we'll instead pick candidate list from FAISS using user's top popular item is not available here.
            # We'll use FAISS with an artificial query: project user_vec into item space by similarity -> use dot to all items but that's dense.
            # Simpler, practical: use FAISS by querying with user's vector normalized if dimension matches.
            try:
                # If item_vectors_normed exists and shapes match, we can query faiss with user vector normalized too.
                if self.item_vectors_normed is not None and self.item_vectors_normed.shape[1] == self.dim:
                    uq = self.user_vectors[user_idx]
                    uq_norm = uq / (np.linalg.norm(uq) + 1e-12)
                    D, I = self.faiss_index.search(uq_norm.reshape(1, -1).astype("float32"), min(self.num_items, candidate_pool_size * overshoot))
                    cand = [int(x) for x in I[0] if x != -1]
                else:
                    # fallback: use a small dense top-k to find seeds then expand via FAISS using the top seed
                    dense_seeds = self.dense_topk(user_idx, k=5, exclude=exclude_seen)
                    if len(dense_seeds) == 0:
                        cand = list(range(min(candidate_pool_size, self.num_items)))
                    else:
                        seed = dense_seeds[0][0]
                        cand = self.faiss_candidates(seed, n_candidates=candidate_pool_size, overshoot=overshoot)
            except Exception:
                # last-resort fallback
                cand = list(range(min(candidate_pool_size, self.num_items)))
            # filter exclude_seen and dedupe
            filtered = []
            seen_local = set()
            for c in cand:
                if c in exclude_seen or c in seen_local:
                    continue
                filtered.append(c)
                seen_local.add(c)
                if len(filtered) >= candidate_pool_size:
                    break
            candidates = filtered
            # score and return top-K
            scored = self.score_candidates(user_idx, candidates)
            topk = scored[:K]
            items = [i for i, s in topk]
            scores = [s for i, s in topk]
            return items, scores

        # fallback: exact dense scoring
        topk = self.dense_topk(user_idx, k=K, exclude=exclude_seen)
        items = [i for i, s in topk]
        scores = [s for i, s in topk]
        return items, scores


# -----------------------
# Simple CLI for quick local test
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="models/ dir with item_factors.npy & user_factors.npy")
    parser.add_argument("--user-idx", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--use-faiss", action="store_true")
    parser.add_argument("--faiss-gpu", action="store_true")
    args = parser.parse_args()

    print("Loading artifacts...")
    artifacts = load_all(args.model_dir)

    # optionally build faiss index if missing and user asked for it
    faiss_index = artifacts.faiss_index
    if args.use_faiss and faiss_index is None:
        print("FAISS index not found on disk. Building CPU FAISS index...")
        faiss_index = build_faiss_cpu_index(artifacts.item_vectors_normed, use_hnsw=True)
        try:
            save_faiss_index(faiss_index, args.model_dir)
            print("Saved faiss.index to model_dir.")
        except Exception:
            print("Could not save faiss.index; continuing with in-memory index.")

    rec = Recommender(artifacts, faiss_index=faiss_index, use_faiss=args.use_faiss)
    # For demo exclude_seen, try empty set (production should use real user's seen list)
    items, scores = rec.recommend(args.user_idx, K=args.k, exclude_seen=set(), use_faiss_candidates=args.use_faiss)
    print("Top-K items:", items)
    print("Scores:", scores)
