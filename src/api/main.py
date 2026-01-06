# src/api/main.py
"""
FastAPI app that loads model artifacts and exposes recommendation endpoints.

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Environment variables (optional):
  MODEL_DIR        path to models/ (default: ./models)
  USE_FAISS        "1" or "0" to control whether to try to use FAISS (default: 1)
  FAISS_GPU        "1" or "0" (if you have faiss-gpu and want to build on GPU)
  CANDIDATE_POOL   default candidate_pool_size used if not provided
"""
import os
import logging
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import get_rec

# import inference primitives
from src.inference.model_loader import load_all, save_faiss_index, ModelArtifacts
from src.inference.recommender import Recommender
from src.inference.faiss_index import build_faiss_cpu_index, build_faiss_gpu_index

from .routes import router as api_router

# Config via environment
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
USE_FAISS = os.environ.get("USE_FAISS", "1") == "1"
FAISS_GPU = os.environ.get("FAISS_GPU", "0") == "1"
CANDIDATE_POOL = int(os.environ.get("CANDIDATE_POOL", 500))
HF_REPO = os.environ.get(
    "HF_MODEL_REPO",
    "KavanaPadaki/product-recommender-als"
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender.api")

app = FastAPI(title="Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
ensure_hf_artifacts(
    repo_id=HF_REPO,
    model_dir=MODEL_DIR,
    filenames=[
        "item_factors.npy",
        "user_factors.npy",
        "item_vectors_normed.npy",
        "faiss.index",
        "mappings.json",
        "item_meta.json",
    ],
)


# Global recommender placeholder
_GLOBAL_RECOMMENDER: Optional[Recommender] = None


def get_recommender():
    if _GLOBAL_RECOMMENDER is None:
        raise RuntimeError("Recommender not initialized")
    return _GLOBAL_RECOMMENDER


# small in-memory cache for repeated top-K queries for the same user (hot users)
# keep cache small to avoid memory surprises
@lru_cache(maxsize=4096)
def _cached_recommend(user_idx: int, k: int, use_faiss_flag: bool, pool_size: int):
    rec = get_recommender()
    items, scores = rec.recommend(user_idx=user_idx, K=k, exclude_seen=set(),
                                  use_faiss_candidates=use_faiss_flag,
                                  candidate_pool_size=pool_size)
    return items, scores


@app.on_event("startup")
def load_model_on_startup():
    global _GLOBAL_RECOMMENDER
    logger.info("Loading artifacts from %s", MODEL_DIR)
    artifacts: ModelArtifacts = load_all(MODEL_DIR)
    logger.info("Artifacts loaded: users=%d items=%d dim=%d", artifacts.num_users, artifacts.num_items, artifacts.factors)

    # if artifacts already include a FAISS index, reuse it
    faiss_index = artifacts.faiss_index
    if USE_FAISS and faiss_index is None:
        logger.info("No persisted FAISS index found. Building in-memory FAISS index (this may take a moment)...")
        if FAISS_GPU:
            logger.info("Building FAISS on GPU")
            faiss_index = build_faiss_gpu_index(artifacts.item_vectors_normed, use_hnsw=True)
        else:
            logger.info("Building FAISS on CPU")
            faiss_index = build_faiss_cpu_index(artifacts.item_vectors_normed, use_hnsw=True)
        # attempt to persist (best-effort)
        try:
            save_faiss_index(faiss_index, MODEL_DIR)
            logger.info("Persisted FAISS index to %s", MODEL_DIR)
        except Exception:
            logger.warning("Could not persist FAISS index to disk (continuing with in-memory index)")

    # initialize recommender
    _GLOBAL_RECOMMENDER = Recommender(artifacts, faiss_index=faiss_index, use_faiss=USE_FAISS and faiss_index is not None)
    logger.info("Recommender initialized; FAISS available=%s", bool(_GLOBAL_RECOMMENDER.use_faiss))

    # include routes, inject dependency resolver so routes can access recommender
   

    app.dependency_overrides[get_rec] = get_recommender
    app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {"status": "ok", "info": "Recommender API. Use /api/recommend or /api/metadata"}


# optional: endpoint that uses the LRU cache
@app.get("/api/recommend_cached/{user_idx}")
def recommend_cached(user_idx: int, k: int = 10, use_faiss: bool = True):
    items, scores = _cached_recommend(user_idx, k, bool(use_faiss), CANDIDATE_POOL)
    return {"user_idx": user_idx, "items": items, "scores": scores}

