from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, Set, List

from .schemas import RecommendRequest, RecommendResponse, MetadataResponse

# Placeholder dependency – overridden in main.py
def get_rec():
    return None

router = APIRouter()


def _to_set(lst: Optional[List[int]]) -> Set[int]:
    if not lst:
        return set()
    return set(int(x) for x in lst)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/metadata", response_model=MetadataResponse)
def metadata(recommender = Depends(get_rec)):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MetadataResponse(
        num_users=recommender.num_users,
        num_items=recommender.num_items,
        factors=recommender.dim,
        faiss=bool(recommender.use_faiss)
    )


@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest, recommender = Depends(get_rec)):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if req.user_idx < 0 or req.user_idx >= recommender.num_users:
        raise HTTPException(status_code=404, detail="Unknown user index")

    exclude = _to_set(req.exclude_seen)
    items, scores = recommender.recommend(
        user_idx=req.user_idx,
        K=req.k,
        exclude_seen=exclude,
        use_faiss_candidates=bool(req.use_faiss),
        candidate_pool_size=int(req.candidate_pool_size)
    )
    return RecommendResponse(user_idx=req.user_idx, items=items, scores=scores)


@router.get("/similar/{item_idx}")
def similar(
    item_idx: int,
    k: int = Query(10, ge=1, le=1000),
    recommender = Depends(get_rec)
):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not recommender.use_faiss:
        raise HTTPException(status_code=400, detail="FAISS index not available on this server")
    if item_idx < 0 or item_idx >= recommender.num_items:
        raise HTTPException(status_code=404, detail="Unknown item index")

    try:
        cand = recommender.faiss_candidates(pos_item=item_idx, n_candidates=k, overshoot=2)
        return {"item_idx": item_idx, "neighbors": cand[:k]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
