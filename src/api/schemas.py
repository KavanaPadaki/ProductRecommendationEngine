# src/api/schemas.py
from typing import List, Optional
from pydantic import BaseModel


class RecommendRequest(BaseModel):
    user_idx: int
    k: int = 10
    exclude_seen: Optional[List[int]] = None
    use_faiss: Optional[bool] = True
    candidate_pool_size: Optional[int] = 500


class RecommendResponse(BaseModel):
    user_idx: int
    items: List[int]
    scores: List[float]


class MetadataResponse(BaseModel):
    num_users: int
    num_items: int
    factors: int
    faiss: bool
