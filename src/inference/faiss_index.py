"""
src/inference/faiss_index.py

Utilities to build FAISS index (CPU or GPU) from item_vectors_normed.
Exposes:
 - build_faiss_cpu_index(item_vectors_normed, use_hnsw=True, ef_search=64)
 - build_faiss_gpu_index(item_vectors_normed, use_hnsw=True, gpu_id=0, temp_mem=256MB, ef_search=64)
These return a FAISS index object ready for .search(...)
"""
from typing import Any, Optional
import numpy as np

try:
    import faiss
except Exception:
    faiss = None


def build_faiss_cpu_index(item_vectors_normed: np.ndarray, use_hnsw: bool = True, ef_search: int = 64) -> Any:
    if faiss is None:
        raise RuntimeError("faiss not installed")
    dim = int(item_vectors_normed.shape[1])
    if use_hnsw:
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 64
        index.hnsw.efSearch = ef_search
        index.add(item_vectors_normed.astype(np.float32))
        return index
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(item_vectors_normed.astype(np.float32))
        return index


def build_faiss_gpu_index(item_vectors_normed: np.ndarray, use_hnsw: bool = True, gpu_id: int = 0,
                          temp_mem_bytes: int = 256 * 1024 * 1024, ef_search: int = 64) -> Any:
    if faiss is None:
        raise RuntimeError("faiss not installed")
    res = faiss.StandardGpuResources()
    # temp mem optional tuning
    try:
        res.setTempMemory(temp_mem_bytes)
    except Exception:
        pass

    dim = int(item_vectors_normed.shape[1])
    if use_hnsw:
        cpu_index = faiss.IndexHNSWFlat(dim, 32)
        cpu_index.hnsw.efConstruction = 64
        cpu_index.add(item_vectors_normed.astype(np.float32))
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
        gpu_index.hnsw.efSearch = ef_search
        return gpu_index
    else:
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(item_vectors_normed.astype(np.float32))
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
        return gpu_index
