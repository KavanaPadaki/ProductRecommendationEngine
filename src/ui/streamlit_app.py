"""
Streamlit UI for Recommender System (Demo / Interview Ready)

Features:
- Robust FastAPI integration
- Graceful handling of FAISS similarity (never crashes)
- User selection via user_idx
- Top-K recommendations
- Optional item-to-item similarity
- Product names via metadata (item_meta.json)
- Embedding visualization (PCA, local only)
- Streamlit deprecation-safe (width="stretch")

Run:
    streamlit run src/ui/streamlit_app.py
"""

import os
import json
import time
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Recommender Demo", layout="wide")

# -------------------------
# Config
# -------------------------
API_URL = os.environ.get("API_URL", "http://localhost:8000")

RECOMMEND_ENDPOINT = f"{API_URL}/api/recommend"
SIMILAR_ENDPOINT = f"{API_URL}/api/similar"
METADATA_ENDPOINT = f"{API_URL}/api/metadata"

MODELS_DIR = "models"
MAPPINGS_PATH = f"{MODELS_DIR}/mappings.json"
ITEM_META_PATH = f"{MODELS_DIR}/item_meta.json"
ITEM_VECS_PATH = f"{MODELS_DIR}/item_vectors_normed.npy"

# -------------------------
# Cached loaders
# -------------------------
@st.cache_data
def load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_item_vectors(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    return np.load(path)


@st.cache_data
def fetch_metadata() -> Dict:
    try:
        r = requests.get(METADATA_ENDPOINT, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# -------------------------
# API helpers
# -------------------------
def post_recommend(
    user_idx: int,
    k: int,
    use_faiss: bool,
    candidate_pool_size: int,
    exclude_seen: Optional[List[int]]
):
    payload = {
        "user_idx": user_idx,
        "k": k,
        "use_faiss": use_faiss,
        "candidate_pool_size": candidate_pool_size
    }
    if exclude_seen:
        payload["exclude_seen"] = exclude_seen

    r = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def get_similar_safe(item_idx: int, k: int):
    """
    Best-effort similarity:
    - Never raises
    - Returns [] if unavailable
    """
    try:
        r = requests.get(f"{SIMILAR_ENDPOINT}/{item_idx}?k={k}", timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("neighbors", [])
    except Exception:
        return []

# -------------------------
# Load local artifacts
# -------------------------
mappings = load_json(MAPPINGS_PATH)
item_meta = load_json(ITEM_META_PATH)
item_vectors = load_item_vectors(ITEM_VECS_PATH)

inv_item_map = {}
if mappings and "item_map" in mappings:
    inv_item_map = {v: k for k, v in mappings["item_map"].items()}

def item_title(item_idx: int) -> str:
    """
    Resolve product title with clean fallback.
    """
    asin = inv_item_map.get(item_idx)
    if asin and item_meta and asin in item_meta:
        return item_meta[asin].get("title", "")
    return f"Item {item_idx}"

# -------------------------
# UI
# -------------------------
st.title("Recommender System — Demo")

col_left, col_right = st.columns([1, 2])

# ===== Left panel =====
with col_left:
    st.subheader("Server status")

    metadata = fetch_metadata()
    if "error" in metadata:
        st.error(f"API unavailable: {metadata['error']}")
        FAISS_AVAILABLE = False
    else:
        st.metric("Users", metadata.get("num_users"))
        st.metric("Items", metadata.get("num_items"))
        st.metric("Factors", metadata.get("factors"))
        FAISS_AVAILABLE = bool(metadata.get("faiss"))
        st.write("FAISS enabled:", FAISS_AVAILABLE)

    st.markdown("---")
    st.subheader("Recommendation request")

    user_idx = st.number_input(
        "User index (0-based)",
        min_value=0,
        value=0,
        step=1
    )

    k = st.slider("Top-K", 1, 50, 10)
    use_faiss = st.checkbox("Use FAISS candidate retrieval", value=True)
    candidate_pool = st.number_input(
        "Candidate pool size",
        min_value=50,
        value=500,
        step=50
    )

    exclude_raw = st.text_input(
        "Exclude seen items (comma-separated item_idx)",
        ""
    )
    exclude_seen = (
        [int(x) for x in exclude_raw.split(",") if x.strip().isdigit()]
        if exclude_raw else None
    )

    if st.button("Recommend"):
        with st.spinner("Fetching recommendations..."):
            try:
                t0 = time.time()
                result = post_recommend(
                    user_idx=user_idx,
                    k=k,
                    use_faiss=use_faiss,
                    candidate_pool_size=candidate_pool,
                    exclude_seen=exclude_seen
                )
                st.session_state["recs"] = result
                st.success(f"Completed in {time.time() - t0:.2f}s")
            except Exception as e:
                st.error(f"Recommendation failed: {e}")
                st.session_state["recs"] = None

# ===== Right panel =====
with col_right:
    st.subheader("Recommendations")

    recs = st.session_state.get("recs")
    if not recs:
        st.info("No recommendations yet.")
    else:
        items = recs["items"]
        scores = recs["scores"]

        df = pd.DataFrame({
            "rank": range(1, len(items) + 1),
            "item_idx": items,
            "product_name": [item_title(i) for i in items],
            "score": scores
        })

        st.dataframe(df, width="stretch")

        # -------- Similar items --------
        st.markdown("### Inspect an item")

        selected_item = st.selectbox(
            "Select an item from recommendations",
            options=items
        )

        if not FAISS_AVAILABLE:
            st.info("Item-to-item similarity unavailable (FAISS disabled on server).")
            neighbors = []
        else:
            if st.button("Get similar items"):
                with st.spinner("Finding similar items..."):
                    neighbors = get_similar_safe(selected_item, k=10)
                    st.session_state["similar"] = neighbors

        if "similar" in st.session_state and st.session_state["similar"]:
            sim_items = st.session_state["similar"]
            sim_df = pd.DataFrame({
                "item_idx": sim_items,
                "product_name": [item_title(i) for i in sim_items]
            })
            st.dataframe(sim_df, width="stretch")
        elif FAISS_AVAILABLE:
            st.caption("No similar items available for this selection.")

        # -------- Embedding visualization --------
        st.markdown("### Embedding visualization (PCA)")

        if item_vectors is None:
            st.info("Local item embeddings not found.")
        else:
            points = list(items)
            if "similar" in st.session_state:
                points += st.session_state["similar"]
            points = list(dict.fromkeys(points))

            if len(points) < 2:
                st.info("Not enough points to visualize.")
            else:
                vecs = item_vectors[np.array(points)]
                pts = PCA(n_components=2).fit_transform(vecs)

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(pts[:, 0], pts[:, 1])
                for i, idx in enumerate(points):
                    ax.annotate(str(idx), (pts[i, 0], pts[i, 1]))
                ax.set_title("Item embeddings (PCA)")
                st.pyplot(fig)

st.markdown("---")
st.caption(
    "ALS candidate generation · Hard-negative evaluation · FastAPI · Streamlit · Amazon Electronics"
)
