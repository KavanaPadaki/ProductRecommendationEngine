"""
Streamlit UI for Recommender System

- Thin client: calls FastAPI backend only
- Shows top-K recommendations
- Shows similar items
- Maps item_idx -> product title (optional, via item_meta.json)

Run locally:
    streamlit run src/ui/streamlit_app.py

Env / secrets:
    API_URL         (default: http://localhost:8000)
    ITEM_META_PATH  (optional, default: not used)
"""

import os
import json
import time
from typing import Dict, Optional

import requests
import pandas as pd
import streamlit as st

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Product Recommender Demo",
    layout="wide",
)

# -------------------------
# Config
# -------------------------
API_URL = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000"))

RECOMMEND_ENDPOINT = f"{API_URL}/api/recommend"
SIMILAR_ENDPOINT = f"{API_URL}/api/similar"
METADATA_ENDPOINT = f"{API_URL}/api/metadata"

ITEM_META_PATH = st.secrets.get("ITEM_META_PATH", os.environ.get("ITEM_META_PATH"))

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_item_meta(path: Optional[str]) -> Dict[str, Dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def fetch_metadata() -> Optional[Dict]:
    try:
        r = requests.get(METADATA_ENDPOINT, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def post_recommend(user_idx: int, k: int, use_faiss: bool, candidate_pool_size: int):
    payload = {
        "user_idx": int(user_idx),
        "k": int(k),
        "use_faiss": bool(use_faiss),
        "candidate_pool_size": int(candidate_pool_size),
    }
    r = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def get_similar(item_idx: int, k: int = 10):
    r = requests.get(
        f"{SIMILAR_ENDPOINT}/{item_idx}",
        params={"k": k},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# -------------------------
# Load lightweight metadata
# -------------------------
item_meta = load_item_meta(ITEM_META_PATH)


def item_title(item_idx: int) -> str:
    return item_meta.get(str(item_idx), {}).get("title", "")


# -------------------------
# UI
# -------------------------
st.title("Product Recommender System — Demo")

meta = fetch_metadata()

backend_ok = isinstance(meta, dict) and "num_users" in meta

if not backend_ok:
    st.error(
        f"Backend not reachable.\n\n"
        f"API_URL = {API_URL}\n\n"
        f"Response = {meta}"
    )
    st.stop()

with st.expander("Backend Info", expanded=False):
    st.write(f"**API URL:** `{API_URL}`")
    st.write(f"**Users:** {meta['num_users']}")
    st.write(f"**Items:** {meta['num_items']}")
    st.write(f"**Latent factors:** {meta['factors']}")
    st.write(f"**FAISS enabled:** {meta['faiss']}")

st.markdown("---")

# -------------------------
# Controls
# -------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Input")

    user_idx = st.number_input(
        "User index",
        min_value=0,
        value=0,
        step=1,
    )

    k = st.slider("Top-K", min_value=1, max_value=50, value=10)

    use_faiss = st.checkbox("Use FAISS candidates", value=True)

    candidate_pool = st.number_input(
        "Candidate pool size",
        min_value=50,
        max_value=5000,
        value=500,
        step=50,
    )

    run_btn = st.button("Recommend", disabled=not backend_ok)

# -------------------------
# Recommendations
# -------------------------
with col_right:
    st.subheader("Recommendations")

    if run_btn:
        with st.spinner("Fetching recommendations..."):
            t0 = time.time()
            try:
                rec = post_recommend(
                    user_idx=user_idx,
                    k=k,
                    use_faiss=use_faiss,
                    candidate_pool_size=candidate_pool,
                )
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

            latency = time.time() - t0
            st.caption(f"Response time: {latency:.2f}s")

        items = rec.get("items", [])
        scores = rec.get("scores", [])

        if not items:
            st.warning("No recommendations returned.")
            st.stop()

        df = pd.DataFrame({
            "rank": range(1, len(items) + 1),
            "item_idx": items,
            "title": [item_title(i) for i in items],
            "score": scores,
        })

        st.dataframe(df, width="stretch")

        # -------------------------
        # Similar items
        # -------------------------
        st.markdown("### Similar Items")

        selected_item = st.selectbox(
            "Select an item",
            options=items,
            format_func=lambda x: f"{x} — {item_title(x)}",
        )

        if st.button("Find similar items"):
            with st.spinner("Fetching similar items..."):
                try:
                    sim = get_similar(selected_item, k=10)
                except Exception as e:
                    st.error(f"Similar-items request failed: {e}")
                    st.stop()

            if "neighbors" not in sim:
                st.error(f"Unexpected response: {sim}")
                st.stop()

            neighbors = sim["neighbors"]

            sim_df = pd.DataFrame({
                "rank": range(1, len(neighbors) + 1),
                "item_idx": neighbors,
                "title": [item_title(i) for i in neighbors],
            })

            st.dataframe(sim_df, width="stretch")

st.markdown("---")
st.caption(
    "This UI is a thin client. All models, embeddings, and FAISS live in the backend."
)
