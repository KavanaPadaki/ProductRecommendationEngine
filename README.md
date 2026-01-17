# Product Recommendation Engine — Production-Style System

An end-to-end recommender system designed with real-world scale, latency, and deployment constraints in mind.

This project demonstrates how a recommender system is designed, evaluated, and deployed beyond notebooks — covering offline training, online inference, approximate nearest neighbor search, and UI integration.

---

## Problem Framing

**Goal:** Recommend relevant products to users based on historical implicit feedback (views / interactions), optimizing ranking quality and low-latency inference.

**Key constraints addressed:**
- Large-scale user–item interaction data
- Implicit feedback (no explicit ratings)
- Fast online inference requirements
- Realistic offline evaluation
- Deployment on commodity infrastructure
---

## Links

- [**Live Demo (UI)**](https://appuctrecommendationengine-2b9wcgguunq9zfnjangey7.streamlit.app/)
- [**Inference API**](https://productrecommendationengine.onrender.com/)


---

## System Architecture (High Level)

- **Offline Training**
  - Train collaborative filtering models on implicit feedback
  - Export dense user/item embeddings
  - Precompute ANN index for inference

- **Online Inference (API)**
  - Load embeddings and ANN index into memory
  - Serve recommendations via REST APIs
  - Stateless, low-latency design

- **Frontend (UI)**
  - Thin client
  - No ML logic
  - Communicates only with inference APIs

---

## Models & Algorithms

### Matrix Factorization (ALS — Implicit Feedback)
- Confidence-weighted implicit interactions (BM25)
- Strong, scalable baseline
- Used as the production inference model

**Rationale:** Well-tuned matrix factorization remains a strong baseline in many production recommender systems due to simplicity, interpretability, and latency guarantees.

---

### LightGCN (Graph-Based Collaborative Filtering)
- BPR loss
- Message passing over user–item interaction graph
- Included to demonstrate understanding of modern graph-based recommenders

---

## Offline Evaluation

- **Hard Negative Sampling**
  - Negatives sampled from high-score non-interacted items
  - Better correlation with online ranking difficulty

- **Metrics**
  - Recall@K
  - NDCG@K

---

## Online Inference & Performance

- **FAISS (Approximate Nearest Neighbors)**
  - Pre-normalized item embeddings
  - CPU-based FAISS index
  - Two-stage retrieval: candidate generation → re-ranking

---

## Deployment & Infrastructure

- **Backend**
  - FastAPI (Python 3.10)
  - Loads embeddings and FAISS index at startup
  - Designed for stateless inference

- **Frontend**
  - Streamlit (Python 3.13)
  - Thin client architecture
  - Calls backend via REST APIs

Deployment explicitly accounts for Python version and dependency constraints.

---

## Engineering Tradeoffs & Decisions

- Decoupled training, inference, and UI
- Used ANN search instead of brute-force scoring
- Avoided metric inflation via hard-negative evaluation
- Chose simpler models where appropriate for reliability
- Designed for real deployment constraints (cold starts, infra limits)

---

## Future Extensions

- Session-based features
- Popularity and recency blending
- Online A/B testing hooks
- Feature store integration
- GPU-based ANN for larger catalogs


---

## One-Line Summary

An end-to-end recommender system with offline training, hard-negative evaluation, FAISS-based online inference, and a thin UI — designed around real production constraints.
