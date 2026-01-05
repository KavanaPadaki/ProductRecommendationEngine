import numpy as np
import scipy.sparse as sp

MODEL_DIR = "models"
TRAIN_NPZ = "data/train.npz"
TEST_NPZ  = "data/test.npz"

K_LIST = [100, 200, 500]
NEG_HARD = 80
NEG_RAND = 20


def load_csr(path):
    z = np.load(path, allow_pickle=True)
    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))


print("Loading ALS embeddings...")
user_emb = np.load(f"{MODEL_DIR}/user_factors.npy")
item_emb = np.load(f"{MODEL_DIR}/item_factors.npy")

train_mat = load_csr(TRAIN_NPZ)
test_mat  = load_csr(TEST_NPZ)

num_users, num_items = train_mat.shape
assert user_emb.shape[0] == num_users
assert item_emb.shape[0] == num_items


def eval_hard_neg(K):
    rng = np.random.default_rng(42)
    recalls, ndcgs = [], []

    for u in range(num_users):
        test_items = test_mat[u].indices
        if len(test_items) == 0:
            continue

        scores = item_emb @ user_emb[u]
        scores[train_mat[u].indices] = -np.inf

        hard = np.argpartition(-scores, NEG_HARD)[:NEG_HARD]
        rand = rng.choice(num_items, NEG_RAND, replace=False)

        candidates = np.unique(np.concatenate([hard, rand, test_items]))
        cand_scores = scores[candidates]

        topk = candidates[np.argsort(-cand_scores)[:K]]
        hits = np.isin(topk, test_items)

        recall = hits.sum() / len(test_items)
        dcg = np.sum(hits / np.log2(np.arange(2, hits.size + 2)))
        idcg = np.sum(1.0 / np.log2(np.arange(2, min(K, len(test_items)) + 2)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        recalls.append(recall)
        ndcgs.append(ndcg)

    return np.mean(recalls), np.mean(ndcgs)


print("\n=== STRONG ALS (BM25) HARD-NEG EVAL ===")
for K in K_LIST:
    r, n = eval_hard_neg(K)
    print(f"Recall@{K}: {r:.6f}")
    print(f"NDCG@{K}:   {n:.6f}")
    print("-----")
