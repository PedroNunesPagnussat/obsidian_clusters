"""Clustering on full embeddings + t-SNE (2D/3D) for visualization."""

from __future__ import annotations

import numpy as np


def _tsne(emb: np.ndarray, n_components: int, seed: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE

    n = len(emb)
    perplexity = float(min(30, max(5, (n - 1) / 3)))
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(emb).astype(np.float32)


def reduce_2d(emb: np.ndarray, seed: int = 42) -> np.ndarray:
    return _tsne(emb, 2, seed)


def reduce_3d(emb: np.ndarray, seed: int = 42) -> np.ndarray:
    return _tsne(emb, 3, seed)


def cluster(emb: np.ndarray, distance_threshold: float = 0.4) -> np.ndarray:
    """Agglomerative clustering on the full embedding space with cosine
    distance. `distance_threshold` auto-picks K: smaller -> more, tighter
    clusters; larger -> fewer, looser. Noise is not produced (no -1 labels).
    """
    from sklearn.cluster import AgglomerativeClustering

    n = len(emb)
    if n < 2:
        return np.zeros(n, dtype=np.int32)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(emb).astype(np.int32)


def centroid_indices(emb: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """For each cluster, return the row index of the note closest to the
    cluster's mean embedding (cosine similarity; vectors are unit-normalized).
    """
    result: dict[int, int] = {}
    for lab in sorted(set(labels.tolist()) - {-1}):
        idx = np.where(labels == lab)[0]
        sub = emb[idx]
        mean = sub.mean(axis=0)
        mean /= np.linalg.norm(mean) + 1e-12
        sims = sub @ mean
        result[int(lab)] = int(idx[int(np.argmax(sims))])
    return result
