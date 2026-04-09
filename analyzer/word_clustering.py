from collections import Counter, defaultdict

import hdbscan
import numpy as np

from analyzer.embedding_cache import EmbeddingCache


def cluster_words(words: list[str], embedding_cache: EmbeddingCache, min_cluster_size: int = 2):
    if not words:
        return [], np.array([], dtype=int)

    unique_words = list(dict.fromkeys(words))
    if len(unique_words) < 2:
        return unique_words, np.array([-1] * len(unique_words), dtype=int)

    min_cluster_size = max(2, min(min_cluster_size, len(unique_words)))

    embeddings = embedding_cache.encode(unique_words)
    similarity = embeddings @ embeddings.T
    distance = 1.0 - np.clip(similarity, -1.0, 1.0)

    # hdbscan (cython path) erwartet float64 / contiguous für precomputed distances
    distance = np.asarray(distance, dtype=np.float64, order="C")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
    labels = clusterer.fit_predict(distance)
    return unique_words, labels


def normalize_words(words: list[str], embedding_cache: EmbeddingCache, min_cluster_size: int = 2):
    if not words:
        return []

    counts = Counter(words)
    unique_words, labels = cluster_words(words, embedding_cache, min_cluster_size=min_cluster_size)

    grouped = defaultdict(list)
    for w, lbl in zip(unique_words, labels):
        grouped[lbl].append(w)

    representative = {}
    for lbl, cluster_words_list in grouped.items():
        if lbl == -1:
            for w in cluster_words_list:
                representative[w] = w
            continue

        best = max(cluster_words_list, key=lambda w: (counts[w], -len(w)))
        for w in cluster_words_list:
            representative[w] = best

    return [representative.get(w, w) for w in words]
