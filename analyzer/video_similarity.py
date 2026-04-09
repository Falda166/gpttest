from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from analyzer.embedding_cache import EmbeddingCache


def compute_video_similarity(video_texts: dict[str, str], embedding_cache: EmbeddingCache, output_csv: Path):
    if not video_texts:
        return pd.DataFrame()

    video_ids = list(video_texts.keys())
    texts = [video_texts[v] if video_texts[v].strip() else "leer" for v in video_ids]
    embeddings = embedding_cache.encode(texts)

    sim = cosine_similarity(embeddings)
    df = pd.DataFrame(sim, index=video_ids, columns=video_ids)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, encoding="utf-8")
    return df
