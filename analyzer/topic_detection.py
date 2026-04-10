from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def _fallback_topics(video_ids: list[str], output_csv: Path):
    out = pd.DataFrame(
        [{"video_id": video_id, "topic": 0, "score": 1.0} for video_id in video_ids]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    return out


def extract_topics(video_texts: dict[str, str], model_name: str, output_csv: Path):
    if not video_texts:
        out = pd.DataFrame(columns=["video_id", "topic", "score"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False, encoding="utf-8")
        return out

    video_ids = list(video_texts.keys())
    docs = [video_texts[v].strip() if video_texts[v].strip() else "leer" for v in video_ids]

    if len(docs) < 2:
        return _fallback_topics(video_ids, output_csv)

    try:
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(model_name, device="cpu")
        embeddings = embedding_model.encode(docs, normalize_embeddings=True)
        n_clusters = max(2, min(6, len(docs)))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        topics = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_

        rows = []
        for idx, video_id in enumerate(video_ids):
            center = centers[int(topics[idx])]
            score = float(cosine_similarity([embeddings[idx]], [center])[0][0])
            rows.append({"video_id": video_id, "topic": int(topics[idx]), "score": score})

        out = pd.DataFrame(rows)
    except Exception:
        out = _fallback_topics(video_ids, output_csv)
        return out

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    return out
