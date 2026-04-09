from pathlib import Path

import pandas as pd


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

    if len(docs) < 3:
        return _fallback_topics(video_ids, output_csv)

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from umap import UMAP

        embedding_model = SentenceTransformer(model_name)

        n_neighbors = max(2, min(10, len(docs) - 1))
        n_components = max(2, min(5, len(docs) - 1))
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric="cosine",
            random_state=42,
        )

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            language="multilingual",
            verbose=False,
            min_topic_size=2,
        )

        topics, probs = topic_model.fit_transform(docs)

        rows = []
        for idx, video_id in enumerate(video_ids):
            score = float(probs[idx].max()) if probs is not None and len(probs[idx]) else 0.0
            rows.append({"video_id": video_id, "topic": int(topics[idx]), "score": score})

        out = pd.DataFrame(rows)
    except Exception:
        out = _fallback_topics(video_ids, output_csv)
        return out

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    return out
