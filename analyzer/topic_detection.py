from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def extract_topics(video_texts: dict[str, str], model_name: str, output_csv: Path):
    if not video_texts:
        return pd.DataFrame(columns=["video_id", "topic", "score"])

    video_ids = list(video_texts.keys())
    docs = [video_texts[v] if video_texts[v].strip() else "leer" for v in video_ids]

    embedding_model = SentenceTransformer(model_name)
    topic_model = BERTopic(embedding_model=embedding_model, language="multilingual", verbose=False)

    topics, probs = topic_model.fit_transform(docs)

    rows = []
    for idx, video_id in enumerate(video_ids):
        score = float(probs[idx].max()) if probs is not None and len(probs[idx]) else 0.0
        rows.append({"video_id": video_id, "topic": int(topics[idx]), "score": score})

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    return out
