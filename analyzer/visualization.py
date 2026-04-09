from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import umap


def visualize_word_embeddings(words: list[str], embeddings: np.ndarray, output_html: Path):
    if not words:
        return None

    if len(words) < 3:
        points = np.column_stack([
            np.arange(len(words), dtype=np.float32),
            np.zeros(len(words), dtype=np.float32),
        ])
    else:
        n_neighbors = max(2, min(15, len(words) - 1))
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=42,
            n_neighbors=n_neighbors,
        )
        points = reducer.fit_transform(embeddings)

    df = pd.DataFrame({"word": words, "x": points[:, 0], "y": points[:, 1]})
    fig = px.scatter(df, x="x", y="y", text="word", hover_data=["word"], title="Word Embedding Clusters")
    fig.update_traces(textposition="top center")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    return fig
