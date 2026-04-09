from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import umap


def visualize_word_embeddings(words: list[str], embeddings: np.ndarray, output_html: Path):
    if not words:
        return None

    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    points = reducer.fit_transform(embeddings)

    df = pd.DataFrame({"word": words, "x": points[:, 0], "y": points[:, 1]})
    fig = px.scatter(df, x="x", y="y", text="word", hover_data=["word"], title="Word Embedding Clusters")
    fig.update_traces(textposition="top center")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    return fig
