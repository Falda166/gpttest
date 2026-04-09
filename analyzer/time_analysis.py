from pathlib import Path

import pandas as pd
import plotly.express as px


def word_frequency_over_time(transcript_words: list[dict], output_html: Path):
    """transcript_words expects [{'start': float, 'word': str}, ...]."""
    if not transcript_words:
        return None

    df = pd.DataFrame(transcript_words)
    if "start" not in df or "word" not in df:
        return None

    df["bucket"] = df["start"].astype(float).floordiv(30).mul(30)
    agg = df.groupby("bucket").size().reset_index(name="frequency")

    fig = px.line(agg, x="bucket", y="frequency", markers=True, title="Word Frequency Over Time")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    return fig
