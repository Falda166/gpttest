from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

OUTPUT = Path("output")
CSV_DIR = OUTPUT / "csv"

st.set_page_config(page_title="NLP Video Analysis", layout="wide")
st.title("NLP Video Analysis Toolkit")


def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


word_df = safe_read_csv(OUTPUT / "word_frequency.csv")
cluster_df = word_df.copy()
topics_df = safe_read_csv(CSV_DIR / "video_topics.csv")
speaker_df = safe_read_csv(CSV_DIR / "speaker_style.csv")
sim_df = safe_read_csv(CSV_DIR / "video_similarity.csv",)


tabs = st.tabs([
    "Word Frequency",
    "Word Clusters",
    "Topics",
    "Speaker Analysis",
    "Video Similarity",
    "Timeline Analysis",
])

with tabs[0]:
    st.subheader("Top Words")
    if word_df.empty:
        st.info("Noch keine Daten vorhanden.")
    else:
        n = st.slider("Anzahl Wörter", 10, 200, 30)
        st.dataframe(word_df.head(n), use_container_width=True)
        fig = px.bar(word_df.head(n), x="word", y="count", title="Top Wörter")
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Embedding Visualization")
    html_path = OUTPUT / "word_clusters.html"
    if html_path.exists():
        st.markdown(f"[Interaktive Cluster-Ansicht öffnen]({html_path.as_posix()})")
    if cluster_df.empty:
        st.info("Keine Cluster-Daten vorhanden.")
    else:
        st.dataframe(cluster_df.head(100), use_container_width=True)

with tabs[2]:
    st.subheader("Topics pro Video")
    if topics_df.empty:
        st.info("Keine Topic-Daten vorhanden.")
    else:
        st.dataframe(topics_df, use_container_width=True)

with tabs[3]:
    st.subheader("Vocabulary Vergleich")
    if speaker_df.empty:
        st.info("Keine Speaker-Analyse vorhanden.")
    else:
        st.dataframe(speaker_df.head(200), use_container_width=True)

with tabs[4]:
    st.subheader("Similarity Matrix")
    sim_matrix_path = CSV_DIR / "video_similarity.csv"
    if sim_matrix_path.exists():
        sim_matrix = pd.read_csv(sim_matrix_path, index_col=0)
        st.dataframe(sim_matrix, use_container_width=True)
        heatmap = px.imshow(sim_matrix.values, x=sim_matrix.columns, y=sim_matrix.index, color_continuous_scale="Viridis")
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Keine Similarity-Daten vorhanden.")

with tabs[5]:
    st.subheader("Word Usage Over Time")
    timeline_path = OUTPUT / "word_timeline.html"
    if timeline_path.exists():
        st.markdown(f"[Timeline öffnen]({timeline_path.as_posix()})")
    else:
        st.info("Keine Timeline vorhanden.")
