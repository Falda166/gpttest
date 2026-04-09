import numpy as np
import pandas as pd

from analyzer.embedding_cache import EmbeddingCache


class CsvCleaner:
    def __init__(self, model_name: str, semantic_threshold: float = 0.87, embedding_cache: EmbeddingCache | None = None):
        self.model_name = model_name
        self.semantic_threshold = semantic_threshold
        self.embedding_cache = embedding_cache or EmbeddingCache(
            model_name=model_name,
            cache_path=None,  # type: ignore[arg-type]
        )

    def basic_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        cleaned = df.copy()
        cleaned["word"] = (
            cleaned["word"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
        )
        cleaned = cleaned[cleaned["word"].str.len() > 0]
        cleaned = cleaned.groupby("word", as_index=False)["count"].sum()
        return cleaned

    def semantic_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) <= 1:
            return df

        words = df["word"].tolist()
        counts = df["count"].to_numpy()

        embeddings = self.embedding_cache.encode(words)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        parent = list(range(len(words)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        sim = embeddings @ embeddings.T
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if sim[i, j] >= self.semantic_threshold:
                    union(i, j)

        merged_rows = []
        groups = {}
        for idx in range(len(words)):
            root = find(idx)
            groups.setdefault(root, []).append(idx)

        for indices in groups.values():
            canonical_idx = max(indices, key=lambda x: (counts[x], -len(words[x])))
            canonical_word = words[canonical_idx]
            merged_count = int(sum(counts[i] for i in indices))
            merged_rows.append({"word": canonical_word, "count": merged_count})

        out = pd.DataFrame(merged_rows)
        out = out.sort_values(by=["count", "word"], ascending=[False, True]).reset_index(drop=True)
        return out
