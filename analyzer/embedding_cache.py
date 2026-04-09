import hashlib
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingCache:
    def __init__(self, model_name: str, cache_path: Path | None, batch_size: int = 64):
        self.model_name = model_name
        self.cache_path = cache_path
        self.batch_size = batch_size
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()
        self._model = None

    def _load_cache(self):
        if self.cache_path is not None and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self):
        if self.cache_path is None:
            return
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._cache, f)

    def _get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}::{text}".encode("utf-8")).hexdigest()

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        model = self._get_model()

        missing_texts = []
        missing_keys = []
        for text in texts:
            key = self._key(text)
            if key not in self._cache:
                missing_texts.append(text)
                missing_keys.append(key)

        if missing_texts:
            vectors = model.encode(
                missing_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize,
            )
            for key, vec in zip(missing_keys, vectors):
                self._cache[key] = np.asarray(vec, dtype=np.float32)
            self._save_cache()

        return np.asarray([self._cache[self._key(t)] for t in texts], dtype=np.float32)
