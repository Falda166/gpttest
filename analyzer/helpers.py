import json
import re
from pathlib import Path

import numpy as np


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-\.]+", "_", name)
    return name[:120]


def read_links_from_txt(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Link-Datei nicht gefunden: {path}")

    links = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line.startswith(("http://", "https://")):
                links.append(line)

    if not links:
        raise ValueError("Keine gültigen Links in der TXT-Datei gefunden.")

    return links


def extract_embedding(entry, name="unknown"):
    if isinstance(entry, dict):
        if "embedding" in entry:
            entry = entry["embedding"]
        else:
            raise ValueError(f"Eintrag '{name}' ist dict ohne 'embedding'-Key")

    if isinstance(entry, str):
        try:
            entry = json.loads(entry)
        except Exception as e:
            raise ValueError(f"Eintrag '{name}' ist String, aber kein parsebares JSON") from e

    arr = np.asarray(entry, dtype=np.float32).squeeze()

    if arr.ndim != 1:
        raise ValueError(f"Embedding '{name}' ist nicht 1D. Shape: {arr.shape}")

    if arr.size < 10:
        raise ValueError(f"Embedding '{name}' ist zu klein. Shape: {arr.shape}")

    if not np.isfinite(arr).all():
        raise ValueError(f"Embedding '{name}' enthält NaN/Inf")

    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError(f"Embedding '{name}' hat Norm 0")

    return arr / norm


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("Mindestens einer der Vektoren hat Norm 0")

    return float(np.dot(a, b) / (na * nb))
