import numpy as np
import pandas as pd

from analyzer.helpers import cosine_similarity


def build_overlap_regions_from_diarization(speaker_diarization):
    boundaries = []

    tracks = list(speaker_diarization.itertracks(yield_label=True))
    for segment, _, speaker in tracks:
        boundaries.append((float(segment.start), "start", speaker))
        boundaries.append((float(segment.end), "end", speaker))

    boundaries.sort(key=lambda x: (x[0], 0 if x[1] == "end" else 1))

    active = set()
    overlap_start = None
    overlap_regions = []

    for t, kind, speaker in boundaries:
        if kind == "start":
            active.add(speaker)
            if len(active) >= 2 and overlap_start is None:
                overlap_start = t
        else:
            if len(active) >= 2 and overlap_start is not None:
                overlap_regions.append((overlap_start, t))
                overlap_start = None
            active.discard(speaker)

    return overlap_regions


def build_diarization_df(annotation):
    rows = []
    for segment, track, speaker in annotation.itertracks(yield_label=True):
        rows.append({
            "segment": segment,
            "label": track,
            "speaker": speaker,
            "start": float(segment.start),
            "end": float(segment.end),
        })
    return pd.DataFrame(rows)


def is_overlapped(start, end, overlap_regions):
    mid = (start + end) / 2.0
    for os_, oe_ in overlap_regions:
        if os_ <= mid <= oe_:
            return True
    return False


def normalize_embedding(emb):
    emb = np.asarray(emb, dtype=np.float32).squeeze()
    if emb.ndim != 1:
        raise ValueError(f"Ungültiges Embedding: {emb.shape}")
    if not np.isfinite(emb).all():
        raise ValueError("Embedding enthält NaN/Inf")
    norm = np.linalg.norm(emb)
    if norm == 0:
        raise ValueError("Embedding-Norm ist 0")
    return emb / norm


def collect_speaker_embeddings(annotation, waveform, sr, embedder, ref_shape, min_seconds=1.5, device=None):
    speaker_embs = {}

    tracks = list(annotation.itertracks(yield_label=True))
    for segment, _, speaker in tracks:
        s = int(segment.start * sr)
        e = int(segment.end * sr)
        chunk = waveform[:, s:e]

        if chunk.shape[1] < sr * min_seconds:
            continue

        sample = {"waveform": chunk.to(device) if device else chunk, "sample_rate": sr}

        try:
            emb = embedder(sample)
            emb = normalize_embedding(emb)
        except Exception:
            continue

        if emb.shape != ref_shape:
            continue

        speaker_embs.setdefault(speaker, []).append(emb)

    return speaker_embs


def choose_best_matching_speaker(speaker_embs, ref_emb):
    best_speaker = None
    best_score = -1.0

    for speaker, embs in speaker_embs.items():
        if not embs:
            continue

        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        mean_emb = normalize_embedding(mean_emb)
        score = cosine_similarity(mean_emb, ref_emb)

        if score > best_score:
            best_score = score
            best_speaker = speaker

    return best_speaker, best_score
