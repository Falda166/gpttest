import os
import re
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import yt_dlp
import whisperx

from colorama import init, Fore, Style
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import SpeakerEmbedding


# =========================
# COLORAMA INIT
# =========================
init(autoreset=True)
load_dotenv()


# =========================
# CONFIG
# =========================
INPUT_LINKS_FILE = Path("./youtube_links.txt")

OUTPUT_DIR = Path("./output")
AUDIO_DIR = OUTPUT_DIR / "audios"
OUTPUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

DB_FILE = "voice_db.json"
FINAL_CSV_FILE = OUTPUT_DIR / "word_frequency.csv"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN fehlt. Bitte als Umgebungsvariable setzen.")

WHISPERX_MODEL = "medium"   # tiny / base / small / medium / large-v2 / large-v3
WHISPER_LANGUAGE = "de"

MATCH_THRESHOLD_STRONG = 0.55
MATCH_THRESHOLD_MAYBE = 0.40
MIN_ACCEPT_SCORE = 0.40

MIN_SEGMENT_SECONDS = 1.5
AUDIO_OVERWRITE = True


# =========================
# DEVICE
# =========================
device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"


# =========================
# LOGGING
# =========================
def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = seconds % 60
    return f"{minutes}m {rest:.1f}s"


def log_info(msg: str):
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {msg}")


def log_step(msg: str):
    print(f"{Fore.BLUE}[STEP]{Style.RESET_ALL} {msg}")


def log_ok(msg: str):
    print(f"{Fore.GREEN}[ OK ]{Style.RESET_ALL} {msg}")


def log_warn(msg: str):
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {msg}")


def log_error(msg: str):
    print(f"{Fore.RED}[ERR ]{Style.RESET_ALL} {msg}")


def timed_step(label: str, func, *args, **kwargs):
    log_step(label)
    t0 = time.time()
    result = func(*args, **kwargs)
    dt = time.time() - t0
    log_ok(f"{label} fertig in {fmt_seconds(dt)}")
    return result


# =========================
# HELPERS
# =========================
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


def load_audio_tensor(path):
    waveform, sr = sf.read(path, always_2d=True, dtype="float32")
    waveform = torch.tensor(waveform.T, dtype=torch.float32)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, sr


def download_audio(url: str, target_wav: Path, overwrite=True):
    if target_wav.exists() and not overwrite:
        return

    stem = target_wav.stem
    outtmpl = str(target_wav.parent / f"{sanitize_filename(stem)}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not target_wav.exists():
        wav_candidates = list(target_wav.parent.glob(f"{sanitize_filename(stem)}*.wav"))
        if wav_candidates:
            wav_candidates[0].rename(target_wav)

    if not target_wav.exists():
        raise FileNotFoundError(f"Audio-Datei wurde nicht erzeugt: {target_wav}")


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


def collect_speaker_embeddings(annotation, waveform, sr, embedder, ref_shape, min_seconds=1.5):
    speaker_embs = {}

    tracks = list(annotation.itertracks(yield_label=True))
    for segment, _, speaker in tracks:
        s = int(segment.start * sr)
        e = int(segment.end * sr)
        chunk = waveform[:, s:e]

        if chunk.shape[1] < sr * min_seconds:
            continue

        sample = {
            "waveform": chunk.to(device_torch),
            "sample_rate": sr
        }

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


def classify_score(score):
    if score >= MATCH_THRESHOLD_STRONG:
        return "PAPAPLATTE"
    if score >= MATCH_THRESHOLD_MAYBE:
        return "MAYBE"
    return "unknown"


def clean_word(word):
    return word.strip()


def normalize_word_for_counting(word: str) -> str:
    word = word.strip().lower()
    word = re.sub(r"^[^\wäöüß]+|[^\wäöüß]+$", "", word, flags=re.IGNORECASE)
    word = re.sub(r"\s+", " ", word)
    return word


def cleanup_audio_files(audio_dir: Path, idx: int, exact_audio_file: Path):
    if exact_audio_file.exists():
        try:
            exact_audio_file.unlink()
        except Exception:
            pass

    for extra_file in audio_dir.glob(f"audio_{idx}*"):
        try:
            if extra_file.exists():
                extra_file.unlink()
        except Exception:
            pass


def process_single_video(
    url: str,
    audio_file: Path,
    papaplatte_emb,
    whisperx_model,
    align_model,
    align_metadata,
    diar,
    embedder,
):
    video_start = time.time()
    log_info(f"Verarbeite: {url}")

    timed_step("Audio herunterladen", download_audio, url, audio_file, AUDIO_OVERWRITE)

    audio_for_whisperx = timed_step(
        "Audio für WhisperX laden",
        whisperx.load_audio,
        str(audio_file),
    )

    asr_result = timed_step(
        "WhisperX Transkription",
        whisperx_model.transcribe,
        audio_for_whisperx,
        batch_size=16,
        language=WHISPER_LANGUAGE,
    )

    aligned_result = timed_step(
        "WhisperX Alignment",
        whisperx.align,
        asr_result["segments"],
        align_model,
        align_metadata,
        audio_for_whisperx,
        device_str,
    )

    waveform, sr = timed_step("Audio-Tensor laden", load_audio_tensor, audio_file)

    audio_dict = {
        "waveform": waveform.to(device_torch),
        "sample_rate": sr
    }

    diar_output = timed_step("Speaker Diarization", diar, audio_dict)
    speaker_diarization = diar_output.speaker_diarization
    annotation = diar_output.exclusive_speaker_diarization

    diar_df = timed_step("Diarization DataFrame bauen", build_diarization_df, annotation)
    overlap_regions = timed_step(
        "Overlap-Regionen berechnen",
        build_overlap_regions_from_diarization,
        speaker_diarization,
    )

    result_with_speakers = timed_step(
        "Wörter Speakern zuordnen",
        whisperx.assign_word_speakers,
        diar_df,
        aligned_result,
    )

    speaker_embs = timed_step(
        "Speaker-Embeddings sammeln",
        collect_speaker_embeddings,
        annotation,
        waveform,
        sr,
        embedder,
        papaplatte_emb.shape,
        MIN_SEGMENT_SECONDS,
    )

    best_speaker, best_score = timed_step(
        "Besten Speaker bestimmen",
        choose_best_matching_speaker,
        speaker_embs,
        papaplatte_emb,
    )

    if best_speaker is None:
        log_warn("Kein passender Speaker gefunden, Video wird übersprungen.")
        return []

    if best_score < MIN_ACCEPT_SCORE:
        log_warn(f"Score zu niedrig ({best_score:.3f}) -> Video übersprungen")
        return []

    label = classify_score(best_score)
    log_ok(f"Match: {best_speaker} ({best_score:.3f}, {label})")

    accepted_words = []

    log_step("Wörter filtern und zählen vorbereiten")
    t0 = time.time()

    for seg in result_with_speakers["segments"]:
        seg_speaker = seg.get("speaker")

        for word in seg.get("words", []):
            if "start" not in word or "end" not in word:
                continue

            word_speaker = word.get("speaker", seg_speaker)
            if word_speaker != best_speaker:
                continue

            if is_overlapped(word["start"], word["end"], overlap_regions):
                continue

            txt = clean_word(word.get("word", ""))
            txt = normalize_word_for_counting(txt)

            if not txt:
                continue

            accepted_words.append(txt)

    dt = time.time() - t0
    log_ok(f"Wörter gefiltert in {fmt_seconds(dt)} | akzeptiert: {len(accepted_words)}")

    total_video_time = time.time() - video_start
    log_ok(f"Video komplett fertig in {fmt_seconds(total_video_time)}")

    return accepted_words


# =========================
# MAIN
# =========================
def main():
    total_start = time.time()

    log_info("Starte Verarbeitung")
    log_info(f"Device: {device_str} | compute_type: {compute_type}")

    links = timed_step("Links aus TXT laden", read_links_from_txt, INPUT_LINKS_FILE)
    log_info(f"{len(links)} Links geladen")

    log_step("voice_db.json laden")
    t0 = time.time()
    with open(DB_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)
    dt = time.time() - t0
    log_ok(f"voice_db.json geladen in {fmt_seconds(dt)}")

    if "papaplatte" not in db:
        raise KeyError("'papaplatte' nicht in voice_db.json gefunden")

    papaplatte_emb = timed_step("Referenz-Embedding laden", extract_embedding, db["papaplatte"], "papaplatte")

    whisperx_model = timed_step(
        "WhisperX Modell laden",
        whisperx.load_model,
        WHISPERX_MODEL,
        device_str,
        compute_type=compute_type,
        language=WHISPER_LANGUAGE,
    )

    align_model, align_metadata = timed_step(
        "Alignment-Modell laden",
        whisperx.load_align_model,
        language_code=WHISPER_LANGUAGE,
        device=device_str,
    )

    diar = timed_step(
        "pyannote Diarization Pipeline laden",
        Pipeline.from_pretrained,
        "pyannote/speaker-diarization-community-1",
        token=HF_TOKEN,
    )
    diar.to(device_torch)
    log_ok("pyannote Pipeline auf Device verschoben")

    embedder = timed_step(
        "SpeakerEmbedding laden",
        SpeakerEmbedding,
        token=HF_TOKEN,
    )
    if hasattr(embedder, "to"):
        embedder.to(device_torch)
        log_ok("SpeakerEmbedding auf Device verschoben")

    total_counter = Counter()
    total_links = len(links)

    for idx, url in enumerate(links, start=1):
        loop_start = time.time()
        print()
        log_info(f"========== VIDEO {idx}/{total_links} ==========")

        audio_file = AUDIO_DIR / f"audio_{idx}.wav"

        try:
            words = process_single_video(
                url=url,
                audio_file=audio_file,
                papaplatte_emb=papaplatte_emb,
                whisperx_model=whisperx_model,
                align_model=align_model,
                align_metadata=align_metadata,
                diar=diar,
                embedder=embedder,
            )

            total_counter.update(words)
            log_ok(f"Aktueller Wortschatz: {len(total_counter)} unique Wörter")

        except Exception as e:
            log_error(f"Fehler bei Link {idx}: {e}")

        finally:
            cleanup_audio_files(AUDIO_DIR, idx, audio_file)
            log_ok("Temporäre Audio-Dateien gelöscht")

        loop_dt = time.time() - loop_start
        log_info(f"Video {idx}/{total_links} abgeschlossen in {fmt_seconds(loop_dt)}")

    log_step("CSV schreiben")
    t0 = time.time()

    df = pd.DataFrame(
        [{"word": word, "count": count} for word, count in total_counter.items()]
    )

    if not df.empty:
        df = df.sort_values(by=["count", "word"], ascending=[False, True]).reset_index(drop=True)

    df.to_csv(FINAL_CSV_FILE, index=False, encoding="utf-8")

    dt = time.time() - t0
    log_ok(f"CSV geschrieben in {fmt_seconds(dt)} -> {FINAL_CSV_FILE}")

    total_dt = time.time() - total_start
    print()
    log_ok(f"Gesamt fertig in {fmt_seconds(total_dt)}")
    log_info(f"Unique Wörter gesamt: {len(total_counter)}")


if __name__ == "__main__":
    main()
