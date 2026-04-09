import json
import time
from collections import Counter

import pandas as pd
import torch
import whisperx
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import SpeakerEmbedding

from analyzer import config
from analyzer.audio_processing import cleanup_audio_files
from analyzer.helpers import read_links_from_txt, extract_embedding
from analyzer.logging_utils import timed_step, log_info, log_ok, log_error, fmt_seconds, log_step
from analyzer.pipeline import process_single_video
from analyzer.csv_cleanup import CsvCleaner

load_dotenv()


def main():
    total_start = time.time()

    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    log_info("Starte Verarbeitung")
    log_info(f"Device: {device_str} | compute_type: {compute_type}")

    links = timed_step("Links aus TXT laden", read_links_from_txt, config.INPUT_LINKS_FILE)
    log_info(f"{len(links)} Links geladen")

    log_step("voice_db.json laden")
    t0 = time.time()
    with open(config.DB_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)
    dt = time.time() - t0
    log_ok(f"voice_db.json geladen in {fmt_seconds(dt)}")

    if "papaplatte" not in db:
        raise KeyError("'papaplatte' nicht in voice_db.json gefunden")

    papaplatte_emb = timed_step("Referenz-Embedding laden", extract_embedding, db["papaplatte"], "papaplatte")

    whisperx_model = timed_step(
        "WhisperX Modell laden",
        whisperx.load_model,
        config.WHISPERX_MODEL,
        device_str,
        compute_type=compute_type,
        language=config.WHISPER_LANGUAGE,
    )

    align_model, align_metadata = timed_step(
        "Alignment-Modell laden",
        whisperx.load_align_model,
        language_code=config.WHISPER_LANGUAGE,
        device=device_str,
    )

    diar = timed_step(
        "pyannote Diarization Pipeline laden",
        Pipeline.from_pretrained,
        "pyannote/speaker-diarization-community-1",
        token=config.HF_TOKEN,
    )
    diar.to(device_torch)
    log_ok("pyannote Pipeline auf Device verschoben")

    embedder = timed_step(
        "SpeakerEmbedding laden",
        SpeakerEmbedding,
        token=config.HF_TOKEN,
    )
    if hasattr(embedder, "to"):
        embedder.to(device_torch)
        log_ok("SpeakerEmbedding auf Device verschoben")

    csv_cleaner = timed_step(
        "CSV-Cleaner Modell laden",
        CsvCleaner,
        config.CSV_CLEANUP_MODEL,
        config.CSV_SEMANTIC_THRESHOLD,
    )

    total_counter = Counter()
    total_links = len(links)

    video_texts = {}
    timeline_words = []
    global_speaker_counts = defaultdict(Counter)

    for idx, url in enumerate(links, start=1):
        loop_start = time.time()
        print()
        log_info(f"========== VIDEO {idx}/{total_links} ==========")

        raw_audio_file = config.AUDIO_DIR / f"audio_{idx}.wav"
        cleaned_audio_file = config.CLEAN_AUDIO_DIR / f"audio_{idx}_cleaned.wav"

        try:
            result = process_single_video(
                url=url,
                raw_audio_file=raw_audio_file,
                cleaned_audio_file=cleaned_audio_file,
                papaplatte_emb=papaplatte_emb,
                whisperx_model=whisperx_model,
                align_model=align_model,
                align_metadata=align_metadata,
                diar=diar,
                embedder=embedder,
                device_torch=device_torch,
                device_str=device_str,
            )

            words = result["words"]
            total_counter.update(words)
            video_texts[f"video_{idx}"] = result["transcript"]
            timeline_words.extend(result["timed_words"])

            for speaker, counts in result["speaker_word_counts"].items():
                global_speaker_counts[speaker].update(counts)

            log_ok(f"Aktueller Wortschatz: {len(total_counter)} unique Wörter")

        except Exception as e:
            log_error(f"Fehler bei Link {idx}: {e}")

        finally:
            cleanup_audio_files(raw_audio_file, cleaned_audio_file)
            log_ok("Temporäre Audio-Dateien gelöscht")

        loop_dt = time.time() - loop_start
        log_info(f"Video {idx}/{total_links} abgeschlossen in {fmt_seconds(loop_dt)}")

    log_step("CSV schreiben")
    t0 = time.time()

    df = pd.DataFrame([{"word": word, "count": count} for word, count in total_counter.items()])

    if not df.empty:
        df = timed_step("CSV Basis-Bereinigung", csv_cleaner.basic_cleanup, df)
        df = timed_step("CSV Semantik-Bereinigung", csv_cleaner.semantic_cleanup, df)
        df = df.sort_values(by=["count", "word"], ascending=[False, True]).reset_index(drop=True)

    df.to_csv(config.FINAL_CSV_FILE, index=False, encoding="utf-8")

    dt = time.time() - t0
    log_ok(f"CSV geschrieben in {fmt_seconds(dt)} -> {config.FINAL_CSV_FILE}")

    total_dt = time.time() - total_start
    print()
    log_ok(f"Gesamt fertig in {fmt_seconds(total_dt)}")
    log_info(f"Unique Wörter gesamt: {len(total_counter)}")


if __name__ == "__main__":
    main()
