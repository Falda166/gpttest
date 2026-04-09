import json
import os
import time
from collections import Counter

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import whisperx
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import SpeakerEmbedding

from analyzer import config
from analyzer.audio_processing import cleanup_audio_files, fetch_video_duration_seconds
from analyzer.csv_cleanup import CsvCleaner
from analyzer.embedding_cache import EmbeddingCache
from analyzer.helpers import read_links_from_txt, extract_embedding
from analyzer.logging_utils import (
    configure_console,
    finish_console,
    fmt_seconds,
    get_console_manager,
    log_error,
    log_info,
    log_ok,
    log_step,
    log_warn,
    timed_step,
)
from analyzer.pipeline import process_single_video
from analyzer.progress_tracking import RuntimeEstimator
from analyzer.runtime import configure_runtime, resolve_device
from analyzer.speaker_style import compare_speakers
from analyzer.time_analysis import word_frequency_over_time
from analyzer.topic_detection import extract_topics
from analyzer.video_similarity import compute_video_similarity
from analyzer.visualization import visualize_word_embeddings
from analyzer.word_clustering import normalize_words

load_dotenv()


def run_optional_step(label: str, func, *args, **kwargs):
    try:
        return timed_step(label, func, *args, **kwargs)
    except Exception as e:
        log_warn(f"{label} übersprungen: {e}")
        return None


def main():
    total_start = time.time()
    configure_console(
        footer_enabled=config.TERMINAL_FOOTER_ENABLED,
        refresh_interval=config.TERMINAL_FOOTER_REFRESH_SECONDS,
        footer_height=config.TERMINAL_FOOTER_HEIGHT,
    )
    console = get_console_manager()

    try:
        configure_runtime(config.QUIET_THIRD_PARTY_WARNINGS, config.ENABLE_TF32)
        device_torch, device_str, compute_type = resolve_device()

        log_info("Starte Verarbeitung")
        log_info(f"Device: {device_str} | compute_type: {compute_type}")

        links = timed_step("Links aus TXT laden", read_links_from_txt, config.INPUT_LINKS_FILE)
        total_links = len(links)
        log_info(f"{total_links} Links geladen")
        console.start_run(total_videos=total_links)

        durations = []
        log_step("Video-Längen laden (für ETA)")
        for i, link in enumerate(links, start=1):
            try:
                d = fetch_video_duration_seconds(link)
                durations.append(d)
            except Exception:
                durations.append(None)
                log_warn(f"Länge für Video {i} konnte nicht geladen werden")
        known_durations = sum(1 for d in durations if d is not None)
        log_ok(f"Video-Längen geladen: {known_durations}/{total_links} bekannt")
        runtime_estimator = RuntimeEstimator(total_videos=total_links, planned_durations_seconds=durations)
        console.set_progress(runtime_estimator.snapshot(0, time.time() - total_start))

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

        embedding_cache = timed_step(
            "Embedding Cache initialisieren",
            EmbeddingCache,
            config.CSV_CLEANUP_MODEL,
            config.EMBEDDINGS_CACHE_FILE,
        )

        csv_cleaner = timed_step(
            "CSV-Cleaner Modell laden",
            CsvCleaner,
            config.CSV_CLEANUP_MODEL,
            config.CSV_SEMANTIC_THRESHOLD,
            embedding_cache,
        )

        total_counter = Counter()
        video_texts = {}
        timeline_words = []
        global_speaker_counts = {}

        for idx, url in enumerate(links, start=1):
            loop_start = time.time()
            console.set_video(
                video_idx=idx,
                total_videos=total_links,
                url=url,
                expected_processing_seconds=runtime_estimator.estimate_processing_seconds_for_video(durations[idx - 1]),
            )
            console.set_step("Video vorbereiten")
            log_info(f"========== VIDEO {idx}/{total_links} ==========")

            raw_audio_file = config.AUDIO_DIR / f"audio_{idx}.wav"
            cleaned_audio_file = config.CLEAN_AUDIO_DIR / f"audio_{idx}_cleaned.wav"
            target_audio_file = config.CLEAN_AUDIO_DIR / f"audio_{idx}_cleaned_target.wav"
            result = None

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
                    return_metadata=True,
                )

                words = result["words"]
                total_counter.update(words)
                video_texts[f"video_{idx}"] = result["transcript"]
                timeline_words.extend(result["timed_words"])

                for speaker, counts in result["speaker_word_counts"].items():
                    if speaker not in global_speaker_counts:
                        global_speaker_counts[speaker] = Counter()
                    global_speaker_counts[speaker].update(counts)

                log_ok(f"Aktueller Wortschatz: {len(total_counter)} unique Wörter")

            except Exception as e:
                log_error(f"Fehler bei Link {idx}: {e}")

            finally:
                cleanup_audio_files(raw_audio_file, cleaned_audio_file, target_audio_file)
                log_ok("Temporäre Audio-Dateien gelöscht")

            loop_dt = time.time() - loop_start
            video_seconds = durations[idx - 1]
            if video_seconds is None:
                video_seconds = float(result.get("source_audio_seconds", 0.0)) if isinstance(result, dict) else 0.0
            runtime_estimator.update(
                video_idx=idx,
                video_seconds=video_seconds,
                processing_seconds=loop_dt,
            )
            total_elapsed = time.time() - total_start
            console.set_progress(runtime_estimator.snapshot(idx, total_elapsed))
            if not console.is_footer_active():
                log_info(runtime_estimator.render_progress_line(idx, total_elapsed))
                log_info(f"Laufzeitformel aktuell: {runtime_estimator.formula_text()}")
            log_info(f"Video {idx}/{total_links} abgeschlossen in {fmt_seconds(loop_dt)}")

        console.set_video(None, total_links)
        log_step("CSV schreiben")
        t0 = time.time()

        df = pd.DataFrame([{"word": word, "count": count} for word, count in total_counter.items()])

        if not df.empty:
            df = timed_step("CSV Basis-Bereinigung", csv_cleaner.basic_cleanup, df)
            df = timed_step("CSV Semantik-Bereinigung", csv_cleaner.semantic_cleanup, df)

            normalized_words = timed_step(
                "Wort-Clustering + Normalisierung",
                normalize_words,
                list(df["word"]),
                embedding_cache,
                config.WORD_CLUSTER_MIN_SIZE,
            )
            df["word"] = normalized_words
            df = df.groupby("word", as_index=False)["count"].sum()

            embeddings = run_optional_step("Embeddings für Visualisierung", embedding_cache.encode, list(df["word"]))
            if embeddings is not None:
                run_optional_step("Word Cluster Plot speichern", visualize_word_embeddings, list(df["word"]), embeddings, config.WORD_CLUSTERS_HTML)
                run_optional_step("Word Cluster Plot (plots/) speichern", visualize_word_embeddings, list(df["word"]), embeddings, config.WORD_CLUSTERS_PLOT_HTML)

            df = df.sort_values(by=["count", "word"], ascending=[False, True]).reset_index(drop=True)

        df.to_csv(config.FINAL_CSV_FILE, index=False, encoding="utf-8")
        df.to_csv(config.CSV_DIR / "word_frequency.csv", index=False, encoding="utf-8")

        run_optional_step("Topics extrahieren", extract_topics, video_texts, config.CSV_CLEANUP_MODEL, config.TOPICS_CSV_FILE)
        run_optional_step("Video Similarity berechnen", compute_video_similarity, video_texts, embedding_cache, config.VIDEO_SIMILARITY_CSV_FILE)
        run_optional_step("Speaker Style exportieren", compare_speakers, {k: dict(v) for k, v in global_speaker_counts.items()}, config.SPEAKER_STYLE_CSV_FILE)
        run_optional_step("Word Timeline speichern", word_frequency_over_time, timeline_words, config.WORD_TIMELINE_HTML)
        run_optional_step("Word Timeline (plots/) speichern", word_frequency_over_time, timeline_words, config.WORD_TIMELINE_PLOT_HTML)
        run_optional_step(
            "Runtime-Schätzung exportieren",
            runtime_estimator.export,
            config.RUNTIME_ESTIMATION_CSV,
            config.RUNTIME_ESTIMATION_HTML,
        )

        dt = time.time() - t0
        log_ok(f"CSV/NLP-Ausgaben geschrieben in {fmt_seconds(dt)}")
        console.set_progress(runtime_estimator.snapshot(total_links, time.time() - total_start))

        total_dt = time.time() - total_start
        log_ok(f"Gesamt fertig in {fmt_seconds(total_dt)}")
        log_info(f"Unique Wörter gesamt: {len(total_counter)}")
    finally:
        finish_console()


if __name__ == "__main__":
    main()
