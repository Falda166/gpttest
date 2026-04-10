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
from analyzer.app_flow import load_channel_items, process_video_batch, filter_new_items
from analyzer.csv_cleanup import CsvCleaner
from analyzer.embedding_cache import EmbeddingCache
from analyzer.helpers import extract_embedding
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
from analyzer.speaker_processing import collect_speaker_embeddings, normalize_embedding
from analyzer.speaker_style import compare_speakers
from analyzer.time_analysis import word_frequency_over_time
from analyzer.topic_detection import extract_topics
from analyzer.video_similarity import compute_video_similarity
from analyzer.visualization import visualize_word_embeddings
from analyzer.word_clustering import normalize_words
from extract_channel_links import extract_channel_video_items

load_dotenv()


def run_optional_step(label: str, func, *args, **kwargs):
    try:
        return timed_step(label, func, *args, **kwargs)
    except Exception as e:
        log_warn(f"{label} übersprungen: {e}")
        return None


def load_analyzed_urls(path):
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    return set(data if isinstance(data, list) else [])


def save_analyzed_urls(path, urls: set[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(urls), f, ensure_ascii=False, indent=2)


def auto_profile_target_embedding(profile_items, diar, embedder, device_torch):
    from analyzer.audio_processing import download_audio, load_audio_tensor
    import numpy as np

    speaker_embs_all = []

    for i, item in enumerate(profile_items, start=1):
        url = item["url"]
        raw_audio_file = config.AUDIO_DIR / f"profile_{i}.wav"
        timed_step("Profiling-Audio herunterladen", download_audio, url, raw_audio_file, True)
        waveform, sr = timed_step("Profiling-Audio laden", load_audio_tensor, raw_audio_file)
        diar_output = timed_step("Profiling-Diarization", diar, {"waveform": waveform.to(device_torch), "sample_rate": sr})
        annotation = diar_output.exclusive_speaker_diarization

        durations = {}
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            durations[speaker] = durations.get(speaker, 0.0) + max(0.0, float(segment.end) - float(segment.start))
        dominant_speaker = max(durations, key=durations.get) if durations else None
        if dominant_speaker is None:
            cleanup_audio_files(raw_audio_file)
            continue

        all_embs = collect_speaker_embeddings(
            annotation,
            waveform,
            sr,
            embedder,
            ref_shape=None,
            min_seconds=config.MIN_SEGMENT_SECONDS,
            device=device_torch,
        )
        speaker_embs_all.extend(all_embs.get(dominant_speaker, []))
        cleanup_audio_files(raw_audio_file)

    if not speaker_embs_all:
        raise RuntimeError("Konnte aus Profiling-Videos kein gültiges Referenz-Embedding erzeugen.")

    return normalize_embedding(np.mean(np.stack(speaker_embs_all, axis=0), axis=0))


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

        items = timed_step("Links aus YouTube Channel laden", load_channel_items, config, extract_channel_video_items)
        analyzed_urls = load_analyzed_urls(config.ANALYZED_VIDEOS_FILE)
        items = filter_new_items(items, analyzed_urls)
        if not items:
            log_ok("Keine neuen Videos zum Analysieren gefunden (alle bereits in Blacklist).")
            return

        log_info(f"{len(items)} neue Videos geladen")

        durations = []
        log_step("Video-Längen laden (für ETA)")
        for i, item in enumerate(items, start=1):
            try:
                d = fetch_video_duration_seconds(item["url"])
                durations.append(d)
            except Exception:
                durations.append(None)
                log_warn(f"Länge für Video {i} konnte nicht geladen werden")

        kept_pairs = []
        skipped_long = 0
        for item, d in zip(items, durations):
            if d is not None and d > config.MAX_VIDEO_DURATION_SECONDS:
                skipped_long += 1
                log_warn(
                    f"Video verworfen (> {config.MAX_VIDEO_DURATION_SECONDS/60:.0f} min): "
                    f"{item.get('url')} ({d/60:.1f} min)"
                )
                continue
            kept_pairs.append((item, d))

        if skipped_long:
            log_info(f"{skipped_long} zu lange Videos wurden übersprungen.")

        items = [it for it, _ in kept_pairs]
        durations = [d for _, d in kept_pairs]
        if not items:
            log_ok("Keine verarbeitbaren Videos übrig (alle zu lang oder bereits verarbeitet).")
            return

        total_links = len(items)
        log_info(f"{total_links} Videos nach Längenfilter übrig")
        console.start_run(total_videos=total_links)

        runtime_estimator = RuntimeEstimator(total_videos=total_links, planned_durations_seconds=durations)
        console.set_progress(runtime_estimator.snapshot(0, time.time() - total_start))

        with open(config.DB_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)

        diarization_token = config.HF_TOKEN
        if config.DIARIZATION_MODEL.startswith("pyannote/speaker-diarization-precision"):
            diarization_token = config.PYANNOTEAI_API_KEY
            if not diarization_token:
                raise RuntimeError(
                    "Für pyannote/speaker-diarization-precision-* wird PYANNOTEAI_API_KEY benötigt. "
                    "Setze die Umgebungsvariable oder nutze z. B. pyannote/speaker-diarization-community-1."
                )

        diar = timed_step("pyannote Diarization Pipeline laden", Pipeline.from_pretrained, config.DIARIZATION_MODEL, token=diarization_token)
        diar.to(device_torch)

        embedder = timed_step("SpeakerEmbedding laden", SpeakerEmbedding, token=config.HF_TOKEN)
        if hasattr(embedder, "to"):
            embedder.to(device_torch)

        target_key = config.TARGET_SPEAKER_KEY
        if target_key not in db and "papaplatte" in db:
            db[target_key] = db["papaplatte"]
            log_warn(f"Migriere legacy voice_db Eintrag 'papaplatte' -> '{target_key}'.")

        if target_key not in db:
            profile_items = items[:max(2, config.SPEAKER_PROFILE_VIDEOS)]
            target_emb = timed_step("Auto-Profiling Hauptsprecher", auto_profile_target_embedding, profile_items, diar, embedder, device_torch)
            db[target_key] = {"embedding": target_emb.tolist(), "source": "auto-profile"}
            with open(config.DB_FILE, "w", encoding="utf-8") as f:
                json.dump(db, f, ensure_ascii=False, indent=2)
            log_ok(f"{target_key} automatisch in voice_db.json gespeichert.")
        else:
            target_emb = timed_step("Referenz-Embedding laden", extract_embedding, db[target_key], target_key)

        whisperx_model = timed_step("WhisperX Modell laden", whisperx.load_model, config.WHISPERX_MODEL, device_str, compute_type=compute_type, language=config.WHISPER_LANGUAGE)
        align_model, align_metadata = timed_step("Alignment-Modell laden", whisperx.load_align_model, language_code=config.WHISPER_LANGUAGE, device=device_str)

        embedding_cache = timed_step("Embedding Cache initialisieren", EmbeddingCache, config.CSV_CLEANUP_MODEL, config.EMBEDDINGS_CACHE_FILE)
        csv_cleaner = timed_step("CSV-Cleaner Modell laden", CsvCleaner, config.CSV_CLEANUP_MODEL, config.CSV_SEMANTIC_THRESHOLD, embedding_cache)

        def _process_one(idx: int, item: dict):
            loop_start = time.time()
            url = item["url"]
            console.set_video(idx, total_links, url=url, expected_processing_seconds=runtime_estimator.estimate_processing_seconds_for_video(durations[idx - 1]))
            log_info(f"========== VIDEO {idx}/{total_links} ==========")
            log_info(f"title='{item.get('title')}', upload_date={item.get('upload_date')}, duration={item.get('duration')}, channel={item.get('channel')}")
            log_info(f"URL: {url}")

            raw_audio_file = config.AUDIO_DIR / f"audio_{idx}.wav"
            cleaned_audio_file = config.CLEAN_AUDIO_DIR / f"audio_{idx}_cleaned.wav"
            target_audio_file = config.CLEAN_AUDIO_DIR / f"audio_{idx}_cleaned_target.wav"
            result = None

            try:
                result = process_single_video(
                    url=url,
                    raw_audio_file=raw_audio_file,
                    cleaned_audio_file=cleaned_audio_file,
                    target_speaker_emb=target_emb,
                    whisperx_model=whisperx_model,
                    align_model=align_model,
                    align_metadata=align_metadata,
                    diar=diar,
                    embedder=embedder,
                    device_torch=device_torch,
                    device_str=device_str,
                    target_label=config.TARGET_SPEAKER_LABEL,
                    return_metadata=True,
                )
                return result
            finally:
                cleanup_audio_files(raw_audio_file, cleaned_audio_file, target_audio_file)
                loop_dt = time.time() - loop_start
                video_seconds = durations[idx - 1]
                if video_seconds is None:
                    video_seconds = float(result.get("source_audio_seconds", 0.0)) if isinstance(result, dict) else 0.0
                runtime_estimator.update(video_idx=idx, video_seconds=video_seconds, processing_seconds=loop_dt)
                total_elapsed = time.time() - total_start
                console.set_progress(runtime_estimator.snapshot(idx, total_elapsed))

        def _on_success(_idx: int, item: dict, _result: dict):
            analyzed_urls.add(item["url"])
            save_analyzed_urls(config.ANALYZED_VIDEOS_FILE, analyzed_urls)

        total_counter, video_texts, timeline_words, global_speaker_counts, interrupted = process_video_batch(
            items,
            _process_one,
            log_warn,
            log_error,
            on_success=_on_success,
        )

        if interrupted:
            log_warn("Abbruch mit Teilergebnissen: CSV/NLP-Ausgaben werden jetzt trotzdem geschrieben.")

        df = pd.DataFrame([{"word": word, "count": count} for word, count in total_counter.items()])
        if not df.empty:
            df = timed_step("CSV Basis-Bereinigung", csv_cleaner.basic_cleanup, df)
            df = timed_step("CSV Semantik-Bereinigung", csv_cleaner.semantic_cleanup, df)
            normalized_words = timed_step("Wort-Clustering + Normalisierung", normalize_words, list(df["word"]), embedding_cache, config.WORD_CLUSTER_MIN_SIZE)
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
        run_optional_step("Runtime-Schätzung exportieren", runtime_estimator.export, config.RUNTIME_ESTIMATION_CSV, config.RUNTIME_ESTIMATION_HTML)

        log_ok(f"CSV/NLP-Ausgaben geschrieben in {fmt_seconds(time.time() - total_start)}")
    finally:
        finish_console()


if __name__ == "__main__":
    main()
