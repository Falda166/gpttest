import json
import os
import shutil
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
from analyzer.logging_utils import timed_step, log_info, log_ok, log_error, log_warn, fmt_seconds, log_step, draw_bottom_panel
from analyzer.speaker_processing import collect_speaker_embeddings, normalize_embedding
from analyzer.pipeline import process_single_video
from analyzer.progress_tracking import RuntimeEstimator
from analyzer.runtime import configure_runtime, resolve_device
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


def auto_profile_papaplatte_embedding(profile_items, diar, embedder, device_torch):
    from analyzer.audio_processing import download_audio, load_audio_tensor

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
        if not durations:
            cleanup_audio_files(raw_audio_file)
            continue

        dominant_speaker = max(durations, key=durations.get)
        all_embs = collect_speaker_embeddings(
            annotation,
            waveform,
            sr,
            embedder,
            ref_shape=None,
            min_seconds=config.MIN_SEGMENT_SECONDS,
            device=device_torch,
        )
        emb_list = all_embs.get(dominant_speaker, [])
        speaker_embs_all.extend(emb_list)
        cleanup_audio_files(raw_audio_file)

    if not speaker_embs_all:
        raise RuntimeError("Konnte aus Profiling-Videos kein gültiges Embedding erzeugen.")

    import numpy as np
    return normalize_embedding(np.mean(np.stack(speaker_embs_all, axis=0), axis=0))


def main():
    total_start = time.time()

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
    known_durations = sum(1 for d in durations if d is not None)
    log_ok(f"Video-Längen geladen: {known_durations}/{len(items)} bekannt")
    runtime_estimator = RuntimeEstimator(total_videos=len(items), planned_durations_seconds=durations)
    panel_width = max(40, shutil.get_terminal_size((24, 120)).columns - 8)
    draw_bottom_panel(runtime_estimator.render_progress_panel(0, 0.0, bar_width=panel_width))

    log_step("voice_db.json laden")
    t0 = time.time()
    with open(config.DB_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)
    dt = time.time() - t0
    log_ok(f"voice_db.json geladen in {fmt_seconds(dt)}")

    diar = timed_step(
        "pyannote Diarization Pipeline laden",
        Pipeline.from_pretrained,
        config.DIARIZATION_MODEL,
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

    if "papaplatte" not in db:
        log_warn("papaplatte nicht in voice_db.json gefunden - starte Auto-Profiling mit 2 Videos.")
        profile_items = items[:max(1, config.SPEAKER_PROFILE_VIDEOS)]
        papaplatte_emb = timed_step(
            "Auto-Profiling Hauptsprecher",
            auto_profile_papaplatte_embedding,
            profile_items,
            diar,
            embedder,
            device_torch,
        )
        db["papaplatte"] = {"embedding": papaplatte_emb.tolist(), "source": "auto-profile"}
        with open(config.DB_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        log_ok("papaplatte automatisch in voice_db.json gespeichert.")
    else:
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

    total_links = len(items)
    def _process_one(idx: int, item: dict):
        loop_start = time.time()
        print()
        meta = f"title='{item.get('title')}', upload_date={item.get('upload_date')}, duration={item.get('duration')}, channel={item.get('channel')}"
        log_info(f"========== VIDEO {idx}/{total_links} ==========")
        log_info(meta)
        url = item["url"]
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
            return result
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
            draw_bottom_panel(runtime_estimator.render_progress_panel(idx, total_elapsed, bar_width=panel_width))
            log_info(runtime_estimator.render_progress_line(idx, total_elapsed))
            log_info(f"Laufzeitformel aktuell: {runtime_estimator.formula_text()}")
            log_info(f"Video {idx}/{total_links} abgeschlossen in {fmt_seconds(loop_dt)}")

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
    log_ok(f"Aktueller Wortschatz: {len(total_counter)} unique Wörter")
    if interrupted:
        log_warn("Abbruch mit Teilergebnissen: CSV/NLP-Ausgaben werden jetzt trotzdem geschrieben.")

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

    total_dt = time.time() - total_start
    draw_bottom_panel(runtime_estimator.render_progress_panel(total_links, total_dt, bar_width=panel_width))
    print()
    log_ok(f"Gesamt fertig in {fmt_seconds(total_dt)}")
    log_info(f"Unique Wörter gesamt: {len(total_counter)}")


if __name__ == "__main__":
    main()
