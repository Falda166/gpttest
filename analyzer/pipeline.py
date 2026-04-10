import time
from pathlib import Path
import shutil

import whisperx

from analyzer import config
from analyzer.audio_processing import (
    load_audio_tensor,
    download_audio,
    remove_silence_from_audio,
    extract_time_regions_to_audio,
)
from analyzer.logging_utils import timed_step, log_info, log_warn, log_ok, log_step, fmt_seconds
from analyzer.helpers import sanitize_filename
from analyzer.speaker_processing import (
    build_overlap_regions_from_diarization,
    collect_speaker_embeddings,
    choose_best_matching_speaker,
)
from analyzer.text_processing import clean_word, normalize_word_for_counting, classify_score


def process_single_video(
    url: str,
    raw_audio_file: Path,
    cleaned_audio_file: Path,
    papaplatte_emb,
    whisperx_model,
    align_model,
    align_metadata,
    diar,
    embedder,
    device_torch,
    device_str,
    return_metadata: bool = False,
):
    video_start = time.time()
    log_info(f"Verarbeite: {url}")

    timed_step("Audio herunterladen", download_audio, url, raw_audio_file, config.AUDIO_OVERWRITE)

    cleaning_stats = timed_step(
        "Audio bereinigen (Stille entfernen)",
        remove_silence_from_audio,
        raw_audio_file,
        cleaned_audio_file,
        config.SILENCE_THRESHOLD,
        config.MIN_SILENCE_SECONDS,
        config.KEEP_SILENCE_SECONDS,
        config.MIN_CHUNK_SECONDS,
    )
    log_ok(
        "Audio bereinigt: "
        f"{cleaning_stats['original_seconds']:.1f}s -> {cleaning_stats['cleaned_seconds']:.1f}s "
        f"(entfernt: {cleaning_stats['removed_seconds']:.1f}s, Bereiche: {cleaning_stats['regions']})"
    )

    audio_for_processing = cleaned_audio_file
    if cleaning_stats["cleaned_seconds"] < config.MIN_CLEANED_AUDIO_SECONDS:
        log_warn(
            f"Bereinigte Datei ist sehr kurz ({cleaning_stats['cleaned_seconds']:.1f}s). "
            "Nutze stattdessen Roh-Audio für stabilere Diarization."
        )
        audio_for_processing = raw_audio_file

    waveform, sr = timed_step("Audio-Tensor laden", load_audio_tensor, audio_for_processing)

    audio_dict = {"waveform": waveform.to(device_torch), "sample_rate": sr}

    diar_output = timed_step("Speaker Diarization", diar, audio_dict)
    speaker_diarization = diar_output.speaker_diarization
    annotation = diar_output.exclusive_speaker_diarization

    overlap_regions = timed_step(
        "Overlap-Regionen berechnen",
        build_overlap_regions_from_diarization,
        speaker_diarization,
    )

    speaker_embs = timed_step(
        "Speaker-Embeddings sammeln",
        collect_speaker_embeddings,
        annotation,
        waveform,
        sr,
        embedder,
        papaplatte_emb.shape,
        config.MIN_SEGMENT_SECONDS,
        device_torch,
    )

    best_speaker, best_score = timed_step(
        "Besten Speaker bestimmen",
        choose_best_matching_speaker,
        speaker_embs,
        papaplatte_emb,
    )

    if best_speaker is None:
        log_warn("Kein passender Speaker gefunden, Video wird übersprungen.")
        return {
            "words": [],
            "transcript": "",
            "timed_words": [],
            "speaker_word_counts": {},
            "source_audio_seconds": cleaning_stats["original_seconds"],
            "target_audio_seconds": 0.0,
            "processing_seconds": time.time() - video_start,
        } if return_metadata else []

    if best_score < config.MIN_ACCEPT_SCORE:
        log_warn(f"Score zu niedrig ({best_score:.3f}) -> Video übersprungen")
        return {
            "words": [],
            "transcript": "",
            "timed_words": [],
            "speaker_word_counts": {},
            "source_audio_seconds": cleaning_stats["original_seconds"],
            "target_audio_seconds": 0.0,
            "processing_seconds": time.time() - video_start,
        } if return_metadata else []

    label = classify_score(best_score, config.MATCH_THRESHOLD_STRONG, config.MATCH_THRESHOLD_MAYBE)
    log_ok(f"Match: {best_speaker} ({best_score:.3f}, {label})")

    def _subtract_overlaps(interval, overlaps):
        start, end = interval
        remaining = [(start, end)]
        for os_, oe_ in overlaps:
            next_remaining = []
            for rs, re in remaining:
                if oe_ <= rs or os_ >= re:
                    next_remaining.append((rs, re))
                    continue
                if os_ > rs:
                    next_remaining.append((rs, os_))
                if oe_ < re:
                    next_remaining.append((oe_, re))
            remaining = next_remaining
            if not remaining:
                break
        return [(s, e) for s, e in remaining if e > s]

    target_regions = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        if speaker != best_speaker:
            continue
        base_region = (float(segment.start), float(segment.end))
        target_regions.extend(_subtract_overlaps(base_region, overlap_regions))

    target_regions.sort(key=lambda x: x[0])
    merged_regions = []
    for start, end in target_regions:
        if not merged_regions or start > merged_regions[-1][1]:
            merged_regions.append([start, end])
        else:
            merged_regions[-1][1] = max(merged_regions[-1][1], end)
    target_regions = [(s, e) for s, e in merged_regions]

    target_audio_file = cleaned_audio_file.parent / f"{cleaned_audio_file.stem}_target.wav"
    target_audio_stats = timed_step(
        "Target-Speaker Audio schneiden",
        extract_time_regions_to_audio,
        audio_for_processing,
        target_audio_file,
        target_regions,
        config.MIN_SEGMENT_SECONDS,
    )

    if target_audio_stats["segments"] == 0 or target_audio_stats["output_seconds"] < config.MIN_SEGMENT_SECONDS:
        log_warn("Zu wenig Target-Speaker Audio gefunden, Video wird übersprungen.")
        return {
            "words": [],
            "transcript": "",
            "timed_words": [],
            "speaker_word_counts": {},
            "source_audio_seconds": cleaning_stats["original_seconds"],
            "target_audio_seconds": target_audio_stats["output_seconds"],
            "processing_seconds": time.time() - video_start,
        } if return_metadata else []

    log_ok(
        "Target-Audio erstellt: "
        f"{target_audio_stats['segments']} Segmente, "
        f"{target_audio_stats['output_seconds']:.1f}s von {target_audio_stats['source_seconds']:.1f}s"
    )

    if config.SAVE_PAPAPLATTE_TRAINING_AUDIO:
        training_name = sanitize_filename(f"{cleaned_audio_file.stem}_{best_score:.3f}_papaplatte.wav")
        training_audio_file = config.PAPAPLATTE_TRAINING_DIR / training_name
        shutil.copyfile(target_audio_file, training_audio_file)
        log_ok(f"Trainings-Audio gespeichert: {training_audio_file}")

    target_audio_for_whisperx = timed_step(
        "Target-Audio für WhisperX laden",
        whisperx.load_audio,
        str(target_audio_file),
    )

    asr_result = timed_step(
        "WhisperX Transkription (Target)",
        whisperx_model.transcribe,
        target_audio_for_whisperx,
        batch_size=16,
        language=config.WHISPER_LANGUAGE,
    )

    aligned_result = timed_step(
        "WhisperX Alignment (Target)",
        whisperx.align,
        asr_result["segments"],
        align_model,
        align_metadata,
        target_audio_for_whisperx,
        device_str,
    )

    def _target_to_source_time(t, mapping):
        ts = float(t)
        for m in mapping:
            if m["target_start"] <= ts <= m["target_end"]:
                return m["source_start"] + (ts - m["target_start"])
        return None

    accepted_words = []

    log_step("Wörter filtern und zählen vorbereiten")
    t0 = time.time()

    for seg in aligned_result.get("segments", []):
        for word in seg.get("words", []):
            if "start" not in word or "end" not in word:
                continue
            txt = clean_word(word.get("word", ""))
            txt = normalize_word_for_counting(txt)
            if txt:
                accepted_words.append(txt)

    dt = time.time() - t0
    log_ok(f"Wörter gefiltert in {fmt_seconds(dt)} | akzeptiert: {len(accepted_words)}")

    transcript_words = []
    speaker_word_counts = {}
    for seg in aligned_result.get("segments", []):
        seg_speaker = best_speaker
        for word in seg.get("words", []):
            txt = normalize_word_for_counting(clean_word(word.get("word", "")))
            if not txt:
                continue
            mapped_start = _target_to_source_time(word.get("start", 0.0), target_audio_stats["mapping"])
            if mapped_start is None:
                continue
            transcript_words.append({"start": mapped_start, "word": txt, "speaker": seg_speaker})
            speaker_word_counts.setdefault(seg_speaker, {})
            speaker_word_counts[seg_speaker][txt] = speaker_word_counts[seg_speaker].get(txt, 0) + 1

    total_video_time = time.time() - video_start
    log_ok(f"Video komplett fertig in {fmt_seconds(total_video_time)}")

    if return_metadata:
        transcript_text = " ".join([w["word"] for w in transcript_words])
        return {
            "words": accepted_words,
            "transcript": transcript_text,
            "timed_words": transcript_words,
            "speaker_word_counts": speaker_word_counts,
            "source_audio_seconds": cleaning_stats["original_seconds"],
            "target_audio_seconds": target_audio_stats["output_seconds"],
            "processing_seconds": total_video_time,
        }

    return accepted_words
