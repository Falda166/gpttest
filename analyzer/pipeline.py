import time
from pathlib import Path

import whisperx

from analyzer import config
from analyzer.audio_processing import load_audio_tensor, download_audio, remove_silence_from_audio
from analyzer.logging_utils import timed_step, log_info, log_warn, log_ok, log_step, fmt_seconds
from analyzer.speaker_processing import (
    build_diarization_df,
    build_overlap_regions_from_diarization,
    collect_speaker_embeddings,
    choose_best_matching_speaker,
    is_overlapped,
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

    audio_for_whisperx = timed_step(
        "Audio für WhisperX laden",
        whisperx.load_audio,
        str(audio_for_processing),
    )

    asr_result = timed_step(
        "WhisperX Transkription",
        whisperx_model.transcribe,
        audio_for_whisperx,
        batch_size=16,
        language=config.WHISPER_LANGUAGE,
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

    waveform, sr = timed_step("Audio-Tensor laden", load_audio_tensor, audio_for_processing)

    audio_dict = {"waveform": waveform.to(device_torch), "sample_rate": sr}

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
        return {"words": [], "transcript": "", "timed_words": [], "speaker_word_counts": {}} if return_metadata else []

    if best_score < config.MIN_ACCEPT_SCORE:
        log_warn(f"Score zu niedrig ({best_score:.3f}) -> Video übersprungen")
        return {"words": [], "transcript": "", "timed_words": [], "speaker_word_counts": {}} if return_metadata else []

    label = classify_score(best_score, config.MATCH_THRESHOLD_STRONG, config.MATCH_THRESHOLD_MAYBE)
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

            if txt:
                accepted_words.append(txt)

    dt = time.time() - t0
    log_ok(f"Wörter gefiltert in {fmt_seconds(dt)} | akzeptiert: {len(accepted_words)}")

    transcript_words = []
    speaker_word_counts = {}
    for seg in result_with_speakers["segments"]:
        seg_speaker = seg.get("speaker", "unknown")
        for word in seg.get("words", []):
            txt = normalize_word_for_counting(clean_word(word.get("word", "")))
            if not txt:
                continue
            transcript_words.append({"start": float(word.get("start", 0.0)), "word": txt, "speaker": seg_speaker})
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
        }

    return accepted_words
