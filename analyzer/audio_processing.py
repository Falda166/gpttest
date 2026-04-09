from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yt_dlp

from analyzer.helpers import sanitize_filename


def load_audio_tensor(path: Path):
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


def _rms_envelope(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if signal.size < frame_size:
        pad = np.zeros(frame_size - signal.size, dtype=np.float32)
        signal = np.concatenate([signal, pad])

    rms_values = []
    for start in range(0, signal.size - frame_size + 1, hop_size):
        frame = signal[start:start + frame_size]
        rms_values.append(np.sqrt(np.mean(np.square(frame)) + 1e-12))

    return np.asarray(rms_values, dtype=np.float32)


def _speech_regions_from_rms(
    rms: np.ndarray,
    sr: int,
    hop_size: int,
    silence_threshold: float,
    min_silence_seconds: float,
    keep_silence_seconds: float,
    min_chunk_seconds: float,
):
    voiced = rms > silence_threshold
    if voiced.size == 0 or not voiced.any():
        return []

    min_silence_frames = max(1, int(min_silence_seconds * sr / hop_size))
    keep_frames = max(0, int(keep_silence_seconds * sr / hop_size))
    min_chunk_frames = max(1, int(min_chunk_seconds * sr / hop_size))

    regions = []
    start = None
    silence_run = 0

    for i, is_voiced in enumerate(voiced):
        if is_voiced:
            if start is None:
                start = i
            silence_run = 0
            continue

        if start is not None:
            silence_run += 1
            if silence_run >= min_silence_frames:
                end = i - silence_run + 1
                if end - start >= min_chunk_frames:
                    regions.append((max(0, start - keep_frames), i + keep_frames))
                start = None
                silence_run = 0

    if start is not None:
        end = len(voiced)
        if end - start >= min_chunk_frames:
            regions.append((max(0, start - keep_frames), end))

    merged = []
    for s, e in sorted(regions):
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return [(s, e) for s, e in merged]


def remove_silence_from_audio(
    input_wav: Path,
    output_wav: Path,
    silence_threshold: float,
    min_silence_seconds: float,
    keep_silence_seconds: float,
    min_chunk_seconds: float,
):
    audio, sr = sf.read(input_wav, always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1)

    frame_size = max(256, int(sr * 0.02))
    hop_size = max(128, int(sr * 0.01))

    rms = _rms_envelope(mono, frame_size=frame_size, hop_size=hop_size)
    speech_regions = _speech_regions_from_rms(
        rms=rms,
        sr=sr,
        hop_size=hop_size,
        silence_threshold=silence_threshold,
        min_silence_seconds=min_silence_seconds,
        keep_silence_seconds=keep_silence_seconds,
        min_chunk_seconds=min_chunk_seconds,
    )

    if not speech_regions:
        sf.write(output_wav, audio, sr)
        return {
            "original_seconds": len(audio) / sr,
            "cleaned_seconds": len(audio) / sr,
            "removed_seconds": 0.0,
            "regions": 0,
        }

    cleaned_parts = []
    for start_frame, end_frame in speech_regions:
        s = min(len(audio), max(0, start_frame * hop_size))
        e = min(len(audio), max(s, end_frame * hop_size))
        cleaned_parts.append(audio[s:e])

    cleaned_audio = np.concatenate(cleaned_parts, axis=0) if cleaned_parts else audio
    sf.write(output_wav, cleaned_audio, sr)

    original_seconds = len(audio) / sr
    cleaned_seconds = len(cleaned_audio) / sr
    return {
        "original_seconds": original_seconds,
        "cleaned_seconds": cleaned_seconds,
        "removed_seconds": max(0.0, original_seconds - cleaned_seconds),
        "regions": len(speech_regions),
    }


def extract_time_regions_to_audio(
    input_wav: Path,
    output_wav: Path,
    regions,
    min_region_seconds: float = 0.0,
):
    audio, sr = sf.read(input_wav, always_2d=True, dtype="float32")
    total_seconds = len(audio) / sr if sr > 0 else 0.0

    normalized_regions = []
    for start, end in regions:
        start = max(0.0, float(start))
        end = min(total_seconds, float(end))
        if end <= start:
            continue
        if (end - start) < float(min_region_seconds):
            continue
        normalized_regions.append((start, end))

    if not normalized_regions:
        sf.write(output_wav, np.zeros((0, audio.shape[1]), dtype=np.float32), sr)
        return {
            "segments": 0,
            "source_seconds": total_seconds,
            "output_seconds": 0.0,
            "mapping": [],
        }

    pieces = []
    mapping = []
    out_cursor = 0.0

    for start, end in normalized_regions:
        s = int(start * sr)
        e = int(end * sr)
        chunk = audio[s:e]
        if chunk.size == 0:
            continue
        pieces.append(chunk)
        chunk_seconds = len(chunk) / sr
        mapping.append({
            "source_start": start,
            "source_end": end,
            "target_start": out_cursor,
            "target_end": out_cursor + chunk_seconds,
        })
        out_cursor += chunk_seconds

    if pieces:
        out_audio = np.concatenate(pieces, axis=0)
    else:
        out_audio = np.zeros((0, audio.shape[1]), dtype=np.float32)

    sf.write(output_wav, out_audio, sr)

    return {
        "segments": len(mapping),
        "source_seconds": total_seconds,
        "output_seconds": out_cursor,
        "mapping": mapping,
    }


def cleanup_audio_files(*files: Path):
    for file_path in files:
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
