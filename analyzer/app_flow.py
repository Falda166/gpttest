from collections import Counter
from typing import Callable


def load_links_from_channel(config, extract_video_links_func) -> list[str]:
    if not config.YOUTUBE_CHANNEL_URL:
        raise RuntimeError(
            "YOUTUBE_CHANNEL_URL fehlt. Bitte in .env setzen, z.B. "
            "YOUTUBE_CHANNEL_URL=https://www.youtube.com/@papaplatte/videos"
        )

    max_links = None if config.YOUTUBE_FETCH_ALL else config.YOUTUBE_MAX_LINKS
    proxy = "" if config.YOUTUBE_NO_PROXY else config.YOUTUBE_PROXY
    links = extract_video_links_func(config.YOUTUBE_CHANNEL_URL, max_links=max_links, proxy=proxy)

    if not links:
        raise RuntimeError("Keine Video-Links vom Channel gefunden.")

    return links


def process_video_batch(
    links: list[str],
    video_processor: Callable[[int, str], dict],
    log_warn: Callable[[str], None],
    log_error: Callable[[str], None],
):
    total_counter = Counter()
    video_texts = {}
    timeline_words = []
    global_speaker_counts = {}
    interrupted = False

    for idx, url in enumerate(links, start=1):
        try:
            result = video_processor(idx, url)
        except KeyboardInterrupt:
            interrupted = True
            log_warn("Strg+C erkannt: Verarbeitung wird sauber beendet und Teilergebnisse werden gespeichert.")
            break
        except Exception as e:
            log_error(f"Fehler bei Link {idx}: {e}")
            continue

        words = result["words"]
        total_counter.update(words)
        video_texts[f"video_{idx}"] = result["transcript"]
        timeline_words.extend(result["timed_words"])

        for speaker, counts in result["speaker_word_counts"].items():
            if speaker not in global_speaker_counts:
                global_speaker_counts[speaker] = Counter()
            global_speaker_counts[speaker].update(counts)

    return total_counter, video_texts, timeline_words, global_speaker_counts, interrupted
