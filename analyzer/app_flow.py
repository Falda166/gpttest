from collections import Counter
from typing import Callable


def resolve_channel_url(config, prompt_func: Callable[[str], str] = input) -> str:
    if config.YOUTUBE_CHANNEL_URL:
        return config.YOUTUBE_CHANNEL_URL

    entered = (prompt_func("Bitte YouTube-Channel-Link eingeben: ") or "").strip()
    if not entered:
        raise RuntimeError("Kein Channel-Link angegeben.")
    return entered


def load_channel_items(config, extract_channel_items_func, prompt_func: Callable[[str], str] = input) -> list[dict]:
    channel_url = resolve_channel_url(config, prompt_func=prompt_func)
    max_links = None if config.YOUTUBE_FETCH_ALL else config.YOUTUBE_MAX_LINKS
    proxy = "" if config.YOUTUBE_NO_PROXY else config.YOUTUBE_PROXY
    items = extract_channel_items_func(channel_url, max_links=max_links, proxy=proxy)

    if not items:
        raise RuntimeError("Keine Video-Links vom Channel gefunden.")

    return items


def filter_new_items(items: list[dict], analyzed_urls: set[str]) -> list[dict]:
    return [item for item in items if item.get("url") not in analyzed_urls]


def process_video_batch(
    items: list[dict],
    video_processor: Callable[[int, dict], dict],
    log_warn: Callable[[str], None],
    log_error: Callable[[str], None],
    on_success: Callable[[int, dict, dict], None] | None = None,
):
    total_counter = Counter()
    video_texts = {}
    timeline_words = []
    global_speaker_counts = {}
    interrupted = False

    for idx, item in enumerate(items, start=1):
        try:
            result = video_processor(idx, item)
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
        if on_success is not None:
            on_success(idx, item, result)

    return total_counter, video_texts, timeline_words, global_speaker_counts, interrupted
