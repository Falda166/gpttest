import argparse
import re
from pathlib import Path

import yt_dlp


def _normalize_channel_url(channel_url: str) -> str:
    return channel_url.rstrip("/")


def _candidate_urls(channel_url: str) -> list[str]:
    base = _normalize_channel_url(channel_url)
    candidates = [base]

    if not re.search(r"/(videos|streams|playlists)$", base):
        candidates.append(f"{base}/videos")

    return candidates


def _extract_info(url: str, *, flat: bool) -> dict:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist" if flat else False,
        "skip_download": True,
        "lazy_playlist": False,
        "playlistend": 1_000_000,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


def _info_to_video_urls(info: dict) -> list[str]:
    entries = info.get("entries") or []
    urls: list[str] = []
    seen = set()

    for entry in entries:
        if not entry:
            continue

        video_id = entry.get("id")
        url = entry.get("url")
        webpage_url = entry.get("webpage_url")

        if video_id:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
        elif webpage_url:
            video_url = webpage_url
        elif url and "watch?v=" in url:
            video_url = url
        else:
            continue

        if video_url not in seen:
            seen.add(video_url)
            urls.append(video_url)

    return urls


def _uploads_playlist_from_channel_id(channel_id: str) -> str | None:
    if channel_id and channel_id.startswith("UC") and len(channel_id) > 2:
        return f"https://www.youtube.com/playlist?list=UU{channel_id[2:]}"
    return None


def _resolve_channel_id(channel_url: str) -> str | None:
    """
    Resolve a stable UC... channel id from various YouTube channel URL formats.
    """
    for candidate in _candidate_urls(channel_url):
        try:
            info = _extract_info(candidate, flat=False)
        except Exception:
            continue

        channel_id = info.get("channel_id") or info.get("id")
        if isinstance(channel_id, str) and channel_id.startswith("UC"):
            return channel_id

    return None


def extract_video_links(channel_url: str) -> list[str]:
    """Extract all uploaded video URLs from a YouTube channel URL."""
    best_links: list[str] = []

    for candidate in _candidate_urls(channel_url):
        try:
            info = _extract_info(candidate, flat=True)
            links = _info_to_video_urls(info)
            if len(links) > len(best_links):
                best_links = links
        except Exception:
            pass

    channel_id = _resolve_channel_id(channel_url)
    if channel_id:
        fallback_urls = [
            f"https://www.youtube.com/channel/{channel_id}/videos",
            _uploads_playlist_from_channel_id(channel_id),
        ]
        for fallback_url in fallback_urls:
            if not fallback_url:
                continue
            try:
                info = _extract_info(fallback_url, flat=True)
                links = _info_to_video_urls(info)
                if len(links) > len(best_links):
                    best_links = links
            except Exception:
                pass

    return best_links


def write_links(links: list[str], out_file: Path, append: bool) -> None:
    mode = "a" if append else "w"

    if append and out_file.exists() and out_file.stat().st_size > 0:
        prefix = "\n"
    else:
        prefix = ""

    with out_file.open(mode, encoding="utf-8") as f:
        if prefix:
            f.write(prefix)
        f.write("\n".join(links))
        if links:
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract all uploaded video links from a YouTube channel URL."
    )
    parser.add_argument("channel_url", help="YouTube channel URL (channel/@handle/videos/...) ")
    parser.add_argument(
        "-o",
        "--output",
        default="youtube_links.txt",
        help="Output TXT file path (default: youtube_links.txt)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append links to output instead of overwriting.",
    )

    args = parser.parse_args()
    out_file = Path(args.output)

    links = extract_video_links(args.channel_url)

    if not links:
        raise RuntimeError("No video links found. Check the channel URL and try again.")

    write_links(links, out_file, append=args.append)
    print(f"Extracted {len(links)} video links -> {out_file}")


if __name__ == "__main__":
    main()
