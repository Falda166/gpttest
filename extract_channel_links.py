import argparse
from pathlib import Path

import yt_dlp


def normalize_channel_url(channel_url: str) -> str:
    channel_url = channel_url.strip()
    if "/watch?v=" in channel_url:
        raise ValueError("Bitte einen Channel/Handle-Link übergeben, keinen einzelnen Video-Link.")

    # Für Handles und Channel-Seiten zuverlässig auf /videos gehen
    if "youtube.com/@" in channel_url or "youtube.com/channel/" in channel_url:
        if "/videos" not in channel_url:
            channel_url = channel_url.rstrip("/") + "/videos"

    return channel_url


def extract_video_links(channel_url: str, max_links: int | None = None, proxy: str | None = None) -> list[str]:
    """Extract uploaded YouTube watch links from a channel URL.

    max_links=None -> alle verfügbaren Videos
    """
    channel_url = normalize_channel_url(channel_url)

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "lazy_playlist": False,
    }
    if max_links is not None:
        ydl_opts["playlistend"] = int(max_links)
    if proxy is not None:
        ydl_opts["proxy"] = proxy

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

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
        elif webpage_url and "watch?v=" in webpage_url:
            video_url = webpage_url
        elif url and "watch?v=" in url:
            video_url = url
        else:
            continue

        if video_url not in seen:
            seen.add(video_url)
            urls.append(video_url)

        if max_links is not None and len(urls) >= max_links:
            break

    return urls


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
        description="Extract uploaded YouTube watch links from a channel URL."
    )
    parser.add_argument("channel_url", help="YouTube channel URL (/@handle, /channel/<id>, ...)")
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
    parser.add_argument(
        "-n",
        "--max-links",
        type=int,
        default=None,
        help="Maximale Anzahl Links (standard: alle verfügbaren).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ignoriert --max-links und versucht alle Videos zu holen.",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Optionaler Proxy (z.B. http://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Proxy-Nutzung deaktivieren (überschreibt Umgebungsproxy).",
    )

    args = parser.parse_args()
    out_file = Path(args.output)

    max_links = None if args.all else args.max_links
    proxy = "" if args.no_proxy else args.proxy
    try:
        links = extract_video_links(args.channel_url, max_links=max_links, proxy=proxy)
    except Exception as e:
        raise RuntimeError(
            "Fehler beim Auslesen der Channel-Seite. Wenn du hinter einem Proxy sitzt, "
            "teste --no-proxy oder gib --proxy explizit an."
        ) from e

    if not links:
        raise RuntimeError("No video links found. Check the channel URL and try again.")

    write_links(links, out_file, append=args.append)
    scope = "alle" if max_links is None else str(max_links)
    print(f"Extracted {len(links)} video links (Ziel: {scope}) -> {out_file}")


if __name__ == "__main__":
    main()
