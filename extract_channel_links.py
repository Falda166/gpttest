import argparse
from pathlib import Path

import yt_dlp


def extract_video_links(channel_url: str) -> list[str]:
    """Extract all video URLs from a YouTube channel/uploads feed."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "lazy_playlist": False,
    }

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
