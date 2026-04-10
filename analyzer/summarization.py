from pathlib import Path

import pandas as pd


def _fallback_summary(text: str, max_chars: int = 500) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text[:max_chars].strip()


def summarize_videos(
    video_texts: dict[str, str],
    model_name: str,
    output_csv: Path,
    min_words: int = 40,
    max_words: int = 120,
):
    if not video_texts:
        out = pd.DataFrame(columns=["video_id", "summary"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False, encoding="utf-8")
        return out

    summarizer = None
    try:
        from transformers import pipeline

        summarizer = pipeline("summarization", model=model_name)
    except Exception:
        summarizer = None

    rows = []
    for video_id, text in video_texts.items():
        clean_text = (text or "").strip()
        if not clean_text:
            rows.append({"video_id": video_id, "summary": ""})
            continue

        if summarizer is None:
            summary = _fallback_summary(clean_text)
        else:
            try:
                summary = summarizer(
                    clean_text[:4000],
                    min_length=min_words,
                    max_length=max_words,
                    do_sample=False,
                    truncation=True,
                )[0]["summary_text"].strip()
            except Exception:
                summary = _fallback_summary(clean_text)

        rows.append({"video_id": video_id, "summary": summary})

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    return out
