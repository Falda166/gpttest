from collections import defaultdict
from pathlib import Path

import pandas as pd


def compare_speakers(word_counts_by_speaker: dict[str, dict[str, int]], output_csv: Path):
    speakers = sorted(word_counts_by_speaker.keys())
    all_words = set()
    for counts in word_counts_by_speaker.values():
        all_words.update(counts.keys())

    rows = []
    for word in sorted(all_words):
        row = {"word": word}
        for speaker in speakers:
            row[speaker] = int(word_counts_by_speaker.get(speaker, {}).get(word, 0))
        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    distinctive = []
    for speaker in speakers:
        for word in all_words:
            own = word_counts_by_speaker.get(speaker, {}).get(word, 0)
            others = sum(
                word_counts_by_speaker.get(other, {}).get(word, 0)
                for other in speakers
                if other != speaker
            )
            score = own - others
            if own > 0:
                distinctive.append({"speaker": speaker, "word": word, "distinctiveness": score, "count": own})

    distinct_df = pd.DataFrame(distinctive).sort_values(
        by=["speaker", "distinctiveness", "count"],
        ascending=[True, False, False],
    )
    distinct_path = output_csv.with_name("speaker_distinctive_words.csv")
    distinct_df.to_csv(distinct_path, index=False, encoding="utf-8")
    return df, distinct_df
