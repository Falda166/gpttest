# YouTube Speaker-Targeted Word Frequency Analyzer

A small Python project that downloads YouTube audio, transcribes German speech with **WhisperX**, performs speaker diarization with **pyannote**, matches a target speaker embedding from a local voice database, and exports a word-frequency CSV.

> This project is intended for research/educational use. Always ensure you have the right to process the media you analyze.

## Features

- Reads YouTube links from a plain text file
- Can auto-extract all uploaded video links from a YouTube channel URL
- Downloads and converts audio to WAV using `yt-dlp` + ffmpeg
- Cleans audio by removing silent sections before ASR/diarization
- Transcribes with WhisperX + word-level alignment
- Performs diarization and overlap filtering
- Matches a known reference voice embedding (e.g. `papaplatte`) against detected speakers
- Applies 2-step CSV cleanup (rule-based + semantic merge with multilingual MPNet)
- Outputs cleaned word counts to `output/word_frequency.csv`
- Colorized step-by-step logs with duration tracking
- Runtime tweaks for cleaner logs (optional warning filters + TF32 enable)

## Project Structure

```text
.
├── app.py
├── analyzer/
│   ├── config.py
│   ├── audio_processing.py
│   ├── speaker_processing.py
│   ├── text_processing.py
│   ├── helpers.py
│   ├── logging_utils.py
│   ├── csv_cleanup.py
│   ├── runtime.py
│   └── pipeline.py
├── extract_channel_links.py
├── requirements.txt
├── .env.example
├── youtube_links.example.txt
├── voice_db.example.json
├── .gitignore
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- ffmpeg installed and available in `PATH`
- Hugging Face access token (`HF_TOKEN`) with access to required pyannote models
- GPU is recommended but optional
- Additional model for CSV cleanup: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

## Quick Start

### 1) Clone and install

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2) Configure environment

```bash
cp .env.example .env
```

Set your token:

```env
HF_TOKEN=hf_xxx
```

### 3) Prepare inputs

- Copy `youtube_links.example.txt` to `youtube_links.txt` and add one YouTube URL per line.
- Copy `voice_db.example.json` to `voice_db.json` and add your real target embedding values.

Optional: generate `youtube_links.txt` automatically from a full YouTube channel:

```bash
python extract_channel_links.py "https://www.youtube.com/@channel_handle/videos"
```

Nur die ersten 25 Links ziehen:

```bash
python extract_channel_links.py "https://www.youtube.com/@channel_handle" -n 25
```

Alle verfügbaren Links ziehen:

```bash
python extract_channel_links.py "https://www.youtube.com/@channel_handle" --all
```

By default this overwrites `youtube_links.txt`. To append instead:

```bash
python extract_channel_links.py "https://www.youtube.com/@channel_handle/videos" --append
```

Wenn ein Umgebungs-Proxy Probleme macht:

```bash
python extract_channel_links.py "https://www.youtube.com/@channel_handle" --no-proxy
```

### 4) Run

```bash
python app.py
```

Output will be written to:

- `output/word_frequency.csv`

## Notes on `voice_db.json`

Expected format:

```json
{
  "papaplatte": {
    "embedding": [0.12, -0.03, 0.44]
  }
}
```

The embedding vector should be a 1D float vector from the same (or compatible) speaker embedding model family used in this pipeline.

## Legal & Privacy

- Respect YouTube Terms of Service and applicable law.
- Only process data you are legally allowed to use.
- If publishing results, anonymize where necessary.

## Troubleshooting

- **`HF_TOKEN fehlt`**: define `HF_TOKEN` in your environment.
- **ffmpeg errors**: verify ffmpeg installation (`ffmpeg -version`).
- **Model access denied**: ensure token permissions and model access are granted on Hugging Face.
- **Very slow runtime**: use GPU (`cuda`) where possible.

## License

MIT — see [LICENSE](LICENSE).


## Full NLP Extension

Neue Module und Outputs:

- `analyzer/visualization.py` -> `output/word_clusters.html`
- `analyzer/word_clustering.py` -> semantische Wort-Normalisierung mit HDBSCAN
- `analyzer/topic_detection.py` -> `output/csv/video_topics.csv`
- `analyzer/video_similarity.py` -> `output/csv/video_similarity.csv`
- `analyzer/speaker_style.py` -> `output/csv/speaker_style.csv`
- `analyzer/time_analysis.py` -> `output/word_timeline.html`
- `analyzer/embedding_cache.py` -> `cache/embeddings.pkl`
- `dashboard.py` -> Streamlit Dashboard mit 6 Tabs

Start Dashboard:

```bash
streamlit run dashboard.py
```

## Merge-Konflikte vermeiden

Für die Kern-Dateien ist in `.gitattributes` `merge=ours` gesetzt, damit bei Merges standardmäßig die aktuelle Branch-Version behalten wird ("einfach überschreiben").
