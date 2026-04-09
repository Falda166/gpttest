import os
from pathlib import Path

INPUT_LINKS_FILE = Path("./youtube_links.txt")

OUTPUT_DIR = Path("./output")
AUDIO_DIR = OUTPUT_DIR / "audios"
CLEAN_AUDIO_DIR = OUTPUT_DIR / "cleaned_audios"
OUTPUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)
CLEAN_AUDIO_DIR.mkdir(exist_ok=True)

DB_FILE = "voice_db.json"
FINAL_CSV_FILE = OUTPUT_DIR / "word_frequency.csv"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN fehlt. Bitte als Umgebungsvariable setzen.")

WHISPERX_MODEL = "medium"
WHISPER_LANGUAGE = "de"

MATCH_THRESHOLD_STRONG = 0.55
MATCH_THRESHOLD_MAYBE = 0.40
MIN_ACCEPT_SCORE = 0.40

MIN_SEGMENT_SECONDS = 1.5
AUDIO_OVERWRITE = True

# Audio-Bereinigung
SILENCE_THRESHOLD = 0.008  # RMS-Schwelle
MIN_SILENCE_SECONDS = 0.25
KEEP_SILENCE_SECONDS = 0.08
MIN_CHUNK_SECONDS = 0.20

CSV_CLEANUP_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CSV_SEMANTIC_THRESHOLD = 0.87
