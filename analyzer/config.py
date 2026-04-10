import os
from pathlib import Path

INPUT_LINKS_FILE = Path("./youtube_links.txt")
YOUTUBE_CHANNEL_URL = os.getenv("YOUTUBE_CHANNEL_URL", "").strip()
YOUTUBE_MAX_LINKS = int(os.getenv("YOUTUBE_MAX_LINKS", "100"))
YOUTUBE_FETCH_ALL = os.getenv("YOUTUBE_FETCH_ALL", "true").lower() in {"1", "true", "yes", "on"}
YOUTUBE_PROXY = os.getenv("YOUTUBE_PROXY")
YOUTUBE_NO_PROXY = os.getenv("YOUTUBE_NO_PROXY", "false").lower() in {"1", "true", "yes", "on"}

OUTPUT_DIR = Path("./output")
PLOTS_DIR = OUTPUT_DIR / "plots"
CSV_DIR = OUTPUT_DIR / "csv"
AUDIO_DIR = OUTPUT_DIR / "audios"
CLEAN_AUDIO_DIR = OUTPUT_DIR / "cleaned_audios"
TRAINING_DIR = OUTPUT_DIR / "training"
PAPAPLATTE_TRAINING_DIR = TRAINING_DIR / "papaplatte"
CACHE_DIR = Path("./cache")

for d in [OUTPUT_DIR, PLOTS_DIR, CSV_DIR, AUDIO_DIR, CLEAN_AUDIO_DIR, TRAINING_DIR, PAPAPLATTE_TRAINING_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DB_FILE = "voice_db.json"
FINAL_CSV_FILE = OUTPUT_DIR / "word_frequency.csv"

TOPICS_CSV_FILE = CSV_DIR / "video_topics.csv"
VIDEO_SIMILARITY_CSV_FILE = CSV_DIR / "video_similarity.csv"
SPEAKER_STYLE_CSV_FILE = CSV_DIR / "speaker_style.csv"
WORD_CLUSTERS_HTML = OUTPUT_DIR / "word_clusters.html"
WORD_CLUSTERS_PLOT_HTML = PLOTS_DIR / "word_clusters.html"
WORD_TIMELINE_HTML = OUTPUT_DIR / "word_timeline.html"
WORD_TIMELINE_PLOT_HTML = PLOTS_DIR / "word_timeline.html"
RUNTIME_ESTIMATION_CSV = CSV_DIR / "runtime_estimation.csv"
RUNTIME_ESTIMATION_HTML = PLOTS_DIR / "runtime_estimation.html"
ANALYZED_VIDEOS_FILE = CSV_DIR / "analyzed_videos.json"

EMBEDDINGS_CACHE_FILE = CACHE_DIR / "embeddings.pkl"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN fehlt. Bitte als Umgebungsvariable setzen.")

WHISPERX_MODEL = "medium"
WHISPER_LANGUAGE = "de"
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")

MATCH_THRESHOLD_STRONG = 0.55
MATCH_THRESHOLD_MAYBE = 0.40
MIN_ACCEPT_SCORE = 0.40

MIN_SEGMENT_SECONDS = 1.5
AUDIO_OVERWRITE = True
SAVE_PAPAPLATTE_TRAINING_AUDIO = True
SPEAKER_PROFILE_VIDEOS = int(os.getenv("SPEAKER_PROFILE_VIDEOS", "2"))

# Audio-Bereinigung
SILENCE_THRESHOLD = 0.008  # RMS-Schwelle
MIN_SILENCE_SECONDS = 0.25
KEEP_SILENCE_SECONDS = 0.08
MIN_CHUNK_SECONDS = 0.20

# NLP-Erweiterung
CSV_CLEANUP_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CSV_SEMANTIC_THRESHOLD = 0.87
WORD_CLUSTER_MIN_SIZE = 2

QUIET_THIRD_PARTY_WARNINGS = True
ENABLE_TF32 = True
MIN_CLEANED_AUDIO_SECONDS = 20.0
