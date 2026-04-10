import re


def classify_score(score, strong_threshold, maybe_threshold, strong_label="TARGET"):
    if score >= strong_threshold:
        return strong_label.upper()
    if score >= maybe_threshold:
        return "MAYBE"
    return "unknown"


def clean_word(word):
    return word.strip()


def normalize_word_for_counting(word: str) -> str:
    word = word.strip().lower()
    word = re.sub(r"^[^\wäöüß]+|[^\wäöüß]+$", "", word, flags=re.IGNORECASE)
    word = re.sub(r"\s+", " ", word)
    return word
