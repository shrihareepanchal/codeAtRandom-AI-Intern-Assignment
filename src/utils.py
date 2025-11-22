import hashlib
import re
from typing import List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, collapse whitespace."""
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_sha256(text: str) -> str:
    """Hash text content for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(text: str) -> List[str]:
    """Very lightweight tokenizer, returns words."""
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    tokens = [t.strip().lower() for t in text.split() if t.strip()]
    return tokens


def filter_keywords(tokens: List[str]) -> List[str]:
    """Filter out stopwords and short tokens to keep only meaningful keywords."""
    keywords = [
        t for t in tokens
        if t not in ENGLISH_STOP_WORDS and len(t) >= 3
    ]
    return keywords
