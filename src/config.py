import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "docs"
CACHE_DIR = BASE_DIR / "cache"

# Create dirs if missing
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding / caching paths
SQLITE_PATH = CACHE_DIR / "embeddings.db"
FAISS_INDEX_PATH = CACHE_DIR / "index.faiss"

# Embedding model name (SentenceTransformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Number of dimensions for the chosen model (MiniLM-L6-v2 → 384)
EMBEDDING_DIM = 384

# Preview length (characters) for API responses
PREVIEW_CHARS = 250

# Minimum token length for keyword overlap
MIN_KEYWORD_LENGTH = 3
