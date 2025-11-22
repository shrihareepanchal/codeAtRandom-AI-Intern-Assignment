import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from .config import (
    DATA_DIR,
    FAISS_INDEX_PATH,
    EMBEDDING_DIM,
    PREVIEW_CHARS,
    MIN_KEYWORD_LENGTH,
)
from .embedder import EmbeddingModel
from .cache_manager import CacheManager
from .utils import clean_text, compute_sha256, tokenize, filter_keywords

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class SearchEngine:
    """
    Main orchestration class.
    - Loads documents
    - Maintains embedding cache
    - Builds vector index
    - Supports semantic search + explanations
    """

    def __init__(
        self,
        docs_dir: Path = DATA_DIR,
        cache_manager: Optional[CacheManager] = None,
        embedder: Optional[EmbeddingModel] = None,
    ):
        self.docs_dir = docs_dir
        self.cache = cache_manager or CacheManager()
        self.embedder = embedder or EmbeddingModel()

        # In-memory document storage
        self.doc_texts: Dict[str, str] = {}     # doc_id -> cleaned text
        self.doc_filenames: Dict[str, str] = {} # doc_id -> filename
        self.doc_lengths: Dict[str, int] = {}   # doc_id -> length (tokens / words)

        # Embedding-related
        self.doc_ids: List[str] = []
        self.embedding_matrix: np.ndarray = np.zeros((0, EMBEDDING_DIM), dtype="float32")

        # Index
        self.faiss_index = None
        self.use_faiss = False

    # ---------- Document loading & indexing ----------

    def _iter_document_files(self) -> List[Path]:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory does not exist: {self.docs_dir}")
        return sorted(self.docs_dir.glob("*.txt"))

    def index_corpus(self) -> None:
        """
        Read all .txt files, compute/update embeddings with caching,
        then build a vector index.
        """
        files = self._iter_document_files()
        texts: List[str] = []
        doc_ids: List[str] = []
        embeddings: List[np.ndarray] = []
        lengths: List[int] = []

        now_str = datetime.datetime.utcnow().isoformat()

        for file_path in files:
            filename = file_path.name
            doc_id = filename  # simple: doc_id == filename, but can be customized

            raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_text(raw_text)
            content_hash = compute_sha256(cleaned)
            word_count = len(cleaned.split())

            # Try cache
            cache_entry = self.cache.get_entry(doc_id)
            if cache_entry and cache_entry["content_hash"] == content_hash:
                # reuse embedding
                embedding = cache_entry["embedding"]
            else:
                # recompute embedding and update cache
                embedding = self.embedder.embed_single(cleaned)
                self.cache.upsert_entry(
                    doc_id=doc_id,
                    filename=filename,
                    content_hash=content_hash,
                    embedding=embedding,
                    doc_length=word_count,
                    updated_at=now_str,
                )

            # collect in-memory structures
            self.doc_texts[doc_id] = cleaned
            self.doc_filenames[doc_id] = filename
            self.doc_lengths[doc_id] = word_count

            texts.append(cleaned)
            doc_ids.append(doc_id)
            embeddings.append(embedding.astype("float32"))
            lengths.append(word_count)

        if embeddings:
            self.doc_ids = doc_ids
            self.embedding_matrix = np.vstack(embeddings).astype("float32")
        else:
            self.doc_ids = []
            self.embedding_matrix = np.zeros((0, EMBEDDING_DIM), dtype="float32")

        # Build index
        self._build_index()

    def _build_index(self) -> None:
        """Create FAISS index if possible, otherwise rely on NumPy cosine similarity."""
        if self.embedding_matrix.size == 0:
            self.faiss_index = None
            self.use_faiss = False
            return

        # L2-normalize embeddings for cosine similarity via inner product
        norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True) + 1e-10
        normalized = self.embedding_matrix / norms

        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            index.add(normalized.astype("float32"))
            self.faiss_index = index
            self.use_faiss = True

            # Optionally, persist the FAISS index
            faiss.write_index(index, str(FAISS_INDEX_PATH))
        else:
            self.faiss_index = None
            self.use_faiss = False

        # Save back normalized matrix for manual cosine
        self.embedding_matrix = normalized

    # ---------- Search ----------

    def _cosine_search(
        self, query_vec: np.ndarray, top_k: int
    ) -> List[tuple[int, float]]:
        """
        Fallback search using pure NumPy cosine similarity.
        Returns list of (idx, score) sorted by score desc.
        """
        if self.embedding_matrix.size == 0:
            return []

        q = query_vec.reshape(1, -1).astype("float32")
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)

        sims = (self.embedding_matrix @ q_norm.T).flatten()
        top_k = min(top_k, sims.shape[0])
        idxs = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[int(i)])) for i in idxs]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        High-level search:
          1. Embed query
          2. Retrieve similar docs
          3. Build explanations
        """
        if not self.doc_ids:
            # No documents indexed
            return []

        cleaned_query = clean_text(query)
        query_vec = self.embedder.embed_single(cleaned_query).astype("float32")

        # Normalize query vector
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)

        if self.use_faiss and self.faiss_index is not None:
            D, I = self.faiss_index.search(
                query_vec.reshape(1, -1), min(top_k, len(self.doc_ids))
            )
            indices = I[0]
            scores = D[0]
            idx_score_pairs = [
                (int(idx), float(score))
                for idx, score in zip(indices, scores)
                if idx >= 0
            ]
        else:
            idx_score_pairs = self._cosine_search(query_vec, top_k)

        results: List[Dict[str, Any]] = []
        for idx, score in idx_score_pairs:
            doc_id = self.doc_ids[idx]
            text = self.doc_texts.get(doc_id, "")
            explanation = self._build_explanation(query=cleaned_query, doc_text=text)
            preview = text[:PREVIEW_CHARS]

            results.append(
                {
                    "doc_id": doc_id,
                    "score": float(score),
                    "preview": preview,
                    "explanation": explanation,
                }
            )

        return results

    # ---------- Ranking explanation ----------

    def _build_explanation(self, query: str, doc_text: str) -> Dict[str, Any]:
        """
        Simple heuristic:
        - Check overlap between query keywords and document keywords
        - Compute overlap ratio
        - Compute a crude length normalization factor
        """
        q_tokens = tokenize(query)
        d_tokens = tokenize(doc_text)

        q_keywords = [t for t in filter_keywords(q_tokens) if len(t) >= MIN_KEYWORD_LENGTH]
        d_keywords = [t for t in filter_keywords(d_tokens) if len(t) >= MIN_KEYWORD_LENGTH]

        q_set = set(q_keywords)
        d_set = set(d_keywords)

        overlap = sorted(q_set.intersection(d_set))
        overlap_ratio = (
            len(overlap) / len(q_set) if q_set else 0.0
        )

        # Length normalization: penalize very long documents
        doc_len = max(len(d_tokens), 1)
        length_norm = 1.0 / (1.0 + np.log1p(doc_len))

        return {
            "matched_keywords": overlap,
            "overlap_ratio": overlap_ratio,
            "length_normalization": float(length_norm),
            "query_keywords": list(q_set),
        }
