import sqlite3
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

from .config import SQLITE_PATH


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_embeddings (
    doc_id TEXT PRIMARY KEY,
    filename TEXT,
    content_hash TEXT,
    embedding BLOB,
    doc_length INTEGER,
    updated_at TEXT
);
"""


class CacheManager:
    """
    Handles persistence of document embeddings in SQLite.
    Each row stores document metadata + serialized vector.
    """

    def __init__(self, db_path: Path = SQLITE_PATH):
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()

    def get_entry(self, doc_id: str) -> Dict[str, Any] | None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT doc_id, filename, content_hash, embedding, doc_length, updated_at "
                "FROM document_embeddings WHERE doc_id = ?",
                (doc_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            embedding = pickle.loads(row[3])
            return {
                "doc_id": row[0],
                "filename": row[1],
                "content_hash": row[2],
                "embedding": embedding,
                "doc_length": row[4],
                "updated_at": row[5],
            }
        finally:
            conn.close()

    def upsert_entry(
        self,
        doc_id: str,
        filename: str,
        content_hash: str,
        embedding: np.ndarray,
        doc_length: int,
        updated_at: str,
    ) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            blob = pickle.dumps(embedding.astype("float32"))
            cur.execute(
                """
                INSERT INTO document_embeddings
                (doc_id, filename, content_hash, embedding, doc_length, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    content_hash=excluded.content_hash,
                    embedding=excluded.embedding,
                    doc_length=excluded.doc_length,
                    updated_at=excluded.updated_at;
                """,
                (doc_id, filename, content_hash, blob, doc_length, updated_at),
            )
            conn.commit()
        finally:
            conn.close()

    def load_all_embeddings(self) -> Tuple[List[str], np.ndarray, List[int]]:
        """
        Returns:
            doc_ids: list of document IDs
            embeddings: matrix of shape (n_docs, dim)
            doc_lengths: list of document lengths
        """
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT doc_id, embedding, doc_length FROM document_embeddings"
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        doc_ids: List[str] = []
        vectors: List[np.ndarray] = []
        lengths: List[int] = []

        for row in rows:
            doc_ids.append(row[0])
            vectors.append(pickle.loads(row[1]))
            lengths.append(row[2])

        if vectors:
            matrix = np.vstack(vectors)
        else:
            matrix = np.zeros((0, 0), dtype="float32")

        return doc_ids, matrix, lengths
