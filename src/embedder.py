from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """
    Wrapper around SentenceTransformers to turn text into dense vectors.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts.
        Returns: np.ndarray of shape (n_docs, dim)
        """
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we'll handle normalization separately
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """
        Convenience function to embed a single text (query).
        Returns: np.ndarray of shape (dim,)
        """
        return self.embed_texts([text])[0]
