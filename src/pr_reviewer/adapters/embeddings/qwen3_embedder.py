"""Embedder backed by Qwen3-Embedding-0.6B running locally (free, no API)."""
import logging

from sentence_transformers import SentenceTransformer

from src.pr_reviewer.rag.ports import Embedder

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# 6,000-char chunks are ~1.5-2k tokens. Capping bounds memory on CPU/small GPUs.
MAX_SEQ_LENGTH = 2048


class Qwen3Embedder(Embedder):
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        # Downloads ~1.2 GB on first run, then loads from the HF cache.
        # Needs: sentence-transformers>=2.7, transformers>=4.51
        self._model = SentenceTransformer(model_name, device=device)
        self._model.max_seq_length = MAX_SEQ_LENGTH
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("loaded %s dim=%d device=%s", model_name, self._dim, self._model.device)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            [text], prompt_name="query", normalize_embeddings=True, show_progress_bar=False
        )[0]
        return vector.tolist()

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            prompt_name="query",          # same prompt your embed_query uses
            batch_size=8,
            normalize_embeddings=True,    # keep identical to embed_query
        )
        return vecs.tolist()