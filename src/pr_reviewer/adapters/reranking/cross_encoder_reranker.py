from sentence_transformers import CrossEncoder

from src.pr_reviewer.rag.ports import Reranker

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 512,
                 batch_size: int = 8, device: str | None = None):
        # Loads once; reused for the app's lifetime (like your embedder).
        self._model = CrossEncoder(model_name, max_length=max_length, device=device)
        self._batch_size = batch_size

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(pairs, batch_size=self._batch_size)
        return [float(s) for s in scores]