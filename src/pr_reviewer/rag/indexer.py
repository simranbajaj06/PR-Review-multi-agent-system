"""Chunks -> embeddings -> DB, doing as little work as possible:
  * new or changed chunk (hash differs)  -> embed + upsert
  * unchanged content                    -> skip embedding, refresh line numbers
  * chunk no longer in the code          -> delete the stale row

Two entry points:
  index()        whole repo (first run / fallback). Deletes anything not in `chunks`.
  index_files()  only the files that changed in a push. Scoped: it never
                 touches rows belonging to other files.
"""
import logging
from dataclasses import dataclass

from src.pr_reviewer.rag.code_chunker import CodeChunk
from src.pr_reviewer.rag.ports import ChunkStore, Embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexStats:
    embedded: int
    unchanged: int
    deleted: int


class ChunkIndexer:
    def __init__(self, embedder: Embedder, store: ChunkStore, batch_size: int = 16):
        self._embedder = embedder
        self._store = store
        self._batch_size = batch_size

    def _embed_and_store(self, existing, chunks: list[CodeChunk]) -> tuple[int, int]:
        changed = [c for c in chunks if existing.get((c.file_path, c.chunkname)) != c.content_hash]
        unchanged = [c for c in chunks if existing.get((c.file_path, c.chunkname)) == c.content_hash]

        # Embed + write in batches so a crash midway keeps the progress made.
        for i in range(0, len(changed), self._batch_size):
            batch = changed[i:i + self._batch_size]
            vectors = self._embedder.embed_documents([c.to_embedding_text() for c in batch])
            self._store.upsert(batch, vectors)
            logger.info("embedded %d/%d chunks", i + len(batch), len(changed))

        if unchanged:
            self._store.update_locations(unchanged)
        return len(changed), len(unchanged)

    def index(self, repo: str, chunks: list[CodeChunk]) -> IndexStats:
        existing = self._store.get_hashes(repo)
        embedded, unchanged = self._embed_and_store(existing, chunks)

        deleted = 0
        # Safety: an empty list almost certainly means a failed checkout/parse,
        # not "the repo has no code". Never wipe the index for that.
        if chunks:
            current = {(c.file_path, c.chunkname) for c in chunks}
            stale = [k for k in existing if k not in current]
            if stale:
                self._store.delete(repo, stale)
            deleted = len(stale)
        else:
            logger.warning("no chunks produced for %s; skipping stale-row deletion", repo)

        stats = IndexStats(embedded, unchanged, deleted)
        logger.info("full index repo=%s %s", repo, stats)
        return stats

    def index_files(
        self, repo: str, files: dict[str, list[CodeChunk]], deleted_paths: list[str]
    ) -> IndexStats:
        """`files`: path -> that file's current chunks (replaces what's stored
        for the file). `deleted_paths`: files that no longer exist."""
        scope = sorted(set(files) | set(deleted_paths))
        if not scope:
            return IndexStats(0, 0, 0)

        existing = self._store.get_hashes(repo, scope)
        chunks = [c for cs in files.values() for c in cs]
        embedded, unchanged = self._embed_and_store(existing, chunks)

        current = {(c.file_path, c.chunkname) for c in chunks}
        stale = [k for k in existing if k not in current]  # existing is scoped to changed files only
        if stale:
            self._store.delete(repo, stale)

        stats = IndexStats(embedded, unchanged, len(stale))
        logger.info("incremental index repo=%s files=%d %s", repo, len(scope), stats)
        return stats