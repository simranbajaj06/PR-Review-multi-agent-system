"""Keeps code_chunks in sync with a repo's branch, as cheaply as possible.

  sync(repo)            -> brings the index up to the branch head
  sync(repo, ref=sha)   -> ...or to a specific commit (e.g. a push webhook's `after`)

First time (no stored index state)          -> full snapshot download + index
Same commit as last time                    -> nothing to do
Otherwise                                   -> ask GitHub which files changed between
                                               the last indexed commit and the new one,
                                               fetch ONLY those files, re-chunk them,
                                               embed only chunks whose hash changed.
Fallbacks to a full index: history rewritten (force-push), or more than
MAX_INCREMENTAL_FILES files changed (a snapshot is cheaper than N requests).

Run it from ONE thread/process at a time per repo (the worker thread is fine);
two concurrent syncs could race on index state.
"""
import ast
import logging

from src.pr_reviewer.rag.code_chunker import SKIP_DIRS, CodeChunk, chunk_python_source, chunk_repo
from src.pr_reviewer.rag.indexer import ChunkIndexer
from src.pr_reviewer.rag.ports import ChunkStore, FileChange, RepoSource

logger = logging.getLogger(__name__)

MAX_INCREMENTAL_FILES = 60


def _is_indexable(path: str) -> bool:
    parts = path.split("/")
    return path.endswith(".py") and not any(p in SKIP_DIRS for p in parts[:-1])


class IndexUpdater:
    def __init__(self, source: RepoSource, store: ChunkStore, indexer: ChunkIndexer):
        self._source = source
        self._store = store
        self._indexer = indexer

    def sync(self, repo: str, ref: str | None = None) -> str:
        head = self._source.resolve_ref(repo, ref)
        base = self._store.get_indexed_sha(repo)

        if base == head:
            logger.info("repo=%s already indexed at %s", repo, head[:7])
            return head

        changes = self._source.changed_files(repo, base, head) if base else None
        if changes is None or len(changes) > MAX_INCREMENTAL_FILES:
            logger.info("repo=%s full index @ %s (base=%s)", repo, head[:7], base and base[:7])
            if not self._full(repo, head):
                # Don't record this commit as indexed: nothing was indexed, and
                # recording it would make later runs think the repo is up to date.
                logger.warning(
                    "repo=%s: no indexable Python code found at %s; index state not advanced",
                    repo, head[:7],
                )
                return head
        else:
            logger.info("repo=%s incremental %s -> %s (%d files)", repo, base[:7], head[:7], len(changes))
            self._incremental(repo, head, changes)

        # Only reached if indexing didn't raise, so a failed run is retried
        # from the same base next time.
        self._store.set_indexed_sha(repo, head)
        return head

    def _full(self, repo: str, head: str) -> bool:
        """Returns False if the checkout produced no chunks at all."""
        with self._source.snapshot(repo, head) as snap:
            chunks = chunk_repo(snap.path, repo, head)
        self._indexer.index(repo, chunks)
        return bool(chunks)

    def _incremental(self, repo: str, head: str, changes: list[FileChange]) -> None:
        files: dict[str, list[CodeChunk]] = {}
        deleted: list[str] = []

        for ch in changes:
            if ch.status == "renamed" and ch.previous_path and _is_indexable(ch.previous_path):
                deleted.append(ch.previous_path)  # key includes file_path, so old rows must go
            if not _is_indexable(ch.path):
                continue
            if ch.status == "removed":
                deleted.append(ch.path)
                continue

            text = self._source.read_file(repo, ch.path, head)
            if text is None:
                deleted.append(ch.path)
                continue
            try:
                ast.parse(text)
            except SyntaxError as exc:
                # Keep the previously indexed chunks rather than wiping the file.
                logger.warning("skipping %s: syntax error: %s", ch.path, exc)
                continue
            files[ch.path] = chunk_python_source(text, ch.path, repo, head)

        deleted = [p for p in dict.fromkeys(deleted) if p not in files]
        self._indexer.index_files(repo, files, deleted)