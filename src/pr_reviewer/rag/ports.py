"""Ports for the RAG layer. The indexer/updater depend on these, never on
sentence-transformers, psycopg or requests directly."""
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

from src.pr_reviewer.rag.code_chunker import CodeChunk

ChunkKey = tuple[str, str]  # (file_path, chunkname) - unique within a repo


@dataclass(frozen=True)
class FileChange:
    path: str
    status: str                       # added | modified | removed | renamed
    previous_path: str | None = None  # set for renames


@dataclass(frozen=True)
class Snapshot:
    repo: str
    ref: str
    commit_sha: str
    path: Path


class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector size. Must equal the `vector(N)` column in code_chunks."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds code chunks (no instruction prefix)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embeds a search query (uses the model's query prompt)."""


class ChunkStore(ABC):
    @abstractmethod
    def get_hashes(self, repo: str, file_paths: list[str] | None = None) -> dict[ChunkKey, str]:
        """{(file_path, chunkname): content_hash} for `repo`, optionally limited to some files."""

    @abstractmethod
    def upsert(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        ...

    @abstractmethod
    def update_locations(self, chunks: list[CodeChunk]) -> None:
        """Unchanged content: refresh line numbers + commit_sha only."""

    @abstractmethod
    def delete(self, repo: str, keys: list[ChunkKey]) -> None:
        ...

    @abstractmethod
    def search(self, repo: str, query_embedding: list[float], k: int = 5) -> list[dict]:
        ...

    @abstractmethod
    def get_indexed_sha(self, repo: str) -> str | None:
        """The commit the index currently reflects, or None if never indexed."""

    @abstractmethod
    def set_indexed_sha(self, repo: str, sha: str) -> None:
        ...


class RepoSource(ABC):
    """Where repo code comes from (GitHub today)."""

    @abstractmethod
    def resolve_ref(self, repo: str, ref: str | None) -> str:
        """Branch/tag/sha (None = default branch) -> exact commit sha."""

    @abstractmethod
    def changed_files(self, repo: str, base: str, head: str) -> list[FileChange] | None:
        """Files changed between two commits, or None if that can't be
        computed reliably (force-push, too many files) -> caller does a full index."""

    @abstractmethod
    def read_file(self, repo: str, path: str, ref: str) -> str | None:
        """File text at `ref`, or None if it doesn't exist there."""

    @abstractmethod
    def snapshot(self, repo: str, sha: str) -> AbstractContextManager[Snapshot]:
        """Full checkout in a temp dir, deleted on exit."""