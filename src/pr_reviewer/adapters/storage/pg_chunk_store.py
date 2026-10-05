"""ChunkStore on PostgreSQL + pgvector.
Needs: psycopg[binary]>=3.1, pgvector, numpy
"""
import re

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from src.pr_reviewer.rag.code_chunker import CodeChunk
from src.pr_reviewer.rag.ports import ChunkKey, ChunkStore

_UPSERT = """
INSERT INTO code_chunks
    (repo, file_path, chunkname, kind, start_line, end_line,
     content, content_hash, commit_sha, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (repo, file_path, chunkname) DO UPDATE SET
    kind         = EXCLUDED.kind,
    start_line   = EXCLUDED.start_line,
    end_line     = EXCLUDED.end_line,
    content      = EXCLUDED.content,
    content_hash = EXCLUDED.content_hash,
    commit_sha   = EXCLUDED.commit_sha,
    embedding    = EXCLUDED.embedding,
    updated_at   = now()
"""

_UPDATE_LOCATION = """
UPDATE code_chunks
SET start_line = %s, end_line = %s, commit_sha = %s, updated_at = now()
WHERE repo = %s AND file_path = %s AND chunkname = %s
"""


def _normalize_dsn(dsn: str) -> str:
    """Accepts SQLAlchemy-style URLs (postgresql+psycopg2://...) as well as
    plain ones. psycopg itself only understands `postgresql://` / `postgres://`."""
    return re.sub(r"^(postgres(?:ql)?)\+\w+://", r"\1://", dsn)


class PgChunkStore(ChunkStore):
    def __init__(self, dsn: str):
        self._dsn = _normalize_dsn(dsn)

    def _connect(self) -> psycopg.Connection:
        # One short-lived connection per operation: simple and thread-safe
        # (the worker runs in its own thread). Swap for a pool if it matters.
        conn = psycopg.connect(self._dsn)
        register_vector(conn)
        return conn

    def get_hashes(self, repo: str, file_paths: list[str] | None = None) -> dict[ChunkKey, str]:
        sql = "SELECT file_path, chunkname, content_hash FROM code_chunks WHERE repo = %s"
        params: list = [repo]
        if file_paths is not None:
            sql += " AND file_path = ANY(%s)"
            params.append(file_paths)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {(fp, qn): h for fp, qn, h in rows}

    def get_indexed_sha(self, repo: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT indexed_sha FROM index_state WHERE repo = %s", (repo,)
            ).fetchone()
        return row[0] if row else None

    def set_indexed_sha(self, repo: str, sha: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO index_state (repo, indexed_sha) VALUES (%s, %s)
                   ON CONFLICT (repo) DO UPDATE
                   SET indexed_sha = EXCLUDED.indexed_sha, updated_at = now()""",
                (repo, sha),
            )

    def upsert(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        params = [
            (c.repo, c.file_path, c.chunkname, c.kind, c.start_line, c.end_line,
             c.content, c.content_hash, c.commit_sha, np.asarray(e, dtype=np.float32))
            for c, e in zip(chunks, embeddings)
        ]
        with self._connect() as conn:
            conn.cursor().executemany(_UPSERT, params)

    def update_locations(self, chunks: list[CodeChunk]) -> None:
        params = [
            (c.start_line, c.end_line, c.commit_sha, c.repo, c.file_path, c.chunkname)
            for c in chunks
        ]
        with self._connect() as conn:
            conn.cursor().executemany(_UPDATE_LOCATION, params)

    def delete(self, repo: str, keys: list[ChunkKey]) -> None:
        with self._connect() as conn:
            conn.cursor().executemany(
                "DELETE FROM code_chunks WHERE repo = %s AND file_path = %s AND chunkname = %s",
                [(repo, fp, qn) for fp, qn in keys],
            )

    def search(self, repo: str, query_embedding: list[float], k: int = 5) -> list[dict]:
        vec = np.asarray(query_embedding, dtype=np.float32)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT file_path, chunkname, kind, start_line, end_line, content,
                       1 - (embedding <=> %s) AS similarity
                FROM code_chunks
                WHERE repo = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vec, repo, vec, k),
            ).fetchall()
        cols = ("file_path", "chunkname", "kind", "start_line", "end_line", "content", "similarity")
        return [dict(zip(cols, r)) for r in rows]