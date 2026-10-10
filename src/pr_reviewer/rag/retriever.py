"""Turns a PR diff into relevant *existing* code from the RAG index.
(src/pr_reviewer/rag/retriever.py)

    diff --> per-hunk queries --> embed_query --> store.search
         --> drop low-similarity hits
         --> drop chunks that ARE the code being modified
         --> dedupe --> best-first within a size budget

Depends only on the Embedder / ChunkStore ports.

About chunks from files the PR touches: the index reflects the base branch,
so a chunk overlapping a changed hunk is the *pre-change* version of code the
diff already shows. Including it would duplicate the diff and could confuse
the model about what's current, so such chunks are dropped (old-side line
ranges from the hunk headers are compared with the chunk's line range).
Untouched chunks in the same file are kept: they're valid base-branch context.
"""
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter

from src.pr_reviewer.rag.ports import ChunkStore, Embedder, Reranker

logger = logging.getLogger(__name__)

_INDEXED_SUFFIXES = (".py",)  # the chunker only indexes Python today
_DIFF_HEADER = re.compile(r"^diff --git a/.+? b/(.+)$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")

MAX_QUERY_CHARS = 1_500   # keep each query well under the embedder's max length
MIN_CHANGE_CHARS = 20     # hunks with less changed text than this aren't worth a query


@dataclass(frozen=True)
class RetrievedChunk:
    file_path: str
    chunkname: str
    kind: str
    start_line: int
    end_line: int
    content: str
    similarity: float
    score: float = 0.0     


@dataclass
class _Hunk:
    path: str
    old_start: int
    old_end: int          # inclusive; == old_start for pure insertions
    scope: str            # text after the closing @@ (usually the enclosing def/class)
    added: list[str]
    removed: list[str]

MAX_RERANK_DOC_CHARS = 2_000

def _doc_text(row: dict) -> str:
    return (f"# {row['file_path']} :: {row['chunkname']} ({row['kind']})\n"
            f"{row['content']}")[:MAX_RERANK_DOC_CHARS]
def _parse_diff(diff: str) -> list[_Hunk]:
    """Hunks of indexable files only. Deleted files and binary files yield none."""
    hunks: list[_Hunk] = []
    path: str | None = None
    cur: _Hunk | None = None

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            m = _DIFF_HEADER.match(line)
            path = m.group(1) if m else None
            cur = None
            continue

        m = _HUNK.match(line)  # content lines start with + - space or \, never "@@"
        if m:
            if path and path.endswith(_INDEXED_SUFFIXES):
                old_start = int(m.group(1))
                old_len = 1 if m.group(2) is None else int(m.group(2))
                cur = _Hunk(path, old_start, old_start + max(old_len, 1) - 1,
                            m.group(5).strip(), [], [])
                hunks.append(cur)
            else:
                cur = None
            continue

        if cur is None:
            if line.startswith("+++ "):  # file header, only before the first hunk
                target = line[4:].split("\t")[0].strip()
                if target == "/dev/null":
                    path = None  # deleted file
                elif target.startswith("b/"):
                    path = target[2:]
            continue

        if line.startswith("+"):
            cur.added.append(line[1:])
        elif line.startswith("-"):
            cur.removed.append(line[1:])
    return hunks


def _build_query(h: _Hunk) -> str | None:
    """Query text for one hunk, or None if the change is too trivial."""
    body = "\n".join(l for l in (h.added or h.removed) if l.strip())
    if len(body.strip()) < MIN_CHANGE_CHARS:
        return None
    # Mirrors the header used when chunks were embedded (see to_embedding_text).
    parts = [f"# {h.path}"]
    if h.scope:
        parts.append(f"# in: {h.scope}")
    parts.append(body)
    return "\n".join(parts)[:MAX_QUERY_CHARS]


def _overlaps(row: dict, ranges: list[tuple[int, int]]) -> bool:
    return any(row["start_line"] <= end and row["end_line"] >= start for start, end in ranges)


class ContextRetriever:
    def __init__(
        self,
        embedder: Embedder,
        store: ChunkStore,
        reranker: Reranker | None = None,
        *,
        k_per_query: int = 4,
        fetch_k: int = 15,                      
        min_similarity: float = 0.45,           
        min_rerank_score: float | None = None,  
        max_queries: int = 12,
        max_context_chars: int = 12_000,
        include_modified_chunks: bool = False,
    ):
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._k = k_per_query
        self._fetch_k = fetch_k
        self._min_similarity = min_similarity
        self._min_rerank_score = min_rerank_score
        self._max_queries = max_queries
        self._max_chars = max_context_chars
        self._include_modified = include_modified_chunks

    def retrieve(self, repo: str, diff: str) -> list[RetrievedChunk]:
            hunks = _parse_diff(diff)
            if not hunks:
                return []

            modified: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for h in hunks:
                modified[h.path].append((h.old_start, h.old_end))

            queries = [q for q in (_build_query(h) for h in hunks) if q]
            queries.sort(key=len, reverse=True)
            queries = queries[: self._max_queries]
            if not queries:
                return []

            # 1) embed all queries in one pass
            t0 = perf_counter()
            vecs = self._embedder.embed_queries(queries)
            t_embed = perf_counter() - t0

            # 2) vector search: collect candidates per query
            t0 = perf_counter()
            fetch_k = self._fetch_k if self._reranker else self._k
            per_query: list[list[dict]] = []
            for vec in vecs:
                rows = []
                for row in self._store.search(repo, vec, k=fetch_k):
                    if row["similarity"] < self._min_similarity:
                        continue
                    if not self._include_modified and _overlaps(row, modified.get(row["file_path"], [])):
                        continue
                    rows.append(row)
                per_query.append(rows)
            t_search = perf_counter() - t0

            # 3) rerank (one batched model call for all pairs), keep top k per query
            t0 = perf_counter()
            scored: list[list[tuple[float, dict]]] = []
            if self._reranker:
                pairs = [(q, _doc_text(r)) for q, rows in zip(queries, per_query) for r in rows]
                scores = iter(self._reranker.score(pairs))   # same order as `pairs`
                for rows in per_query:
                    ranked = sorted(((next(scores), r) for r in rows),
                                    key=lambda x: x[0], reverse=True)
                    if self._min_rerank_score is not None:
                        ranked = [x for x in ranked if x[0] >= self._min_rerank_score]
                    scored.append(ranked[: self._k])
            else:
                scored = [[(r["similarity"], r) for r in rows[: self._k]] for rows in per_query]
            t_rerank = perf_counter() - t0

            # 4) merge across queries, keeping each chunk's best score
            best: dict[tuple[str, str], RetrievedChunk] = {}
            for ranked in scored:
                for score, row in ranked:
                    key = (row["file_path"], row["chunkname"])
                    if key not in best or score > best[key].score:
                        best[key] = RetrievedChunk(
                            file_path=row["file_path"], chunkname=row["chunkname"],
                            kind=row["kind"], start_line=row["start_line"],
                            end_line=row["end_line"], content=row["content"],
                            similarity=row["similarity"], score=score,
                        )

            # 5) best-first within the size budget
            selected: list[RetrievedChunk] = []
            used = 0
            for c in sorted(best.values(), key=lambda c: c.score, reverse=True):
                if used + len(c.content) > self._max_chars:
                    continue
                selected.append(c)
                used += len(c.content)

            logger.info(
                "retrieval repo=%s queries=%d candidates=%d selected=%d chars=%d "
                "(%.2fs embed + %.2fs search + %.2fs rerank)",
                repo, len(queries), len(best), len(selected), used,
                t_embed, t_search, t_rerank,
            )
            return selected

def format_context(chunks: list[RetrievedChunk]) -> str:
    """Renders chunks as a prompt-ready block. Empty string if there are none."""
    return "\n\n".join(
        f"### {c.file_path}:{c.start_line}-{c.end_line} - {c.chunkname} ({c.kind})\n"
        f"```python\n{c.content}\n```"
        for c in chunks
    )