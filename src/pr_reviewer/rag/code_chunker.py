"""Splits a Python codebase into function / method / class chunks, shaped to
match the `code_chunks` table. No DB, embedding, or GitHub dependencies - it
takes source text in and returns plain CodeChunk objects, so it stays
unit-testable and the embedder / PgChunkStore can be built separately.

Chunking rules
- function : top-level `def` / `async def`
- method   : `def` directly inside a class (chunkname "Class.method")
- class    : the class "header" only (decorators, signature, docstring and
             class-level attributes, i.e. everything before the first method).
             Method bodies are NOT repeated here; they are their own chunks.
- Nested functions stay inside their parent chunk.
- Module-level code (imports, constants) is not chunked; the table's `kind`
  has no "module" value.
- Oversized chunks are split into parts: "chunkname", "chunkname#part2", ...
- Duplicate chunknames in one file (overloads, if/else redefinitions) get a
  "#L<line>" suffix so UNIQUE (repo, file_path, chunkname) never trips the
  upsert.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ~100 lines of typical code. Bigger chunks are split on line boundaries.
MAX_CHUNK_CHARS = 6_000

SKIP_DIRS = {
    ".git", ".hg", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "build", "dist",
    "site-packages", ".idea", ".vscode",
}
MAX_FILE_BYTES = 500_000  # skip generated / vendored giant files


@dataclass(frozen=True)
class CodeChunk:
    repo: str
    file_path: str      # relative to repo root, forward slashes
    chunkname: str
    kind: str           # "function" | "method" | "class"
    start_line: int     # 1-based, inclusive
    end_line: int
    content: str
    content_hash: str   # sha256(content)
    commit_sha: str

    def to_embedding_text(self) -> str:
        """Text to feed the embedder. A small header makes retrieval work
        better (a query like "where is the signature verified" can match on
        the path or name, not only the body). Store `content` raw; embed this."""
        return f"# {self.file_path} :: {self.chunkname} ({self.kind})\n{self.content}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _node_start(node: ast.AST) -> int:
    """First line including decorators (ast's lineno points at `def`)."""
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def _split_oversized(content: str, start_line: int) -> list[tuple[str, int, int]]:
    """Returns [(text, start_line, end_line)] with each text <= MAX_CHUNK_CHARS
    (a single line longer than the limit is kept whole)."""
    if len(content) <= MAX_CHUNK_CHARS:
        return [(content, start_line, start_line + content.count("\n"))]

    parts: list[tuple[str, int, int]] = []
    buf: list[str] = []
    size = 0
    buf_start = start_line
    for line in content.splitlines():
        if buf and size + len(line) + 1 > MAX_CHUNK_CHARS:
            parts.append(("\n".join(buf), buf_start, buf_start + len(buf) - 1))
            buf_start += len(buf)
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        parts.append(("\n".join(buf), buf_start, buf_start + len(buf) - 1))
    return parts


def _walk(
    body: list[ast.stmt], prefix: str, in_class: bool
) -> Iterator[tuple[str, str, int, int]]:
    """Yields (chunkname, kind, start_line, end_line) for each chunk-worthy node."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield (
                f"{prefix}{node.name}",
                "method" if in_class else "function",
                _node_start(node),
                node.end_lineno,
            )
        elif isinstance(node, ast.ClassDef):
            qual = f"{prefix}{node.name}"
            first_method = next(
                (n for n in node.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
                None,
            )
            header_end = (
                _node_start(first_method) - 1 if first_method else node.end_lineno
            )
            yield qual, "class", _node_start(node), header_end
            yield from _walk(node.body, f"{qual}.", in_class=True)


def chunk_python_source(
    source: str, file_path: str, repo: str, commit_sha: str
) -> list[CodeChunk]:
    """Chunks one Python file. Returns [] (and logs) on a syntax error rather
    than raising - one broken file must not abort a whole repo index."""
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        logger.warning("skipping %s: syntax error: %s", file_path, exc)
        return []

    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    seen: set[str] = set()

    for chunkname, kind, start, end in _walk(tree.body, "", in_class=False):
        if end < start:
            continue
        content = "\n".join(lines[start - 1:end])
        if not content.strip():
            continue

        # Unique-ify duplicates (overloads, conditional redefinitions).
        unique_name = chunkname if chunkname not in seen else f"{chunkname}#L{start}"
        seen.add(unique_name)

        for i, (text, p_start, p_end) in enumerate(_split_oversized(content, start), 1):
            name = unique_name if i == 1 else f"{unique_name}#part{i}"
            chunks.append(
                CodeChunk(
                    repo=repo,
                    file_path=file_path,
                    chunkname=name,
                    kind=kind,
                    start_line=p_start,
                    end_line=p_end,
                    content=text,
                    content_hash=_sha256(text),
                    commit_sha=commit_sha,
                )
            )
    return chunks


def iter_python_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                path = Path(dirpath) / name
                if path.stat().st_size <= MAX_FILE_BYTES:
                    yield path


def chunk_repo(root: str | Path, repo: str, commit_sha: str) -> list[CodeChunk]:
    """Chunks every .py file under `root` (a checked-out working tree).
    `repo` is the "owner/name" string, same as PullRequestEvent.repo_full_name."""
    root = Path(root)
    chunks: list[CodeChunk] = []
    for path in iter_python_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            logger.warning("skipping %s: %s", rel, exc)
            continue
        chunks.extend(chunk_python_source(source, rel, repo, commit_sha))
    logger.info("chunked repo=%s commit=%s chunks=%d", repo, commit_sha[:7], len(chunks))
    return chunks


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    for c in chunk_repo(target, repo="PR-Review-multi-agent-system", commit_sha="0" * 40):
        print(f"{c.file_path}:{c.start_line}-{c.end_line:<4} {c.kind:<8} {c.chunkname}")