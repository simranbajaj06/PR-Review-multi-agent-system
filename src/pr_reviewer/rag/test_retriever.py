"""tests/rag/test_retriever.py - no DB, model or network needed."""
from src.pr_reviewer.rag.ports import ChunkStore, Embedder
from src.pr_reviewer.rag.retriever import ContextRetriever, _parse_diff, format_context

DIFF = '''\
diff --git a/app/a.py b/app/a.py
--- a/app/a.py
+++ b/app/a.py
@@ -15,3 +15,4 @@ def handler(req):
     user = load(req)
-    return render(user)
+    audit_log(user, req.headers)
+    return render(user, strict=True)
@@ -70,2 +71,3 @@ class Cache:
     x = 1
+    self._ttl_seconds = compute_ttl(config.cache_settings)
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1,1 +1,2 @@
+docs only change that must be ignored entirely
diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def gone():
-    return "removed file must be ignored"
'''


class FakeEmbedder(Embedder):
    dim = 3
    def embed_documents(self, texts): return [[0.0] * 3 for _ in texts]
    def embed_query(self, text): return [0.0] * 3


class FakeStore(ChunkStore):
    def __init__(self, rows): self.rows = rows
    def search(self, repo, query_embedding, k=5): return self.rows
    # unused by the retriever
    def get_hashes(self, *a, **k): ...
    def upsert(self, *a, **k): ...
    def update_locations(self, *a, **k): ...
    def delete(self, *a, **k): ...
    def get_indexed_sha(self, *a, **k): ...
    def set_indexed_sha(self, *a, **k): ...


def _row(path, name, start, end, sim, content="x = 1"):
    return dict(file_path=path, chunkname=name, kind="function",
                start_line=start, end_line=end, content=content, similarity=sim)


def test_parse_diff_keeps_only_python_hunks_in_live_files():
    hunks = _parse_diff(DIFF)
    assert [(h.path, h.old_start, h.old_end) for h in hunks] == [
        ("app/a.py", 15, 17), ("app/a.py", 70, 71)]
    assert hunks[0].scope == "def handler(req):"
    assert "audit_log(user, req.headers)" in hunks[0].added[0]


def test_retrieve_filters_dedupes_and_respects_budget():
    rows = [
        _row("app/a.py", "handler", 10, 30, 0.9),            # overlaps hunk 15-17 -> dropped
        _row("app/a.py", "other", 40, 60, 0.8),              # same file, untouched -> kept
        _row("app/b.py", "helper", 1, 5, 0.7),               # kept
        _row("app/c.py", "weak", 1, 5, 0.1),                 # below min_similarity -> dropped
        _row("app/d.py", "huge", 1, 500, 0.6, "y" * 20_000), # exceeds budget -> skipped
    ]
    r = ContextRetriever(FakeEmbedder(), FakeStore(rows), max_context_chars=1_000)
    out = r.retrieve("o/r", DIFF)  # 2 queries -> same rows twice -> must dedupe
    assert [c.chunkname for c in out] == ["other", "helper"]


def test_modified_chunks_can_be_included():
    rows = [_row("app/a.py", "handler", 10, 30, 0.9)]
    r = ContextRetriever(FakeEmbedder(), FakeStore(rows), include_modified_chunks=True)
    assert [c.chunkname for c in r.retrieve("o/r", DIFF)] == ["handler"]


def test_empty_when_no_python_changes_and_format():
    r = ContextRetriever(FakeEmbedder(), FakeStore([]))
    assert r.retrieve("o/r", "diff --git a/x.md b/x.md\n+++ b/x.md\n@@ -1 +1 @@\n+hi\n") == []
    assert format_context([]) == ""