"""Bring a GitHub repo's index up to date. Cheap to re-run: the first run
indexes everything; later runs only process files changed since last time.

  python -m scripts.index_repo owner/name                       # default branch head
  python -m scripts.index_repo owner/name --ref main
  python -m scripts.index_repo owner/name --query "where is auth checked?"

Needs DATABASE_URL (and GITHUB_TOKEN for private repos) in the environment/.env.
"""
import argparse
import logging
import os

from dotenv import load_dotenv

from src.pr_reviewer.adapters.embeddings.qwen3_embedder import Qwen3Embedder
from src.pr_reviewer.adapters.github.github_repo_fetcher import GitHubSource
from src.pr_reviewer.adapters.storage.pg_chunk_store import PgChunkStore
from src.pr_reviewer.rag.indexer import ChunkIndexer
from src.pr_reviewer.rag.updater import IndexUpdater


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("repo", help="owner/name of the repo to index")
    p.add_argument("--ref", help="branch/tag/sha (default: repo's default branch)")
    p.add_argument("--query", help="after syncing, run a test search")
    args = p.parse_args()

    embedder = Qwen3Embedder()
    if embedder.dim != 1024:
        raise SystemExit(f"model dim {embedder.dim} != vector(1024) column")
    store = PgChunkStore(os.environ["DATABASE_URL"])

    updater = IndexUpdater(GitHubSource(), store, ChunkIndexer(embedder, store))
    sha = updater.sync(args.repo, args.ref)
    logging.info("index is at %s", sha[:7])

    if args.query:
        for r in store.search(args.repo, embedder.embed_query(args.query), k=5):
            print(f"{r['similarity']:.3f}  {r['file_path']}:{r['start_line']}-{r['end_line']}  {r['chunkname']}")


if __name__ == "__main__":
    main()