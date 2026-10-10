"""Composition root. This is the only file that imports both the abstract
interfaces AND the concrete adapter classes, and wires them together.
Everything else in the app depends on interfaces only.
"""
from dotenv import load_dotenv
load_dotenv()  # must run before GitHubClient/GroqReviewer read os.environ below

import logging
import threading
from dataclasses import dataclass

from fastapi import FastAPI

from src.pr_reviewer.adapters.github.signature_verifier import GitHubSignatureVerifier
from src.pr_reviewer.adapters.queue.in_memory_queue import InMemoryReviewQueue
from src.pr_reviewer.api.webhook_router import router as webhook_router
from src.pr_reviewer.config import Settings, get_settings
from src.pr_reviewer.domain.services.webhook_handler import WebhookHandler
from src.pr_reviewer.adapters.github.github_client import GitHubClient
from src.pr_reviewer.adapters.llm.grok_reviewer import GroqReviewer
from src.pr_reviewer.workers.review_worker import run_worker
from src.pr_reviewer.rag.ports import ChunkStore, Embedder, RepoSource
from src.pr_reviewer.rag.indexer import ChunkIndexer
from src.pr_reviewer.rag.updater import IndexUpdater
from src.pr_reviewer.rag.retriever import ContextRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Must equal the vector(N) column in code_chunks.
EMBEDDING_DIM = 1024


@dataclass(frozen=True)
class RagServices:
    """Everything the RAG layer exposes to the rest of the app. Typed against
    the ports, so consumers (worker, retriever) never see concrete adapters."""
    embedder: Embedder
    store: ChunkStore
    source: RepoSource
    updater: IndexUpdater
    retriever: ContextRetriever


def build_rag(settings: Settings) -> RagServices | None:
    """Builds the RAG services once at startup. Returns None when RAG is not
    configured, so the app still works with diff-only reviews."""
    if not settings.rag_enabled:
        logger.warning("DATABASE_URL not set; RAG disabled, reviews will use the diff only")
        return None

    # Imported lazily: sentence-transformers/torch are heavy, and there's no
    # reason to pay that import cost when RAG is off.
    from src.pr_reviewer.adapters.embeddings.qwen3_embedder import Qwen3Embedder
    from src.pr_reviewer.adapters.github.github_repo_fetcher import GitHubSource
    from src.pr_reviewer.adapters.storage.pg_chunk_store import PgChunkStore

    embedder = Qwen3Embedder()  # loads the model once; reused for the app's lifetime
    if embedder.dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"embedder dim {embedder.dim} != vector({EMBEDDING_DIM}) column in code_chunks"
        )

    store = PgChunkStore(settings.database_url)
    source = GitHubSource()  # reads GITHUB_TOKEN from the environment
    updater = IndexUpdater(source, store, ChunkIndexer(embedder, store))
    import os
    reranker = None
    if os.getenv("RERANKER_ENABLED", "true").lower() == "true":
        from src.pr_reviewer.adapters.reranking.cross_encoder_reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()   # model loads once here
        logger.info("reranker enabled")

    retriever = ContextRetriever(embedder, store, reranker)

    logger.info("RAG enabled (embedding dim=%d)", embedder.dim)
    return RagServices(
        embedder=embedder, store=store, source=source, updater=updater, retriever=retriever
    )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="PR Reviewer Agent")

    verifier = GitHubSignatureVerifier(settings.github_webhook_secret)
    review_queue = InMemoryReviewQueue()

    # RAG services: used by the worker for push indexing now, and by the
    # review graph for retrieval later. None if RAG is disabled.
    rag = build_rag(settings)
    app.state.rag = rag

    app.state.webhook_handler = WebhookHandler(
        verifier, review_queue, index_on_push=rag is not None
    )
    app.state.review_queue = review_queue

    app.include_router(webhook_router)
    vcs_client = GitHubClient()
    reviewer = GroqReviewer(retriever=rag.retriever if rag else None)

    worker_thread = threading.Thread(
        target=run_worker,
        args=(review_queue, vcs_client, reviewer, rag.updater if rag else None),
        daemon=True,
    )
    worker_thread.start()

    @app.get("/health")
    def health():
        return {"status": "ok", "rag_enabled": rag is not None}

    return app

app = create_app()
@app.on_event("shutdown")
def _shutdown():
    rag = app.state.rag
    if rag is not None:
        rag.store.close()