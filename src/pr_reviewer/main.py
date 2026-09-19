"""Composition root. This is the only file that imports both the abstract
interfaces AND the concrete adapter classes, and wires them together.
Everything else in the app depends on interfaces only.
"""
from dotenv import load_dotenv
load_dotenv()  # must run before GitHubClient/GroqReviewer read os.environ below

from fastapi import FastAPI

from pr_reviewer.adapters.github.signature_verifier import GitHubSignatureVerifier
from pr_reviewer.adapters.queue.in_memory_queue import InMemoryReviewQueue
from pr_reviewer.api.webhook_router import router as webhook_router
from pr_reviewer.config import get_settings
from pr_reviewer.domain.services.webhook_handler import WebhookHandler
import threading
from pr_reviewer.adapters.github.github_client import GitHubClient
from pr_reviewer.adapters.llm.grok_reviewer import GroqReviewer
from pr_reviewer.workers.review_worker import run_worker
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="PR Reviewer Agent")

    verifier = GitHubSignatureVerifier(settings.github_webhook_secret)
    review_queue = InMemoryReviewQueue()

    app.state.webhook_handler = WebhookHandler(verifier, review_queue)
    app.state.review_queue = review_queue  # a worker will read from this later

    app.include_router(webhook_router)
    vcs_client = GitHubClient()
    reviewer = GroqReviewer()

    worker_thread = threading.Thread(
        target=run_worker, args=(review_queue, vcs_client, reviewer), daemon=True
    )
    worker_thread.start()
    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()