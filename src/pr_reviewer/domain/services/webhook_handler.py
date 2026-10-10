"""Core logic: verify -> dedupe -> dispatch by event type -> parse -> filter -> enqueue.
Depends only on the abstract ports (WebhookVerifier, ReviewQueue) - never on
FastAPI or a concrete GitHub adapter class. That's what makes it unit-testable
with fakes and swappable without changes here.
"""
from src.pr_reviewer.adapters.github.payload_parser import parse_pull_request_event
from src.pr_reviewer.adapters.github.push_parser import parse_push_event
from src.pr_reviewer.domain.interfaces import ReviewQueue, WebhookVerifier


class WebhookHandler:
    class InvalidSignature(Exception):
        pass

    def __init__(
        self,
        verifier: WebhookVerifier,
        review_queue: ReviewQueue,
        index_on_push: bool = False,
    ):
        self._verifier = verifier
        self._queue = review_queue
        # When False (RAG disabled) push events are acknowledged but ignored.
        self._index_on_push = index_on_push
        # MVP idempotency guard against GitHub's at-least-once delivery retries.
        # In-memory + unbounded: fine for one process, but it resets on restart
        # and grows forever. Swap for a Redis SETNX with a TTL once you move
        # the queue off-process (same spot you'll swap InMemoryReviewQueue).
        self._seen_deliveries: set[str] = set()

    def handle(
        self,
        payload_body: bytes,
        signature_header: str | None,
        payload: dict,
        delivery_id: str | None = None,
        event_type: str = "pull_request",
    ) -> dict:
        if not self._verifier.verify(payload_body, signature_header):
            raise WebhookHandler.InvalidSignature("signature verification failed")

        if delivery_id:
            if delivery_id in self._seen_deliveries:
                return {"status": "ignored", "reason": "duplicate delivery"}
            self._seen_deliveries.add(delivery_id)

        if event_type == "pull_request":
            return self._handle_pull_request(payload)
        if event_type == "push":
            return self._handle_push(payload)
        return {"status": "ignored", "reason": f"event type '{event_type}' not handled"}

    def _handle_pull_request(self, payload: dict) -> dict:
        if "pull_request" not in payload or "action" not in payload:
            return {"status": "ignored", "reason": "not a pull_request event"}

        event = parse_pull_request_event(payload)

        if not event.is_reviewable():
            return {"status": "ignored", "reason": f"action '{event.action}' not reviewable"}

        self._queue.enqueue(event)
        return {"status": "accepted", "pr_number": event.pr_number}

    def _handle_push(self, payload: dict) -> dict:
        if not self._index_on_push:
            return {"status": "ignored", "reason": "indexing disabled"}

        job, reason = parse_push_event(payload)
        if job is None:
            return {"status": "ignored", "reason": reason}

        self._queue.enqueue(job)
        return {"status": "accepted", "index_sha": job.sha}