"""A minimal in-process queue - good enough for one repo / local development.
Swap this for a Redis- or SQS-backed ReviewQueue later without touching
anything in domain/ or api/, since they only ever depend on ReviewQueue.
"""
import queue

from src.pr_reviewer.domain.interfaces import ReviewQueue
from src.pr_reviewer.domain.models import IndexJob, PullRequestEvent, QueueJob


class InMemoryReviewQueue(ReviewQueue):
    def __init__(self):
        self._queue: "queue.Queue[QueueJob]" = queue.Queue()

    def enqueue(self, job: QueueJob) -> None:
        self._queue.put(job)
        if isinstance(job, PullRequestEvent):
            print(f"[queue] enqueued PR #{job.pr_number} ({job.repo_full_name}): {job.pr_title!r}")
        elif isinstance(job, IndexJob):
            print(f"[queue] enqueued index job ({job.repo_full_name} @ {job.sha[:7]})")

    def dequeue(self) -> QueueJob | None:
        """Used by the worker to pull the next job off the queue."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None