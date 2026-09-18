"""A minimal in-process queue - good enough for one repo / local development.
Swap this for a Redis- or SQS-backed ReviewQueue later without touching
anything in domain/ or api/, since they only ever depend on ReviewQueue.
"""
import queue

from pr_reviewer.domain.interfaces import ReviewQueue
from pr_reviewer.domain.models import PullRequestEvent


class InMemoryReviewQueue(ReviewQueue):
    def __init__(self):
        self._queue: "queue.Queue[PullRequestEvent]" = queue.Queue()

    def enqueue(self, event: PullRequestEvent) -> None:
        self._queue.put(event)
        print(f"[queue] enqueued PR #{event.pr_number} ({event.repo_full_name}): {event.pr_title!r}")

    def dequeue(self) -> PullRequestEvent | None:
        """Used later by a worker process to pull the next event off the queue."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
