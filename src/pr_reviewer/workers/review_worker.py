"""Drains the queue and dispatches by job type:
  PullRequestEvent -> fetch diff -> get a review -> post it back to the PR
  IndexJob         -> bring the RAG index up to the pushed commit
"""
import time

from src.pr_reviewer.domain.interfaces import ReviewQueue, Reviewer, VCSClient
from src.pr_reviewer.domain.models import IndexJob, PullRequestEvent
from src.pr_reviewer.rag.updater import IndexUpdater

import hashlib
import threading
from collections import OrderedDict


class _RecentReviews:
    """Remembers the last N (repo, PR, diff) combos that were reviewed AND posted.
    In-memory: cleared on restart, which is fine for webhook retries."""

    def __init__(self, max_items: int = 256):
        self._items: OrderedDict[str, None] = OrderedDict()
        self._max = max_items
        self._lock = threading.Lock()

    @staticmethod
    def key(repo: str, pr_number: int, diff: str) -> str:
        digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        return f"{repo}#{pr_number}:{digest}"

    def seen(self, key: str) -> bool:
        with self._lock:
            return key in self._items

    def add(self, key: str) -> None:
        with self._lock:
            self._items[key] = None
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)


_recent = _RecentReviews()
def _handle_review(event, vcs_client, reviewer) -> None:
    print(f"[worker] reviewing PR #{event.pr_number}: ...")

    try:
        diff = vcs_client.fetch_diff(event)
    except Exception as exc:
        print(f"[worker] failed to fetch diff ...: {exc}")
        return

    cache_key = _RecentReviews.key(event.repo_full_name, event.pr_number, diff)
    if _recent.seen(cache_key):
        print(f"[worker] PR #{event.pr_number}: diff unchanged since last review, skipping")
        return

    try:
        review = reviewer.review(event, diff)
    except Exception as exc:
        print(f"[worker] review failed ...: {exc}")
        return

    try:
        vcs_client.post_comment(event, review.summary)
        _recent.add(cache_key)          # only after a successful post
        print(f"[worker] posted review comment on PR #{event.pr_number}")
    except Exception as exc:
        print(f"[worker] failed to post comment ...: {exc}")


def _handle_index(job: IndexJob, updater: IndexUpdater | None) -> None:
    if updater is None:
        print(f"[worker] index job for {job.repo_full_name} skipped: RAG disabled")
        return

    print(f"[worker] indexing {job.repo_full_name} @ {job.sha[:7]}")
    try:
        # sync() is incremental: it diffs the last indexed commit against
        # job.sha and re-embeds only changed chunks. A failed run doesn't
        # advance index state, so the next push retries from the same base.
        updater.sync(job.repo_full_name, job.sha)
        print(f"[worker] index up to date for {job.repo_full_name} @ {job.sha[:7]}")
    except Exception as exc:
        print(f"[worker] indexing failed for {job.repo_full_name}: {exc}")


def run_worker(
    review_queue: ReviewQueue,
    vcs_client: VCSClient,
    reviewer: Reviewer,
    updater: IndexUpdater | None = None,
    poll_interval: float = 1.0,
) -> None:
    print("[worker] started, waiting for jobs...")
    while True:
        job = review_queue.dequeue()
        if job is None:
            time.sleep(poll_interval)
            continue

        if isinstance(job, PullRequestEvent):
            _handle_review(job, vcs_client, reviewer)
        elif isinstance(job, IndexJob):
            _handle_index(job, updater)
        else:
            print(f"[worker] ignoring unknown job type: {type(job).__name__}")