"""Drains the review queue: fetch diff -> get a review -> post it back to the PR."""
import time

from pr_reviewer.domain.interfaces import ReviewQueue, Reviewer, VCSClient


def run_worker(
    review_queue: ReviewQueue,
    vcs_client: VCSClient,
    reviewer: Reviewer,                        # ← new
    poll_interval: float = 1.0,
) -> None:
    print("[worker] started, waiting for PR events...")
    while True:
        event = review_queue.dequeue()
        if event is None:
            time.sleep(poll_interval)
            continue

        print(f"[worker] reviewing PR #{event.pr_number}: {event.pr_title!r} "
              f"({event.repo_full_name})")

        try:
            diff = vcs_client.fetch_diff(event)
        except Exception as exc:
            print(f"[worker] failed to fetch diff for PR #{event.pr_number}: {exc}")
            continue

        try:
            review = reviewer.review(event, diff)
        except Exception as exc:
            print(f"[worker] review failed for PR #{event.pr_number}: {exc}")
            continue

        try:
            vcs_client.post_comment(event, review.summary)
            print(f"[worker] posted review comment on PR #{event.pr_number}")
        except Exception as exc:
            print(f"[worker] failed to post comment for PR #{event.pr_number}: {exc}")