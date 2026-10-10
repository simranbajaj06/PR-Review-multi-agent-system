"""Ports (abstract interfaces). Concrete implementations live under adapters/
and are wired in at the composition root (main.py) — this is the Dependency
Inversion piece of SOLID: the domain never imports a concrete adapter.
"""
from abc import ABC, abstractmethod

from src.pr_reviewer.domain.models import PullRequestEvent, QueueJob, Review


class WebhookVerifier(ABC):
    """Confirms an inbound webhook request genuinely came from GitHub."""

    @abstractmethod
    def verify(self, payload_body: bytes, signature_header: str | None) -> bool:
        ...


class ReviewQueue(ABC):
    """Where work items (PR reviews, index jobs) go to be picked up later."""

    @abstractmethod
    def enqueue(self, job: QueueJob) -> None:
        ...

    @abstractmethod
    def dequeue(self) -> QueueJob | None:
        """Returns the next job, or None if the queue is currently empty."""
        ...


class VCSClient(ABC):
    """Fetches a pull request's content and posts results back to the host (GitHub, etc.)."""

    @abstractmethod
    def fetch_diff(self, event: PullRequestEvent) -> str:
        ...

    @abstractmethod
    def post_comment(self, event: PullRequestEvent, body: str) -> None:
        """Posts a general (non-inline) comment on the pull request."""
        ...


class Reviewer(ABC):
    """Produces a review for a PR given its diff."""

    @abstractmethod
    def review(self, event: PullRequestEvent, diff: str) -> Review:
        ...