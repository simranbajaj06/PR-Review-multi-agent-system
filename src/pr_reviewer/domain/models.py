"""Framework-agnostic data shapes. No FastAPI or GitHub-specific imports here."""
from dataclasses import dataclass

# The pull_request "action" values worth triggering a review for.
REVIEWABLE_ACTIONS = {"opened", "synchronize", "reopened", "ready_for_review"}


@dataclass(frozen=True)
class PullRequestEvent:
    """The subset of GitHub's payload our app actually needs."""
    action: str
    repo_full_name: str
    pr_number: int
    pr_title: str
    head_sha: str
    head_ref: str
    base_ref: str
    author: str
    diff_url: str

    def is_reviewable(self) -> bool:
        return self.action in REVIEWABLE_ACTIONS


@dataclass(frozen=True)
class Review:                                 
    """The result of a reviewer pass over a PR's diff."""
    pr_number: int
    summary: str
    verdict: str  # "approve" | "comment" | "request_changes" — richer states come in Phase 8