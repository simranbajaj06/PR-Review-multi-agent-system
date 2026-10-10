"""Turns a GitHub `push` webhook payload into an IndexJob, or explains why
the push isn't worth indexing. Lives next to payload_parser.py
(src/pr_reviewer/adapters/github/push_parser.py).
"""
from src.pr_reviewer.domain.models import IndexJob

_ZERO_SHA = "0" * 40


def parse_push_event(payload: dict) -> tuple[IndexJob | None, str | None]:
    """Returns (job, None) to index, or (None, reason) to ignore."""
    repo = payload.get("repository") or {}
    full_name = repo.get("full_name")
    default_branch = repo.get("default_branch")
    ref = payload.get("ref", "")
    after = payload.get("after", "")

    if not full_name or not default_branch:
        return None, "malformed push payload"
    if payload.get("deleted") or after == _ZERO_SHA:
        return None, "branch deleted"
    if ref != f"refs/heads/{default_branch}":
        # The index tracks the default branch (what PRs are diffed against).
        return None, f"push to '{ref}' is not the default branch"
    if not after:
        return None, "push has no head commit"

    return IndexJob(repo_full_name=full_name, sha=after), None