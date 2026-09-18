"""Turns GitHub's raw pull_request webhook JSON into our domain model."""
from typing import Any

from pr_reviewer.domain.models import PullRequestEvent


def parse_pull_request_event(payload: dict[str, Any]) -> PullRequestEvent:
    pr = payload["pull_request"]
    return PullRequestEvent(
        action=payload["action"],
        repo_full_name=payload["repository"]["full_name"],
        pr_number=pr["number"],
        pr_title=pr["title"],
        head_sha=pr["head"]["sha"],
        head_ref=pr["head"]["ref"],
        base_ref=pr["base"]["ref"],
        author=pr["user"]["login"],
        diff_url=pr["diff_url"],
    )
