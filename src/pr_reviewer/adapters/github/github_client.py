"""Concrete VCSClient for GitHub - fetches the raw diff and posts comments back."""
import os
import requests
from pr_reviewer.domain.interfaces import VCSClient
from pr_reviewer.domain.models import PullRequestEvent


class GitHubClient(VCSClient):
    def __init__(self, token: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN")
        if not self._token:
            # Posting a comment always requires auth, even on a public repo,
            # so a missing token surfaces as a confusing 401 mid-review
            # instead of here, at startup, where it's obvious what's wrong.
            raise ValueError(
                "GITHUB_TOKEN is not set. GitHubClient needs it to post "
                "comments (and to reliably fetch diffs without hitting the "
                "anonymous rate limit). Set it in your environment or .env file."
            )

    def fetch_diff(self, event: PullRequestEvent) -> str:
        # Uses api.github.com directly, NOT event.diff_url - that field
        # redirects to a different host (patch-diff.githubusercontent.com),
        # and requests strips the Authorization header on cross-domain
        # redirects, which would silently break auth for private repos.
        url = f"https://api.github.com/repos/{event.repo_full_name}/pulls/{event.pr_number}"
        headers = {"Accept": "application/vnd.github.v3.diff"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text

    def post_comment(self, event: PullRequestEvent, body: str) -> None:      # ← new
        # PRs are issues under the hood on GitHub's API, so the issue-comments
        # endpoint works fine for a general (non-inline) comment. Line-level
        # comments come later via the /pulls/{n}/reviews endpoint, once the
        # Finding contract carries file/line info (Phase 8 in the roadmap).
        url = f"https://api.github.com/repos/{event.repo_full_name}/issues/{event.pr_number}/comments"
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        response = requests.post(url, headers=headers, json={"body": body}, timeout=15)
        response.raise_for_status()