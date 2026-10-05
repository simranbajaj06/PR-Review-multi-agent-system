"""Downloads a snapshot of a GitHub repo (no git needed) into a temp dir so it
can be chunked. Uses the tarball API: one HTTP request instead of a clone.

  with fetch_snapshot("owner/name") as snap:        # default branch
      chunks = chunk_repo(snap.path, snap.repo, snap.commit_sha)

Private repos need GITHUB_TOKEN with read access to repository contents.
"""
import os
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import re
from urllib.parse import quote

import requests

from src.pr_reviewer.rag.ports import FileChange, RepoSource, Snapshot

API = "https://api.github.com"


def _headers(token: str | None, accept: str) -> dict:
    h = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract regular files/dirs only; refuse anything escaping `dest`."""
    dest = dest.resolve()
    for member in tar:
        if not (member.isfile() or member.isdir()):
            continue  # skip symlinks, devices, etc.
        target = (dest / member.name).resolve()
        if dest != target and dest not in target.parents:
            raise ValueError(f"unsafe path in tarball: {member.name}")
        tar.extract(member, dest)


@contextmanager
def fetch_snapshot(
    repo: str, ref: str | None = None, token: str | None = None
) -> Iterator[Snapshot]:
    """Yields a Snapshot of `repo` at `ref` (default: the repo's default
    branch). The temp dir is deleted when the block exits."""
    token = token or os.environ.get("GITHUB_TOKEN")

    if ref is None:
        r = requests.get(f"{API}/repos/{repo}", headers=_headers(token, "application/vnd.github+json"), timeout=15)
        r.raise_for_status()
        ref = r.json()["default_branch"]

    # Resolve to an exact commit so every chunk is tied to a real SHA.
    r = requests.get(f"{API}/repos/{repo}/commits/{ref}",
                     headers=_headers(token, "application/vnd.github.sha"), timeout=15)
    r.raise_for_status()
    commit_sha = r.text.strip()

    tmp = Path(tempfile.mkdtemp(prefix="repo-snap-"))
    try:
        archive = tmp / "repo.tar.gz"
        with requests.get(f"{API}/repos/{repo}/tarball/{commit_sha}",
                          headers=_headers(token, "application/vnd.github+json"),
                          stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(archive, "wb") as f:
                for block in resp.iter_content(1 << 20):
                    f.write(block)

        extract_dir = tmp / "src"
        extract_dir.mkdir()
        with tarfile.open(archive) as tar:
            _safe_extract(tar, extract_dir)

        # GitHub wraps everything in one top-level "owner-repo-<sha>/" folder.
        (root,) = [p for p in extract_dir.iterdir() if p.is_dir()]
        yield Snapshot(repo=repo, ref=ref, commit_sha=commit_sha, path=root)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PAGE = 100
_COMPARE_FILE_CAP = 300  # GitHub truncates compare results beyond this


class GitHubSource(RepoSource):
    def __init__(self, token: str | None = None):
        self._token = token or os.environ.get("GITHUB_TOKEN")

    def resolve_ref(self, repo: str, ref: str | None) -> str:
        if ref and _SHA_RE.match(ref):
            return ref
        if ref is None:
            r = requests.get(f"{API}/repos/{repo}", headers=_headers(self._token, "application/vnd.github+json"), timeout=15)
            r.raise_for_status()
            ref = r.json()["default_branch"]
        r = requests.get(f"{API}/repos/{repo}/commits/{ref}",
                         headers=_headers(self._token, "application/vnd.github.sha"), timeout=15)
        r.raise_for_status()
        return r.text.strip()

    def changed_files(self, repo: str, base: str, head: str) -> list[FileChange] | None:
        files: list[dict] = []
        for page in range(1, _COMPARE_FILE_CAP // _PAGE + 1):
            r = requests.get(
                f"{API}/repos/{repo}/compare/{base}...{head}",
                headers=_headers(self._token, "application/vnd.github+json"),
                params={"per_page": _PAGE, "page": page}, timeout=30,
            )
            if r.status_code in (404, 422):  # base no longer exists (force-push) etc.
                return None
            r.raise_for_status()
            batch = r.json().get("files", [])
            files.extend(batch)
            if len(batch) < _PAGE:
                break
        if len(files) >= _COMPARE_FILE_CAP:
            return None  # possibly truncated: don't trust it

        out = []
        for f in files:
            status = f["status"]
            if status == "renamed":
                out.append(FileChange(f["filename"], "renamed", f.get("previous_filename")))
            elif status == "removed":
                out.append(FileChange(f["filename"], "removed"))
            elif status in ("added", "modified", "changed", "copied"):
                out.append(FileChange(f["filename"], "added" if status == "added" else "modified"))
        return out

    def read_file(self, repo: str, path: str, ref: str) -> str | None:
        r = requests.get(
            f"{API}/repos/{repo}/contents/{quote(path)}",
            headers=_headers(self._token, "application/vnd.github.raw+json"),
            params={"ref": ref}, timeout=30,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text

    def snapshot(self, repo: str, sha: str):
        return fetch_snapshot(repo, sha, self._token)