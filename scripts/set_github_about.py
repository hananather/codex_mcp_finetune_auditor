from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass

import requests

DEFAULT_DESCRIPTION = (
    "Interpretability-augmented auditing agent for detecting adversarial fine-tunes (SAE diffing + MCP tools)."
)
DEFAULT_TOPICS = [
    "mcp",
    "model-context-protocol",
    "llm-safety",
    "auditing",
    "adversarial-finetuning",
    "interpretability",
    "mechanistic-interpretability",
    "sparse-autoencoder",
    "sae",
]


@dataclass(frozen=True)
class RepoSlug:
    owner: str
    name: str

    @property
    def full(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def repo_url(self) -> str:
        return f"https://github.com/{self.full}"


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _parse_slug(repo: str) -> RepoSlug:
    repo = repo.strip().removesuffix(".git")
    if repo.count("/") != 1:
        raise ValueError(f"Invalid repo slug: {repo!r} (expected 'owner/name')")
    owner, name = repo.split("/", 1)
    if not owner or not name:
        raise ValueError(f"Invalid repo slug: {repo!r} (expected 'owner/name')")
    return RepoSlug(owner=owner, name=name)


def _slug_from_remote(remote_url: str) -> RepoSlug:
    remote_url = remote_url.strip()

    m = re.match(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\\.git)?$", remote_url)
    if m:
        return RepoSlug(owner=m.group("owner"), name=m.group("repo"))

    m = re.match(r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\\.git)?$", remote_url)
    if m:
        return RepoSlug(owner=m.group("owner"), name=m.group("repo"))

    raise ValueError(f"Unsupported git remote URL format: {remote_url!r}")


def _default_homepage(slug: RepoSlug) -> str:
    return f"{slug.repo_url}#readme"


def _gh_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "codex-mcp-finetune-auditor/set_github_about.py",
    }


def _api_patch_repo(slug: RepoSlug, token: str, description: str, homepage: str) -> None:
    url = f"https://api.github.com/repos/{slug.full}"
    body = {"description": description, "homepage": homepage}
    r = requests.patch(url, headers=_gh_headers(token), json=body, timeout=30)
    if r.status_code >= 400:
        raise SystemExit(f"GitHub API PATCH failed ({r.status_code}): {r.text}")


def _api_put_topics(slug: RepoSlug, token: str, topics: list[str]) -> None:
    url = f"https://api.github.com/repos/{slug.full}/topics"
    body = {"names": topics}
    r = requests.put(url, headers=_gh_headers(token), json=body, timeout=30)
    if r.status_code >= 400:
        raise SystemExit(f"GitHub API topics update failed ({r.status_code}): {r.text}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Set GitHub repo About metadata (description/homepage/topics).")
    p.add_argument(
        "--repo",
        help="Repo slug like 'owner/name' (default: derived from git remote 'origin').",
        default=None,
    )
    p.add_argument("--description", default=DEFAULT_DESCRIPTION)
    p.add_argument("--homepage", default=None)
    p.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help="Space-separated topics (default: built-in suggestions).",
    )
    p.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Env var to read the GitHub token from (default: GITHUB_TOKEN).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would be changed without calling the API.")
    args = p.parse_args(argv)

    if args.repo:
        slug = _parse_slug(args.repo)
    else:
        remote_url = _run(["git", "remote", "get-url", "origin"])
        slug = _slug_from_remote(remote_url)

    token = os.getenv(args.token_env, "").strip()
    if not token:
        print(f"Missing token: set {args.token_env} to a GitHub token with repo edit permission.", file=sys.stderr)
        print(f"Repo: {slug.full}", file=sys.stderr)
        return 2

    homepage = args.homepage or _default_homepage(slug)
    topics = args.topics if args.topics is not None else DEFAULT_TOPICS

    if args.dry_run:
        print(f"Repo: {slug.full}")
        print(f"Description: {args.description}")
        print(f"Homepage: {homepage}")
        print(f"Topics: {topics}")
        return 0

    _api_patch_repo(slug, token, description=args.description, homepage=homepage)
    _api_put_topics(slug, token, topics=topics)
    print(f"Updated GitHub About metadata for {slug.full}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

