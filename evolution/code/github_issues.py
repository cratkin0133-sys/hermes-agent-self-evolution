"""Fetch a GitHub issue (title, body, first comments) to use as a Phase 4 bug description."""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

API = "https://api.github.com"


@dataclass
class Issue:
    repo: str          # "owner/name"
    number: int
    title: str
    body: str
    url: str           # html_url
    comments: list[str]


def _get_json(url: str, headers: dict, timeout: int, opener):
    request = urllib.request.Request(url, headers=headers)
    try:
        with opener(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        msg = f"GitHub returned HTTP {e.code} {e.reason} for {url}"
        if e.code == 404:
            msg += ("; the repo may be private or the issue number wrong "
                    "(set GITHUB_TOKEN to access private repos)")
        raise RuntimeError(msg) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach GitHub ({url}): {e.reason}") from e


def fetch_issue(repo: str, number: int, token: Optional[str] = None,
                max_comments: int = 10, timeout: int = 20, opener=None) -> Issue:
    token = token if token is not None else os.environ.get("GITHUB_TOKEN")
    opener = opener or urllib.request.urlopen
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "hermes-agent-self-evolution"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    base = f"{API}/repos/{repo}/issues/{number}"
    data = _get_json(base, headers, timeout, opener)
    if "pull_request" in data:
        raise ValueError(f"#{number} is a pull request, not an issue")
    comments: list[str] = []
    if data.get("comments"):
        raw = _get_json(f"{base}/comments?per_page={max_comments}", headers, timeout, opener)
        comments = [c["body"] for c in raw if c.get("body")]
    return Issue(repo=repo, number=number, title=data.get("title") or "", body=data.get("body") or "",
                 url=data.get("html_url") or "", comments=comments)


def issue_to_bug_text(issue: Issue, max_chars: int = 6000) -> str:
    text = f"{issue.title} ({issue.url})\n\n{issue.body}"
    for c in issue.comments:
        text += "\n\n--- comment ---\n" + c
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[truncated]"
    return text
