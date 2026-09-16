#!/usr/bin/env python3
from __future__ import annotations
import io
import os
import tarfile
import time
from pathlib import Path
import requests

GITHUB_API = "https://api.github.com"


def search_lean_repos(token: str, max_repos: int = 200) -> list[str]:
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    repos: list[str] = []
    page = 1
    while len(repos) < max_repos:
        resp = requests.get(
            f"{GITHUB_API}/search/repositories",
            headers=headers,
            params={
                "q": "language:Lean",
                "sort": "stars",
                "order": "desc",
                "per_page": 100,
                "page": page,
            },
            timeout=30,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            break
        repos.extend((item["full_name"] for item in items))
        page += 1
        time.sleep(1)
    return repos[:max_repos]


def download_and_extract(owner_repo: str, out_dir: Path, token: str) -> int:
    owner, repo = owner_repo.split("/")
    url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/main"
    headers = {"Authorization": f"token {token}"}
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code != 200:
        url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/master"
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code != 200:
            return 0
    count = 0
    target_dir = out_dir / owner_repo.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".lean"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                rel_name = member.name.replace("/", "__")
                (target_dir / rel_name).write_bytes(f.read())
                count += 1
    return count


def build_corpus(
    token: str, out_dir: str = "data/lean_corpus", max_repos: int = 200
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    repos = search_lean_repos(token, max_repos=max_repos)
    print(f"Найдено репозиториев: {len(repos)}")
    total_files = 0
    for i, repo in enumerate(repos, 1):
        try:
            n = download_and_extract(repo, out_path, token)
            total_files += n
            print(f"[{i}/{len(repos)}] {repo}: {n} .lean файлов")
        except Exception as e:
            print(f"[{i}/{len(repos)}] {repo}: ошибка {e}")
        time.sleep(1)
    print(f"\nВсего собрано .lean файлов: {total_files}")


if __name__ == "__main__":
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Установи переменную окружения GITHUB_TOKEN")
    build_corpus(token)
