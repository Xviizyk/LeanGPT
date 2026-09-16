#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SOURCES_REPO:-}" ]]; then
  echo "Set SOURCES_REPO"; exit 1
fi

if [[ ! -d .git ]]; then
  git init
  git remote add origin "$SOURCES_REPO"
fi

git add lean4gen/ scripts/ tests/ .github/ requirements.txt requirements-dev.txt README.md LICENSE .gitignore
git commit -m "update sources ($(date -u +%Y-%m-%dT%H:%M:%SZ))" || true
git branch -M main
git push -u origin main
