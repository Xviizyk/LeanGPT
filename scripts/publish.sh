#!/usr/bin/env bash
set -euo pipefail

read_config_value() {
  local key="$1"
  if [[ -f config.local.yaml ]]; then
    grep "^${key}:" config.local.yaml | sed "s/^${key}: *//" | tr -d '\r'
  fi
}

SOURCES_REPO="${SOURCES_REPO:-$(read_config_value sources_repo)}"

if [[ -z "${SOURCES_REPO:-}" ]]; then
  echo "Set SOURCES_REPO (env var or sources_repo in config.local.yaml)"; exit 1
fi

if [[ ! -d .git ]]; then
  git init
  git remote add origin "$SOURCES_REPO"
fi

git add leangpt/ scripts/ tests/ .github/ requirements.txt requirements-dev.txt README.md LICENSE .gitignore
git commit -m "update sources ($(date -u +%Y-%m-%dT%H:%M:%SZ))" || true
git branch -M main
git push -u origin main
