#!/usr/bin/env bash
# Публикация репозитория "Исходники" — код пайплайна, меняет только
# разработчик (не бот). Доказательства пушатся отдельным ботом
# (bot_push_proofs.sh) в отдельный репозиторий.
#
# Использование:
#   export SOURCES_REPO=git@github.com:you/lean4gen.git
#   ./scripts/publish.sh

set -euo pipefail

if [[ -z "${SOURCES_REPO:-}" ]]; then
  echo "Укажи SOURCES_REPO (например git@github.com:you/lean4gen.git)"; exit 1
fi

if [[ ! -d .git ]]; then
  git init
  git remote add origin "$SOURCES_REPO"
fi

git add lean4gen/ scripts/ requirements.txt README.md .gitignore
git commit -m "update sources ($(date -u +%Y-%m-%dT%H:%M:%SZ))" || echo "нечего коммитить"
git branch -M main
git push -u origin main
