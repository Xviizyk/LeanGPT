#!/usr/bin/env bash
# Бот для репозитория "Доказательства" — публичный, пуш без курирования.
# Использование:
#   export PROOFS_REPO=git@github.com:you/lean4gen-proofs.git
#   ./scripts/bot_push_proofs.sh

set -euo pipefail

if [[ -z "${PROOFS_REPO:-}" ]]; then
  echo "Укажи PROOFS_REPO"; exit 1
fi

WORKDIR="proofs_dump_repo"

if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone "$PROOFS_REPO" "$WORKDIR" || (mkdir -p "$WORKDIR" && cd "$WORKDIR" && git init && git remote add origin "$PROOFS_REPO")
fi

python3 scripts/bot_dump_proofs.py

cd "$WORKDIR"
git add .
git commit -m "bot: dump $(date -u +%Y-%m-%dT%H:%M:%SZ)" || echo "нечего коммитить"
git branch -M main
git push -u origin main
