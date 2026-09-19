#!/usr/bin/env bash
set -euo pipefail

read_config_value() {
  local key="$1"
  if [[ -f config.local.yaml ]]; then
    grep "^${key}:" config.local.yaml | sed "s/^${key}: *//" | tr -d '\r'
  fi
}

PROOFS_REPO="${PROOFS_REPO:-$(read_config_value proofs_repo)}"

if [[ -z "${PROOFS_REPO:-}" ]]; then
  echo "Set PROOFS_REPO (env var or proofs_repo in config.local.yaml)"; exit 1
fi

WORKDIR="proofs_dump_repo"

if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone "$PROOFS_REPO" "$WORKDIR" || (mkdir -p "$WORKDIR" && cd "$WORKDIR" && git init && git remote add origin "$PROOFS_REPO")
fi

python3 scripts/bot_dump_proofs.py

cd "$WORKDIR"
git add .
git commit -m "bot: dump $(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
git branch -M main
git push -u origin main
