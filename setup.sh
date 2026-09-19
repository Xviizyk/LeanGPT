#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LANG_CHOICE=""

pick_language() {
  echo "1) English"
  echo "2) Русский"
  read -rp "> " LANG_CHOICE
  if [[ "$LANG_CHOICE" != "1" && "$LANG_CHOICE" != "2" ]]; then
    LANG_CHOICE="1"
  fi
}

t() {
  local key="$1"
  if [[ "$LANG_CHOICE" == "2" ]]; then
    case "$key" in
      title) echo "🧠 LeanGPT — установщик" ;;
      menu) echo "
1) 📦 Установить зависимости
2) ⚙️  Настроить пути и токены
3) 📚 Скачать корпус Lean4 с GitHub
4) 🔤 Обучить токенизатор
5) 🏋️ Претрейн модели с нуля
6) 🚀 Запустить самообучение (curriculum + GRPO)
7) 📤 Собрать снапшот доказательств
8) 🌐 Опубликовать код в GitHub
9) 🤖 Запушить доказательства ботом
0) 🚪 Выход
" ;;
      choose) echo "Выбери пункт:" ;;
      done) echo "✅ Готово" ;;
      py_missing) echo "❌ Не найден python3. Установи Python 3.11+ и запусти скрипт снова." ;;
      config_saved) echo "✅ Конфиг сохранён в config.local.yaml" ;;
      ask_repl) echo "Путь к бинарнику Lean REPL:" ;;
      ask_project) echo "Путь к твоему lake-проекту (с mathlib и т.п.):" ;;
      ask_token) echo "GitHub token (для сбора корпуса, можно пусто):" ;;
      ask_sources_repo) echo "Ссылка на репозиторий 'Исходники' (git@... или пусто):" ;;
      ask_proofs_repo) echo "Ссылка на репозиторий 'Доказательства' (git@... или пусто):" ;;
      bye) echo "👋 Пока!" ;;
    esac
  else
    case "$key" in
      title) echo "🧠 LeanGPT — setup" ;;
      menu) echo "
1) 📦 Install dependencies
2) ⚙️  Configure paths and tokens
3) 📚 Collect Lean4 corpus from GitHub
4) 🔤 Train tokenizer
5) 🏋️ Pretrain the model from scratch
6) 🚀 Run self-play (curriculum + GRPO)
7) 📤 Build a proof snapshot
8) 🌐 Publish sources to GitHub
9) 🤖 Push proofs via bot
0) 🚪 Exit
" ;;
      choose) echo "Choose an option:" ;;
      done) echo "✅ Done" ;;
      py_missing) echo "❌ python3 not found. Install Python 3.11+ and run this again." ;;
      config_saved) echo "✅ Config saved to config.local.yaml" ;;
      ask_repl) echo "Path to your Lean REPL binary:" ;;
      ask_project) echo "Path to your lake project (with mathlib etc.):" ;;
      ask_token) echo "GitHub token (for corpus collection, can be empty):" ;;
      ask_sources_repo) echo "Sources repo URL (git@... or empty):" ;;
      ask_proofs_repo) echo "Proofs repo URL (git@... or empty):" ;;
      bye) echo "👋 Bye!" ;;
    esac
  fi
}

check_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    t py_missing
    exit 1
  fi
}

install_deps() {
  check_python
  python3 -m pip install -r requirements.txt --break-system-packages 2>/dev/null || python3 -m pip install -r requirements.txt
  t done
}

configure() {
  read -rp "$(t ask_repl) " REPL_BIN_IN
  read -rp "$(t ask_project) " PROJECT_IN
  read -rp "$(t ask_token) " TOKEN_IN
  read -rp "$(t ask_sources_repo) " SOURCES_IN
  read -rp "$(t ask_proofs_repo) " PROOFS_IN

  cat > config.local.yaml <<CFG
repl_bin: ${REPL_BIN_IN}
lean_project_dir: ${PROJECT_IN}
env_import: null
checkpoint_path: data/checkpoints/lean_gpt_latest.pt
tokenizer_dir: data/tokenizer
results_jsonl: data/results.jsonl
github_token: ${TOKEN_IN}
proofs_repo: ${PROOFS_IN}
sources_repo: ${SOURCES_IN}
CFG
  t config_saved
}

collect_corpus() { check_python; python3 scripts/collect_corpus.py; t done; }
train_tokenizer() { check_python; python3 -m leangpt.train_tokenizer; t done; }
pretrain() { check_python; python3 -m leangpt.train; t done; }
run_selfplay() { check_python; python3 scripts/run_pipeline.py; t done; }
export_proofs() { check_python; python3 scripts/export_proofs.py; t done; }
publish_sources() { bash scripts/publish.sh; t done; }
bot_push() { bash scripts/bot_push_proofs.sh; t done; }

main() {
  pick_language
  while true; do
    t title
    t menu
    t choose
    read -rp "> " choice
    case "$choice" in
      1) install_deps ;;
      2) configure ;;
      3) collect_corpus ;;
      4) train_tokenizer ;;
      5) pretrain ;;
      6) run_selfplay ;;
      7) export_proofs ;;
      8) publish_sources ;;
      9) bot_push ;;
      0) t bye; exit 0 ;;
      *) ;;
    esac
  done
}

main
