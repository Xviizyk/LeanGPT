#!/usr/bin/env bash

set -uo pipefail
cd "$(dirname "$0")"

CONFIG="config.local.yaml"
LOG_DIR="logs"
TRAIN_LOG="$LOG_DIR/train.log"
TRAIN_PID="$LOG_DIR/train.pid"
LANG_CHOICE="${LEANGPT_LANG:-}"
PY=""
ANS=""
mkdir -p "$LOG_DIR"

trap 'echo' INT

declare -A EN=(
  [title]="🧠 LeanGPT — setup"
  [menu]="
 1) 📦 Install dependencies
 2) ⚙️  Configure paths and tokens
 3) 📚 Collect Lean4 corpus from GitHub
 4) 🔤 Train tokenizer
 5) 🏋️ Pretrain the model from scratch
 6) 🚀 Run self-play (curriculum + GRPO)
 7) 📤 Build a proof snapshot
 8) 🌐 Publish sources to GitHub
 9) 🤖 Push proofs via bot
10) 📊 Training status
11) 📜 Follow training log
12) 🛑 Stop training (saves a checkpoint)
 0) 🚪 Exit
"
  [choose]="Choose an option:"
  [done]="✅ Done"
  [failed]="❌ Step failed, exit code"
  [py_missing]="❌ python3 not found. Install Python 3.11+ and run this again."
  [no_req]="❌ requirements.txt not found"
  [config_saved]="✅ Config saved to config.local.yaml"
  [gitignore_warn]="⚠️  config.local.yaml holds your token: add it to .gitignore"
  [ask_repl]="Path to your Lean REPL binary:"
  [ask_project]="Path to your lake project (with mathlib etc.):"
  [ask_token]="GitHub token (for corpus collection, can be empty):"
  [keep_hint]="(Enter = keep current)"
  [ask_sources_repo]="Sources repo URL (git@... or empty):"
  [ask_proofs_repo]="Proofs repo URL (git@... or empty):"
  [ask_steps]="Training steps:"
  [ask_batch]="Micro-batch size (lower it if you run out of RAM):"
  [ask_accum]="Gradient accumulation steps:"
  [ask_bg]="Run in background, survives terminal/VS Code reconnects? (Y/n):"
  [bad_number]="❌ Enter a positive integer"
  [already_running]="⚠️  Training is already running (option 10 for status, 12 to stop)"
  [started]="🏋️ Training started in background, PID"
  [log_hint]="Log: logs/train.log (option 11 follows it)"
  [running]="Training is running, PID"
  [not_running]="Training is not running"
  [stopping]="Stopping, waiting for the checkpoint to be saved..."
  [stopped]="✅ Stopped, checkpoint saved"
  [still_running]="⚠️  Process is still saving, check option 10 in a minute"
  [no_log]="No log yet"
  [ctrlc]="Ctrl+C — back to the menu"
  [bye]="👋 Bye!"
  [bg_note]="Background training keeps running."
)

declare -A RU=(
  [title]="🧠 LeanGPT — установщик"
  [menu]="
 1) 📦 Установить зависимости
 2) ⚙️  Настроить пути и токены
 3) 📚 Скачать корпус Lean4 с GitHub
 4) 🔤 Обучить токенизатор
 5) 🏋️ Претрейн модели с нуля
 6) 🚀 Запустить самообучение (curriculum + GRPO)
 7) 📤 Собрать снапшот доказательств
 8) 🌐 Опубликовать код в GitHub
 9) 🤖 Запушить доказательства ботом
10) 📊 Статус обучения
11) 📜 Смотреть лог обучения
12) 🛑 Остановить обучение (с сохранением чекпоинта)
 0) 🚪 Выход
"
  [choose]="Выбери пункт:"
  [done]="✅ Готово"
  [failed]="❌ Шаг завершился с ошибкой, код"
  [py_missing]="❌ Не найден python3. Установи Python 3.11+ и запусти скрипт снова."
  [no_req]="❌ Не найден requirements.txt"
  [config_saved]="✅ Конфиг сохранён в config.local.yaml"
  [gitignore_warn]="⚠️  В config.local.yaml лежит токен: добавь файл в .gitignore"
  [ask_repl]="Путь к бинарнику Lean REPL:"
  [ask_project]="Путь к твоему lake-проекту (с mathlib и т.п.):"
  [ask_token]="GitHub token (для сбора корпуса, можно пусто):"
  [keep_hint]="(Enter = оставить текущий)"
  [ask_sources_repo]="Ссылка на репозиторий 'Исходники' (git@... или пусто):"
  [ask_proofs_repo]="Ссылка на репозиторий 'Доказательства' (git@... или пусто):"
  [ask_steps]="Число шагов обучения:"
  [ask_batch]="Микробатч (уменьши, если не хватает ОЗУ):"
  [ask_accum]="Шагов накопления градиента:"
  [ask_bg]="Запустить в фоне (переживёт переподключение терминала/VS Code)? (Y/n):"
  [bad_number]="❌ Введи целое положительное число"
  [already_running]="⚠️  Обучение уже идёт (пункт 10 — статус, 12 — остановить)"
  [started]="🏋️ Обучение запущено в фоне, PID"
  [log_hint]="Лог: logs/train.log (пункт 11 — смотреть вживую)"
  [running]="Обучение идёт, PID"
  [not_running]="Обучение не запущено"
  [stopping]="Останавливаю, жду сохранения чекпоинта..."
  [stopped]="✅ Остановлено, чекпоинт сохранён"
  [still_running]="⚠️  Процесс ещё сохраняется, проверь пункт 10 через минуту"
  [no_log]="Лога пока нет"
  [ctrlc]="Ctrl+C — вернуться в меню"
  [bye]="👋 Пока!"
  [bg_note]="Фоновое обучение продолжает работать."
)

t() {
  if [[ "$LANG_CHOICE" == "2" ]]; then printf '%s\n' "${RU[$1]}"; else printf '%s\n' "${EN[$1]}"; fi
}

pick_language() {
  if [[ "$LANG_CHOICE" == "1" || "$LANG_CHOICE" == "2" ]]; then return; fi
  echo "1) English"
  echo "2) Русский"
  read -rp "> " LANG_CHOICE || LANG_CHOICE="1"
  [[ "$LANG_CHOICE" == "2" ]] || LANG_CHOICE="1"
}

ask() {
  local hint=""
  [[ -n "$2" ]] && hint=" [$2]"
  read -rp "$1$hint " ANS || ANS=""
  ANS="${ANS:-$2}"
}

ask_int() {
  ask "$1" "$2"
  if ! [[ "$ANS" =~ ^[0-9]+$ ]] || (( ANS < 1 )); then
    t bad_number
    return 1
  fi
}

find_python() {
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
  elif [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  elif command -v python >/dev/null 2>&1; then
    PY="python"
  else
    t py_missing
    return 1
  fi
}

run_sh() {
  "$@"
  local rc=$?
  if (( rc == 0 )); then t done; else echo "$(t failed) $rc"; fi
  return 0
}

run_py() {
  find_python || return 0
  run_sh "$PY" "$@"
}

install_deps() {
  find_python || return 0
  if [[ ! -f requirements.txt ]]; then t no_req; return 0; fi
  if [[ -n "${VIRTUAL_ENV:-}" || "$PY" == .venv/* ]]; then
    run_sh "$PY" -m pip install -r requirements.txt
  else
    "$PY" -m pip install -r requirements.txt --break-system-packages 2>/dev/null \
      && t done \
      || run_sh "$PY" -m pip install -r requirements.txt
  fi
}

cfg_get() {
  [[ -f "$CONFIG" ]] || return 0
  sed -n "s/^$1: *//p" "$CONFIG" | head -n1 | sed "s/^null\$//; s/^'\(.*\)'\$/\1/; s/''/'/g"
}

yaml_str() {
  local q="'"
  if [[ -z "$1" ]]; then echo "null"; else printf "'%s'\n" "${1//$q/$q$q}"; fi
}

configure() {
  local repl project token sources proofs cur_token hint=""
  ask "$(t ask_repl)" "$(cfg_get repl_bin)"; repl="$ANS"
  ask "$(t ask_project)" "$(cfg_get lean_project_dir)"; project="$ANS"
  cur_token="$(cfg_get github_token)"
  [[ -n "$cur_token" ]] && hint="$(t keep_hint)"
  read -rsp "$(t ask_token) $hint " token || token=""
  echo
  token="${token:-$cur_token}"
  ask "$(t ask_sources_repo)" "$(cfg_get sources_repo)"; sources="$ANS"
  ask "$(t ask_proofs_repo)" "$(cfg_get proofs_repo)"; proofs="$ANS"

  cat > "$CONFIG" <<CFG
repl_bin: $(yaml_str "$repl")
lean_project_dir: $(yaml_str "$project")
env_import: null
checkpoint_path: data/checkpoints/lean_gpt_latest.pt
tokenizer_dir: data/tokenizer
results_jsonl: data/results.jsonl
github_token: $(yaml_str "$token")
proofs_repo: $(yaml_str "$proofs")
sources_repo: $(yaml_str "$sources")
CFG
  chmod 600 "$CONFIG"
  t config_saved
  grep -qxF "$CONFIG" .gitignore 2>/dev/null || t gitignore_warn
}

train_running() {
  [[ -f "$TRAIN_PID" ]] && kill -0 "$(cat "$TRAIN_PID")" 2>/dev/null
}

pretrain() {
  find_python || return 0
  if train_running; then t already_running; return 0; fi
  local steps batch accum bg
  ask_int "$(t ask_steps)" 5000 || return 0; steps="$ANS"
  ask_int "$(t ask_batch)" 4 || return 0; batch="$ANS"
  ask_int "$(t ask_accum)" 4 || return 0; accum="$ANS"
  ask "$(t ask_bg)" "Y"; bg="$ANS"

  local args=(--max-steps "$steps" --batch-size "$batch" --grad-accum "$accum")
  if [[ "$bg" =~ ^[YyДд] ]]; then
    nohup "$PY" -u -m leangpt.train "${args[@]}" >> "$TRAIN_LOG" 2>&1 &
    echo $! > "$TRAIN_PID"
    disown
    echo "$(t started) $(cat "$TRAIN_PID")"
    t log_hint
  else
    run_sh "$PY" -u -m leangpt.train "${args[@]}"
  fi
}

show_status() {
  if train_running; then echo "▶ $(t running) $(cat "$TRAIN_PID")"; else echo "⏹ $(t not_running)"; fi
  free -h 2>/dev/null | sed -n '1,2p'
  if [[ -f "$TRAIN_LOG" ]]; then echo "---"; tail -n 8 "$TRAIN_LOG"; fi
  if [[ -d data/checkpoints ]]; then echo "---"; ls -lh data/checkpoints; fi
}

follow_log() {
  if [[ ! -f "$TRAIN_LOG" ]]; then t no_log; return 0; fi
  t ctrlc
  tail -n 20 -f "$TRAIN_LOG"
}

stop_training() {
  if ! train_running; then t not_running; rm -f "$TRAIN_PID"; return 0; fi
  local pid; pid="$(cat "$TRAIN_PID")"
  kill -TERM "$pid"
  t stopping
  local i
  for i in $(seq 1 90); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then t still_running; else rm -f "$TRAIN_PID"; t stopped; fi
}

main() {
  pick_language
  local choice
  while true; do
    t title
    t menu
    t choose
    read -rp "> " choice || break
    case "$choice" in
      1) install_deps ;;
      2) configure ;;
      3) run_py scripts/collect_corpus.py ;;
      4) run_py -m leangpt.train_tokenizer ;;
      5) pretrain ;;
      6) run_py scripts/run_pipeline.py ;;
      7) run_py scripts/export_proofs.py ;;
      8) run_sh bash scripts/publish.sh ;;
      9) run_sh bash scripts/bot_push_proofs.sh ;;
      10) show_status ;;
      11) follow_log ;;
      12) stop_training ;;
      0) train_running && t bg_note; t bye; exit 0 ;;
      *) ;;
    esac
  done
}

main