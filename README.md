# lean4gen — репозиторий "Исходники"

Собственная (не готовая LLM) нейросеть, генерирующая Lean4-код, с
самообучением напрямую по сигналу от Lean REPL.

## Три репозитория

1. **Исходники** (этот) — код пайплайна. Публичный, меняет только
   разработчик. Личные пути (REPL-бинарник, lake-проект, чекпоинты) сюда
   НЕ хардкодятся — берутся из конфига (`lean4gen/config.py`), который
   читается из `config.local.yaml` или переменных окружения. Публикация:
   `./scripts/publish.sh` (переменная `SOURCES_REPO`).

2. **Доказательства** (`lean4gen-proofs`) — публичный, пушит только бот
   (`scripts/bot_push_proofs.sh` + `bot_dump_proofs.py`), без курирования:
   все успешные попытки сваливаются как есть, дубликаты/мусор — это
   нормально, это сырой лог, а не витрина. Переменная `PROOFS_REPO`.

3. **Клиент** (`lean4gen-client`) — приватный, только для тебя: личные
   пути/токены (`config.local.yaml`, `.env`), собранный корпус, чекпоинты,
   `data/results.jsonl`. Не публикуется никуда, живёт только локально
   (или в приватном репозитории, если хочешь бэкапить между машинами).

## Архитектура

**Модель** (`model.py`) — decoder-only трансформер с нуля: RMSNorm, RoPE,
SwiGLU FFN, `scaled_dot_product_attention`, weight tying.

**Токенизатор** (`train_tokenizer.py`) — byte-level BPE + спецтокены под
тактики (`simp`, `ring`, `omega`, ...) и юникод-символы Lean.

**Верификация** (`repl.py`) — `verify(code)` целиком, либо пошагово по
тактикам через `open_goal()`/`run_tactic()` (частичный сигнал).

**Поиск** (`search.py`) — best-first beam search по тактикам.

**Curriculum** (`curriculum.py`) — синтетические леммы 5 уровней сложности
с авто-переходом по success rate.

**Обучение**: `train.py` — self-supervised претрейн (bf16 + `torch.compile`);
`rl_train.py` — GRPO-lite, RL напрямую по награде от REPL.

**Хранилище** (`store.py`) — JSONL + дедуп по хэшу кода + `iter_balanced()`
(кап по сигнатуре тактик и по числу вариантов на одну теорему — разные
пути к одной теореме не считаются дублями, просто не даём одной теореме
занять весь датасет).

**Дашборд** (`dashboard.py`) — своя live-панель на `rich` (без форка
готового LLM-клиента): токены/сек (скользящее окно), текущий уровень
curriculum, текущая теорема, success rate. Подключена в `run_pipeline.py`.

## Установка

```bash
pip install -r requirements.txt
```

Плюс Lean REPL:

```bash
git clone https://github.com/leanprover-community/repl
cd repl && lake build
```

## Конфигурация (личное — в `lean4gen-client`, не здесь)

```yaml
# config.local.yaml
repl_bin: /path/to/repl/.lake/build/bin/repl
lean_project_dir: /path/to/your/lean_project
env_import: null
checkpoint_path: data/checkpoints/lean_gpt_latest.pt
tokenizer_dir: data/tokenizer
results_jsonl: data/results.jsonl
github_token: ghp_xxx
proofs_repo: git@github.com:you/lean4gen-proofs.git
sources_repo: git@github.com:you/lean4gen.git
```

```bash
export LEAN4GEN_CONFIG=/path/to/config.local.yaml
# либо отдельными переменными: REPL_BIN, LEAN_PROJECT_DIR, GITHUB_TOKEN, ...
```

## Порядок запуска

### 1. Сбор корпуса

```bash
python scripts/collect_corpus.py   # использует GITHUB_TOKEN из конфига
```

### 2. Токенизатор

```bash
python -m lean4gen.train_tokenizer
```

### 3. Претрейн

```bash
python -m lean4gen.train
```

### 4. Самообучение (curriculum + GRPO + live-дашборд)

```bash
python scripts/run_pipeline.py
```

Покажет живую панель: уровень, текущую теорему, токены/сек, success rate.
После каждого раунда можно запустить бота, чтобы выгрузить новые
доказательства в публичный репозиторий:

```bash
export PROOFS_REPO=git@github.com:you/lean4gen-proofs.git
./scripts/bot_push_proofs.sh
```

### 5. (опционально) Курированный снапшот доказательств

Если хочешь не сырой поток, а организованную выборку (например, для
документации или демонстрации) — `export_proofs.py` делает выборку с
дедупом/балансировкой (`iter_balanced`), в отличие от бота:

```bash
python scripts/export_proofs.py --out proof_library
```

### 6. Публикация исходников

```bash
export SOURCES_REPO=git@github.com:you/lean4gen.git
./scripts/publish.sh
```

## Структура

```
lean4gen/
  model.py           — LeanGPT: RoPE + RMSNorm + SwiGLU + SDPA
  config.py          — загрузка личных путей/секретов извне репозитория
  dashboard.py        — live TUI-панель (rich): токены/сек, success rate
  repl.py            — Lean REPL: verify() целиком + open_goal/run_tactic пошагово
  search.py          — best-first beam search по тактикам
  curriculum.py       — синтетические леммы нарастающей сложности
  generator.py       — инференс модели, замер токенов/сек
  train.py           — self-supervised претрейн (LM-лосс)
  train_tokenizer.py — BPE-токенизатор + спецтокены
  rl_train.py        — GRPO-lite: RL по награде от REPL
  pipeline.py         — простой цикл generate->verify->save (без RL)
  store.py           — JSONL + дедуп + балансировка
scripts/
  collect_corpus.py    — сбор .lean кода с GitHub
  run_pipeline.py       — curriculum + GRPO + live-дашборд
  export_proofs.py     — курированный снапшот доказательств (не для бота)
  bot_dump_proofs.py   — сырой инкрементальный дамп новых доказательств
  bot_push_proofs.sh   — коммит+пуш дампа в репозиторий "Доказательства"
  publish.sh            — пуш кода в репозиторий "Исходники"
  test_repl_only.py    — тест связки с REPL без модели
```

## Открытые вопросы

- `open_goal`/`run_tactic` в `repl.py` — сверить имена полей ответа с
  актуальной версией Lean REPL (протокол иногда меняется).
- "Одна строка = одна тактика" в `search.py` — упрощение, многострочные
  тактики (`simp only [...]`) потребуют парсинга по `;`/отступам.
- `RLConfig` (KL-коэффициент, `group_size`) — гиперпараметры под подбор.
- Размер модели в `ModelConfig` (~50-100M) — уменьшить, если корпус
  окажется меньше нескольких сотен MB.
