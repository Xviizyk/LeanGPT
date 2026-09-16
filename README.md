# LeanGPT

A neural net that writes Lean4 proofs, checks them against the real Lean compiler, and only keeps what actually compiles. No pretrained LLM under the hood — the model (LeanGPT) is trained from scratch on a scraped Lean4 corpus, then pushed forward with self-play against the Lean REPL.

Three repos make up the whole thing:

- **lean4gen** (this one) — the code. Public, hand-edited only.
- **lean4gen-proofs** — a raw, unfiltered dump of everything the bot has proven. Public, bot-pushed, messy on purpose.
- **lean4gen-client** — your personal paths, tokens, corpus, and checkpoints. Private, never leaves your machine unless you want it to.

## Quick start

If you don't want to touch a terminal command by hand, just run the setup wizard:

```
./setup.sh        # Linux / macOS
setup.bat         # Windows
```

It walks you through installing dependencies, filling in your paths/tokens, collecting a corpus, training, running self-play, and publishing — pick a language, pick a number, done.

If you'd rather run things manually, here's the full path:

```bash
pip install -r requirements.txt
```

You'll also need the Lean REPL, built separately:

```bash
git clone https://github.com/leanprover-community/repl
cd repl && lake build
```

Point the project at your machine-specific paths and secrets with a `config.local.yaml` (see `lean4gen-client/config.local.yaml.example`), or export the equivalent environment variables:

```bash
export LEAN4GEN_CONFIG=/path/to/config.local.yaml
```

Then:

```bash
python scripts/collect_corpus.py
python -m lean4gen.train_tokenizer
python -m lean4gen.train
python scripts/run_pipeline.py
```

The last command runs the actual self-play loop — a curriculum of increasingly hard lemmas, GRPO-style reinforcement from the REPL's pass/fail signal, and a live dashboard showing tokens/sec and success rate as it goes.

## How it's built

- **Model** — a small decoder-only transformer built from scratch: RoPE, RMSNorm, SwiGLU, `scaled_dot_product_attention`, KV-cache for generation.
- **Tokenizer** — byte-level BPE with tactic and Lean-symbol tokens baked in as special tokens.
- **REPL bridge** — talks to a long-lived Lean REPL process, either checking a whole proof at once or stepping through tactics one at a time to get partial credit.
- **Curriculum** — synthetic lemmas from trivial to mildly interesting, with a held-out eval set the model never trains on, deciding when to move up a level.
- **Search** — best-first beam search over tactic sequences instead of generating a whole proof in one shot.
- **Training** — a normal pretraining pass on the corpus, then GRPO-lite: sample a group of candidates per statement, reward from the REPL, normalize within the group, push the policy.
- **Storage** — a JSONL log with deduping (exact repeats only — different proofs of the same theorem both survive) and capping so one trivial lemma can't flood the dataset.

## Publishing

```bash
export SOURCES_REPO=git@github.com:you/lean4gen.git
./scripts/publish.sh
```

```bash
export PROOFS_REPO=git@github.com:you/lean4gen-proofs.git
./scripts/bot_push_proofs.sh
```

The bot script just dumps whatever's new since the last run — no curation, no dedup. If you want a clean, organized snapshot instead (for a demo, a blog post, whatever), use:

```bash
python scripts/export_proofs.py --out proof_library
```

## Repo layout

```
lean4gen/
  model.py, config.py, dashboard.py, repl.py, repl_pool.py,
  search.py, curriculum.py, eval.py, generator.py, schedules.py,
  train.py, train_tokenizer.py, rl_train.py, pipeline.py, store.py
scripts/
  collect_corpus.py, run_pipeline.py, export_proofs.py,
  bot_dump_proofs.py, bot_push_proofs.sh, publish.sh, test_repl_only.py
tests/
setup.sh, setup.bat
```

## Known rough edges

- The REPL's tactic-mode protocol (`open_goal`/`run_tactic`) matches the documented Lean REPL API, but it's worth double-checking against whatever version you're running — it's changed before.
- Beam search treats one line as one tactic, which breaks on multi-line tactics like `simp only [...]` — fine for now, but a real tactic parser would help.
- GRPO's KL coefficient and group size are just starting points, not tuned for any particular model size.
- The default model config (~50-100M params) assumes a corpus in the hundreds of MB range — shrink it if your corpus ends up smaller, or you'll overfit fast.
