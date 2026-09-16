from __future__ import annotations
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer


def train_tokenizer(
    corpus_dir: str = "data/lean_corpus",
    out_dir: str = "data/tokenizer",
    vocab_size: int = 16000,
) -> None:
    files = [str(p) for p in Path(corpus_dir).rglob("*.lean")]
    if not files:
        raise SystemExit(
            f"В {corpus_dir} не найдено .lean файлов — сначала запусти collect_corpus.py"
        )
    print(f"Файлов для обучения токенизатора: {len(files)}")
    tactic_tokens = [
        "rfl",
        "decide",
        "simp",
        "ring",
        "omega",
        "linarith",
        "induction",
        "cases",
        "exact",
        "apply",
        "intro",
        "constructor",
        "sorry",
        "by",
        "theorem",
        "lemma",
        "def",
        "instance",
        "have",
        "show",
        "from",
    ]
    symbol_tokens = [
        "∀",
        "∃",
        "≠",
        "→",
        "↔",
        "∧",
        "∨",
        "¬",
        "∈",
        "⊆",
        "≤",
        "≥",
        "•",
        "∘",
    ]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
        + tactic_tokens
        + symbol_tokens,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(out_path))
    print(f"Токенизатор сохранён в {out_path}")


if __name__ == "__main__":
    train_tokenizer()
