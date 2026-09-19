from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer


def text_iterator(files: list[str]) -> Iterator[str]:
    """Генератор для безопасного потокового чтения файлов.

    - errors="ignore" исключает падение на не-UTF-8 байтах.
    - Чтение по одному файлу не переполняет оперативную память на больших датасетах.
    """
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content.strip():
                    yield content
        except Exception:
            # Пропускаем недоступные или битые файлы
            continue


def train_tokenizer(
    corpus_dir: str = "data/lean_corpus",
    out_dir: str = "data/tokenizer",
    vocab_size: int = 16000,
) -> None:
    # Ищем только файлы исходников Lean
    files = [str(p) for p in Path(corpus_dir).rglob("*.lean")]
    if not files:
        raise SystemExit(
            f"В {corpus_dir} не найдено .lean файлов — сначала запусти collect_corpus.py"
        )
    print(f"Файлов для обучения токенизатора: {len(files)}")

    # Настоящие специальные токены (служебные теги разметки).
    # Обычные тактики (simp, rw, rfl) и символы (∀, ∃, →) BPE сам автоматически
    # добавит в словарь и свяжет с пробелами намного эффективнее.
    special_tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        # Теги структуры промптов (для пар "цель -> тактика" и доказательств)
        "<state>",
        "</state>",
        "<goal>",
        "</goal>",
        "<tactic>",
        "</tactic>",
        "<proof>",
        "</proof>",
    ]

    tokenizer = ByteLevelBPETokenizer()

    print(
        f"Запуск обучения BPE токенизатора (размер словаря: {vocab_size})..."
    )
    tokenizer.train_from_iterator(
        iterator=text_iterator(files),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Сохраняем классические vocab.json и merges.txt
    tokenizer.save_model(str(out_path))

    # 2. Сохраняем полный единый конфиг tokenizer.json со всеми спецтокенами
    tokenizer.save(str(out_path / "tokenizer.json"))

    print(f"Токенизатор успешно сохранён в: {out_path.resolve()}")


if __name__ == "__main__":
    train_tokenizer()
