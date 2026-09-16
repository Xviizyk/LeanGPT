"""
Обучение LeanGPT с нуля (next-token prediction) на собранном
и токенизированном корпусе.

Запуск:
    python lean4gen/train.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader, Dataset

from .model import LeanGPT, ModelConfig


class LeanTextDataset(Dataset):
    """Режет весь корпус на непересекающиеся окна длиной block_size."""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        self.data = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) // self.block_size - 1)

    def __getitem__(self, i: int):
        start = i * self.block_size
        chunk = self.data[start: start + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_corpus_ids(corpus_dir: str, tokenizer: ByteLevelBPETokenizer) -> list[int]:
    ids: list[int] = []
    for path in Path(corpus_dir).rglob("*.lean"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        ids.extend(tokenizer.encode(text).ids)
        ids.append(tokenizer.token_to_id("<eos>") or 0)
    return ids


def train(
    corpus_dir: str = "data/lean_corpus",
    tokenizer_dir: str = "data/tokenizer",
    out_dir: str = "data/checkpoints",
    block_size: int = 512,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    tokenizer = ByteLevelBPETokenizer(
        f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt"
    )

    print("Токенизация корпуса...")
    ids = load_corpus_ids(corpus_dir, tokenizer)
    print(f"Всего токенов в корпусе: {len(ids)}")

    dataset = LeanTextDataset(ids, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size(), block_size=block_size)
    model = LeanGPT(cfg).to(device)
    if device == "cuda":
        model = torch.compile(model)  # ускоряет обучение на современных GPU
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, y)

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1

        ckpt_path = Path(out_dir) / f"lean_gpt_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
        print(f"Сохранён чекпоинт: {ckpt_path}")


if __name__ == "__main__":
    train()
