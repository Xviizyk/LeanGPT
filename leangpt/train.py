from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer

from .device import pick_device
from .model import LeanGPT, ModelConfig
from .schedules import lr_schedule

MAX_FILE_BYTES = 2_000_000
BATCH_FILES = 128


def _raise_interrupt(signum, frame):
    # kill -TERM (пункт «Остановить» в run.sh) -> сохранить чекпоинт и выйти
    raise KeyboardInterrupt


def build_corpus_cache(
    corpus_dir: str, tokenizer: ByteLevelBPETokenizer, cache_path: Path
) -> np.ndarray:
    dtype = np.uint16 if tokenizer.get_vocab_size() < 65536 else np.int32
    itemsize = np.dtype(dtype).itemsize
    progress_path = cache_path.with_suffix(".json")
    done_path = cache_path.with_suffix(".done")

    if done_path.exists():
        return np.memmap(cache_path, dtype=dtype, mode="r")

    paths = sorted(Path(corpus_dir).rglob("*.lean"))
    files_done, n_tokens = 0, 0
    if progress_path.exists() and cache_path.exists():
        p = json.loads(progress_path.read_text())
        files_done, n_tokens = p["files"], p["tokens"]
        print(f"Продолжаю с файла {files_done}/{len(paths)}")

    eos = tokenizer.token_to_id("<eos>")
    eos = 0 if eos is None else eos

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "r+b" if files_done else "wb") as f:
        f.seek(n_tokens * itemsize)
        f.truncate()
        for start in range(files_done, len(paths), BATCH_FILES):
            batch = paths[start : start + BATCH_FILES]
            texts = []
            for p in batch:
                if p.stat().st_size > MAX_FILE_BYTES:
                    print(f"пропуск (большой файл): {p}")
                    continue
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            for enc in tokenizer.encode_batch(texts):
                arr = np.array(enc.ids + [eos], dtype=dtype)
                arr.tofile(f)
                n_tokens += len(arr)
            f.flush()
            files_done = start + len(batch)
            progress_path.write_text(
                json.dumps({"files": files_done, "tokens": n_tokens})
            )
            print(f"файлов: {files_done}/{len(paths)}, токенов: {n_tokens}")

    done_path.touch()
    return np.memmap(cache_path, dtype=dtype, mode="r")


def get_batch(
    data: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, device: str):
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=lr, betas=(0.9, 0.95), fused=device.startswith("cuda")
    )


def save_ckpt(path: Path, raw_model, optim, cfg, step: int) -> None:
    """Атомарная запись: при падении посреди сохранения старый файл остаётся целым."""
    tmp = path.with_suffix(".tmp")
    torch.save(
        {"model": raw_model.state_dict(), "optim": optim.state_dict(),
         "config": cfg, "step": step},
        tmp,
    )
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, iters, rng, device, amp) -> float:
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, rng)
        with amp():
            _, loss, _ = model(x.to(device), y.to(device))
        total += loss.item()
    model.train()
    return total / iters


def train(
    corpus_dir: str = "data/lean_corpus",
    tokenizer_dir: str = "data/tokenizer",
    out_dir: str = "data/checkpoints",
    cache_path: str = "data/corpus_ids.bin",
    block_size: int = 512,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_steps: int = 5000,
    epochs: float | None = None,
    base_lr: float = 0.0003,
    warmup_frac: float = 0.05,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    val_frac: float = 0.005,
    eval_every: int = 500,
    eval_iters: int = 10,
    save_every: int = 250,
    log_every: int = 10,
    resume: bool = True,
    seed: int = 0,
    num_threads: int | None = None,
    cpu_bf16: bool = False,
    compile_model: bool = False,
    device: str | None = None,
) -> None:
    device = device or pick_device()
    if num_threads:
        torch.set_num_threads(num_threads)
    tokenizer = ByteLevelBPETokenizer(
        f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt"
    )
    print(f"device: {device}, потоков: {torch.get_num_threads()}")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ids = build_corpus_cache(corpus_dir, tokenizer, Path(cache_path))
    print(f"Всего токенов в корпусе: {len(ids)}")

    n_val = max(block_size + 2, int(len(ids) * val_frac))
    train_data, val_data = ids[:-n_val], ids[-n_val:]

    tokens_per_step = batch_size * grad_accum * block_size
    if epochs is not None:
        max_steps = max(1, int(epochs * len(train_data) / tokens_per_step))
    warmup_steps = max(1, int(max_steps * warmup_frac))

    cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size(), block_size=block_size)
    raw_model = LeanGPT(cfg).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"параметров: {n_params / 1e6:.1f}M, токенов за шаг: {tokens_per_step}, "
          f"шагов: {max_steps}")
    model = torch.compile(raw_model) if (compile_model and device.startswith("cuda")) else raw_model
    optim = make_optimizer(raw_model, base_lr, weight_decay, device)

    on_cuda = device.startswith("cuda")
    device_type = "cuda" if on_cuda else "cpu"
    use_amp = on_cuda or (device == "cpu" and cpu_bf16)
    if on_cuda and torch.cuda.get_device_capability()[0] < 8:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=on_cuda and amp_dtype == torch.float16)
    print(f"точность: {str(amp_dtype).split('.')[-1] if use_amp else 'float32'}")

    def amp():
        return torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    last_path = out / "leangpt_last.pt"
    step = 0
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        step = ck["step"]
        print(f"Возобновляю с шага {step}")

    rng = np.random.default_rng(seed + step)
    val_rng = np.random.default_rng(seed)

    signal.signal(signal.SIGTERM, _raise_interrupt)
    model.train()
    t0, last_log = time.time(), step
    loss_acc = 0.0
    try:
        while step < max_steps:
            lr = lr_schedule(step, max_steps, base_lr, warmup_steps)
            for g in optim.param_groups:
                g["lr"] = lr
            optim.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                x, y = get_batch(train_data, block_size, batch_size, rng)
                with amp():
                    _, loss, _ = model(x.to(device), y.to(device))
                scaler.scale(loss / grad_accum).backward()
                loss_acc = loss_acc + loss.detach() / grad_accum
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
            step += 1

            if step % log_every == 0:
                dt = time.time() - t0
                done = step - last_log
                sec_step = dt / done
                eta_min = (max_steps - step) * sec_step / 60
                print(f"step {step}/{max_steps} lr {lr:.2e} "
                      f"loss {float(loss_acc) / log_every:.4f} "
                      f"{tokens_per_step / sec_step:.0f} ток/с ETA {eta_min:.0f} мин")
                loss_acc, t0, last_log = 0.0, time.time(), step
            if step % eval_every == 0:
                v = evaluate(model, val_data, block_size, batch_size,
                             eval_iters, val_rng, device, amp)
                print(f"step {step} val loss {v:.4f}")
            if step % save_every == 0:
                save_ckpt(last_path, raw_model, optim, cfg, step)
    except KeyboardInterrupt:
        print("Прервано, сохраняю чекпоинт...")
    finally:
        save_ckpt(last_path, raw_model, optim, cfg, step)

    final_path = out / "lean_gpt_latest.pt"
    torch.save({"model": raw_model.state_dict(), "config": cfg}, final_path)
    print(f"Сохранено: {last_path}, {final_path}")


def _cli() -> None:
    ap = argparse.ArgumentParser(description="LeanGPT pretraining")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--cpu-bf16", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out-dir", default="data/checkpoints")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    a = ap.parse_args()
    train(
        max_steps=a.max_steps,
        batch_size=a.batch_size,
        grad_accum=a.grad_accum,
        block_size=a.block_size,
        num_threads=a.threads,
        resume=not a.no_resume,
        cpu_bf16=a.cpu_bf16,
        compile_model=a.compile,
        out_dir=a.out_dir,
        save_every=a.save_every,
        eval_every=a.eval_every,
        log_every=a.log_every,
    )


if __name__ == "__main__":
    _cli()