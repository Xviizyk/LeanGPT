from __future__ import annotations
import time
from dataclasses import dataclass, field
import torch
from tokenizers import ByteLevelBPETokenizer
from .device import pick_device
from .model import LeanGPT, ModelConfig


@dataclass
class GenConfig:
    checkpoint_path: str = "data/checkpoints/leangpt_epoch2.pt"
    tokenizer_dir: str = "data/tokenizer"
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.95
    num_candidates: int = 8
    device: str = field(default_factory=pick_device)


class LeanGenerator:

    def __init__(self, cfg: GenConfig) -> None:
        self.cfg = cfg
        self._tokenizer: ByteLevelBPETokenizer | None = None
        self._model: LeanGPT | None = None
        self.last_gen_stats: tuple[int, float] = (0, 0.0)

    def load(self) -> None:
        print(f"[DEBUG] Попытка загрузить токенизатор из папки: {os.path.abspath(self.cfg.tokenizer_dir)}")
        self._tokenizer = ByteLevelBPETokenizer(
            f"{self.cfg.tokenizer_dir}/vocab.json",
            f"{self.cfg.tokenizer_dir}/merges.txt",
        )
        ckpt = torch.load(self.cfg.checkpoint_path, map_location=self.cfg.device)
        model_cfg: ModelConfig = ckpt["config"]
        self._model = LeanGPT(model_cfg).to(self.cfg.device)
        self._model.load_state_dict(ckpt["model"])
        self._model.eval()

    def generate(self, prompt: str) -> list[str]:
        assert self._model is not None, "вызови load() перед generate()"
        prompt_ids = self._tokenizer.encode(prompt).ids
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.cfg.device)
        idx = idx.repeat(self.cfg.num_candidates, 1)
        start = time.perf_counter()
        out = self._model.generate(
            idx,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
        elapsed = time.perf_counter() - start
        n_new_tokens = (out.shape[1] - idx.shape[1]) * out.shape[0]
        self.last_gen_stats = (n_new_tokens, elapsed)
        results = []
        for seq in out:
            full_ids = seq.tolist()
            completion_ids = full_ids[len(prompt_ids) :]
            text = self._tokenizer.decode(completion_ids)
            results.append(text.strip())
        return results


def build_prompt(statement: str) -> str:
    return f"{statement} := "
