from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    rope_base: float = 10000.0


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-05) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(
    seq_len: int, head_dim: int, base: float, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / base ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos().to(dtype), emb.sin().to(dtype))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.rope_base = cfg.rope_base

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        cos, sin = build_rope_cache(
            position_offset + T, self.head_dim, self.rope_base, x.device, x.dtype
        )
        cos, sin = (cos[position_offset:], sin[position_offset:])
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)
        is_causal = kv_cache is None
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return (self.proj(out), new_cache)


class SwiGLU(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(4 * cfg.n_embd * 2 / 3)
        hidden = (hidden + 63) // 64 * 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache, position_offset)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return (x, new_cache)


class LeanGPT(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        position_offset: int = 0,
    ):
        B, T = idx.shape
        assert (
            T <= self.cfg.block_size
        ), "последовательность длиннее block_size, увеличь cfg.block_size и переобучи"
        x = self.drop(self.tok_emb(idx))
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_cache_i = block(x, cache_i, position_offset)
            new_caches.append(new_cache_i)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return (logits, loss, new_caches)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        logits, _, kv_caches = self(idx)
        position_offset = idx.shape[1]
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cum_probs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, next_sorted)
            idx = torch.cat([idx, next_token], dim=1)
            logits, _, kv_caches = self(
                next_token, kv_caches=kv_caches, position_offset=position_offset
            )
            position_offset += 1
        return idx

    @torch.no_grad()
    def logprob_of_continuation(
        self, prompt_ids: torch.Tensor, cont_ids: torch.Tensor
    ) -> torch.Tensor:
        full = torch.cat([prompt_ids, cont_ids], dim=1)
        logits, _, _ = self(full[:, :-1])
        logp = F.log_softmax(logits, dim=-1)
        target = full[:, 1:]
        tok_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        start = prompt_ids.shape[1] - 1
        return tok_logp[:, start:].sum(dim=1)
