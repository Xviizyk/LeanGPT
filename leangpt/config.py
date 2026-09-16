from __future__ import annotations
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class Config:
    repl_bin: str = ""
    lean_project_dir: str = ""
    env_import: Optional[str] = None
    checkpoint_path: str = "data/checkpoints/leangpt_latest.pt"
    tokenizer_dir: str = "data/tokenizer"
    results_jsonl: str = "data/results.jsonl"
    github_token: Optional[str] = None
    proofs_repo: Optional[str] = None
    sources_repo: Optional[str] = None


def load_config() -> Config:
    path = os.environ.get("LEANGPT_CONFIG")
    if not path and Path("config.local.yaml").exists():
        path = "config.local.yaml"
    data: dict = {}
    if path and Path(path).exists():
        if yaml is None:
            raise RuntimeError(
                "Установи pyyaml, чтобы читать config.local.yaml: pip install pyyaml"
            )
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    env_map = {
        "repl_bin": "REPL_BIN",
        "lean_project_dir": "LEAN_PROJECT_DIR",
        "env_import": "ENV_IMPORT",
        "checkpoint_path": "CHECKPOINT_PATH",
        "tokenizer_dir": "TOKENIZER_DIR",
        "results_jsonl": "RESULTS_JSONL",
        "github_token": "GITHUB_TOKEN",
        "proofs_repo": "PROOFS_REPO",
        "sources_repo": "SOURCES_REPO",
    }
    for field_name, env_name in env_map.items():
        if os.environ.get(env_name):
            data[field_name] = os.environ[env_name]
    valid_keys = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return Config(**filtered)
