#!/usr/bin/env python3
"""
Configuration management for universal checkpoint training
Reads from .conaconfig file
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict, fields


@dataclass
class ModelConfig:
    name: str = "gpt3-xl-1.3b"
    layers: int = 24
    hidden: int = 2048
    heads: int = 16
    seq: int = 1024
    ffn_hidden: Optional[int] = None
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    normalization: str = "layernorm"
    layernorm_epsilon: float = 1e-5

    def __post_init__(self):
        if self.ffn_hidden is None:
            self.ffn_hidden = 4 * int(self.hidden)


def _safe_model_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    return safe or "gpt"


def apply_model_preset(model: ModelConfig, preset: str) -> ModelConfig:
    p = (preset or "").strip().lower()

    if p in {"gpt3-xl", "gpt3-xl-1.3b", "gpt3_xl_1.3b", "gpt3-1.3b", "gpt3_1.3b"}:
        model.name = "gpt3-xl-1.3b"
        model.layers = 24
        model.hidden = 2048
        model.heads = 16
        model.seq = 1024
        model.ffn_hidden = 8192
        model.attention_dropout = 0.1
        model.hidden_dropout = 0.1
        model.normalization = "layernorm"
        model.layernorm_epsilon = 1e-5
        return model

    raise ValueError(f"Unknown model preset: {preset}")


@dataclass
class TrainingConfig:
    """Training configuration"""
    wandb_project: str = "gpt-python-chain"
    zero_stage: int = 1
    dtype: str = "fp16"
    base_gbs: int = 8
    base_lr: float = 3e-4
    base_min_lr: float = 2e-4
    scale_min_lr: bool = False
    lr_scale_strategy: str = "linear"
    lr_scale_exp: float = 0.25
    lr_scale_pivot_gbs: int = 32
    lr_scale_low_exp: float = 0.5
    lr_scale_high_exp: float = 0.25
    lr_scale_lr_cap: float = 0.0
    fallback: bool = False
    fallback_ema_alpha: float = 0.1
    fallback_patience: int = 100
    lr_decay_tokens: int = 414171136
    lr_warmup_tokens: Optional[int] = None
    lr_warmup_frac: float = 0.01
    weight_decay: float = 0.01
    log_interval: int = 1
    eval_interval: int = 100
    eval_iters: int = 100
    split: str = "98,2,0"
    data_prefix: str = "/workspace/datasets/gpt/wikitext_gpt_text_document"
    vocab_file: str = "/workspace/datasets/gpt/vocab.json"
    merge_file: str = "/workspace/datasets/gpt/merges.txt"
    tokenizer_type: str = "GPT2BPETokenizer"
    data_cache_path: Optional[str] = None
    wandb_mode: Optional[str] = None
    distributed_timeout_minutes: int = 30
    wandb_dir: str = "/workspace/wandb"
    log_root: str = "/workspace/logs"


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = self._find_config_file()

        self.config_path = Path(config_path) if config_path else None
        self.config_data = self._load_config()

        self.model = self._build_section("model", ModelConfig)
        self.training = self._build_section("training", TrainingConfig)

    def _build_section(self, section: str, cls):
        values = self.config_data.get(section)
        if not isinstance(values, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            print(
                f"[WARN] Ignoring unknown {section} keys in "
                f"{self.config_path}: {', '.join(unknown)}"
            )
        return cls(**{k: v for k, v in values.items() if k in known})

    def _find_config_file(self) -> Optional[str]:
        """Find .conaconfig file in current directory or parent directories"""
        current = Path.cwd()
        max_depth = 5

        for _ in range(max_depth):
            config_file = current / ".conaconfig"
            if config_file.exists():
                return str(config_file)
            if current.parent == current:  # Reached root
                break
            current = current.parent

        return None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from .conaconfig file"""
        if self.config_path is None or not self.config_path.exists():
            return {}

        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] Failed to load config from {self.config_path}: {e}")
            return {}

    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file"""
        if path is None:
            path = self.config_path or ".conaconfig"

        config_dict = {
            "model": asdict(self.model),
            "training": asdict(self.training)
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_checkpoint_dir(self, dp: int, tp: int, pp: int, workdir: str) -> str:
        """Get checkpoint directory path"""
        model_name = _safe_model_name(getattr(self.model, "name", "gpt"))
        return (
            f"{workdir}/checkpoints/{model_name}/z{self.training.zero_stage}/"
            f"{self.training.dtype}/tp{tp}_pp{pp}_dp{dp}_sp1"
        )
