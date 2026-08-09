from __future__ import annotations

import hashlib
import argparse
from dataclasses import asdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from configs.zconfig import ModelConfig

IGNORE_INDEX = -100
TOKENIZER_PATH = Path("artifacts/tokenizer.json")
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"

_config: dict[str, Mapping[str, Any]] | None = None

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def record_model_config(
    tokenizer: Any,
    model_config: ModelConfig,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
) -> None:
    global _config
    tokenizer_sha256 = file_sha256(TOKENIZER_PATH)

    checkpoint_model_config = {
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        **asdict(model_config),
    }
    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "learning_rate": args.lr,
        "moe_auxloss_coeff": args.moe_auxloss_coeff,
        "moe_zloss_coeff": args.moe_zloss_coeff,
        "max_grad_norm": 1.0,
        "ignore_index": IGNORE_INDEX,
        "optimizer": type(optimizer).__name__,
    }
    dataset_config = {
        "name": DATASET_NAME,
        "config": DATASET_CONFIG,
        "train_split": "train",
        "validation_split": "validation",
        "empty_texts_filtered": True,
    }
    tokenizer_config = {
        "path": str(TOKENIZER_PATH),
        "sha256": tokenizer_sha256,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "padding_side": tokenizer.padding_side,
    }

    _config = {
        "model_config": checkpoint_model_config,
        "training_config": training_config,
        "dataset_config": dataset_config,
        "tokenizer_config": tokenizer_config,
    }

def save_model(
    saved_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    global_step: int,
    validation_loss: float,
    best_validation_loss: float,
    device: torch.device,
) -> None:
    """Save a resumable model checkpoint and its training metadata."""
    if _config is None:
        raise RuntimeError(
            "Checkpoint config is not initialized; call record_model_config() before save_model()"
        )

    saved_path = Path(saved_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    rng_state: dict[str, object] = {
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()

    checkpoint = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "progress": {
            "completed_epoch": completed_epoch,
            "global_step": global_step,
        },
        "metrics": {
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
        },
        "model_config": dict(_config["model_config"]),
        "training_config": dict(_config["training_config"]),
        "dataset_config": dict(_config["dataset_config"]),
        "tokenizer_config": dict(_config["tokenizer_config"]),
        "rng_state": rng_state,
        "runtime": {
            "torch_version": torch.__version__,
            "device_type": device.type,
        },
    }

    torch.save(checkpoint, saved_path)
