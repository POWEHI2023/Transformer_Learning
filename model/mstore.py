from __future__ import annotations

import hashlib
import argparse
import random
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
    validation_lm_loss: float,
    validation_aux_loss: float,
    validation_z_loss: float,
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
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        rng_state["mps"] = torch.mps.get_rng_state()

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
            "validation_lm_loss": validation_lm_loss,
            "validation_aux_loss": validation_aux_loss,
            "validation_z_loss": validation_z_loss,
            "best_validation_loss": best_validation_loss,
        },
        "model_config": dict(_config["model_config"]),
        "training_config": dict(_config["training_config"]),
        "dataset_config": dict(_config["dataset_config"]),
        "tokenizer_config": dict(_config["tokenizer_config"]),
        "rng_state": rng_state,
        "runtime": {
            "torch_version": str(torch.__version__),
            "device_type": device.type,
        },
    }

    torch.save(checkpoint, saved_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    resume: bool = False,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint for model initialization or training continuation.

    With ``resume=False``, only model weights are restored when ``model`` is
    provided. With ``resume=True``, both ``model`` and ``optimizer`` are
    required, and optimizer plus available RNG states are restored as well.
    Progress and metrics are always returned as part of the checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    # Older format-v1 files may contain torch.__version__ as TorchVersion
    # instead of a plain string. Allow only this known metadata type while
    # keeping the safer weights-only loader enabled.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Invalid checkpoint: expected a mapping at the top level")

    required_keys = {
        "format_version",
        "model",
        "model_config",
        "training_config",
        "tokenizer_config",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Invalid checkpoint: missing required keys: {missing}")

    if checkpoint["format_version"] != 1:
        raise ValueError(
            f"Unsupported checkpoint format version: {checkpoint['format_version']}"
        )
    if not isinstance(checkpoint["model"], Mapping):
        raise ValueError("Invalid checkpoint: 'model' must contain a state dict")

    resume_rng_state: Mapping[str, Any] | None = None
    if resume:
        if model is None:
            raise ValueError("Resume requires a model instance")
        if optimizer is None:
            raise ValueError("Resume requires an optimizer instance")

        resume_keys = {"optimizer", "progress", "metrics", "rng_state"}
        missing_resume_keys = resume_keys.difference(checkpoint)
        if missing_resume_keys:
            missing = ", ".join(sorted(missing_resume_keys))
            raise ValueError(f"Checkpoint cannot be resumed: missing keys: {missing}")
        if not isinstance(checkpoint["optimizer"], Mapping):
            raise ValueError("Checkpoint cannot be resumed: 'optimizer' must be a state dict")
        if not isinstance(checkpoint["progress"], Mapping):
            raise ValueError("Checkpoint cannot be resumed: 'progress' must be a mapping")
        missing_progress_keys = {"completed_epoch", "global_step"}.difference(
            checkpoint["progress"]
        )
        if missing_progress_keys:
            missing = ", ".join(sorted(missing_progress_keys))
            raise ValueError(f"Checkpoint cannot be resumed: missing progress keys: {missing}")
        if not isinstance(checkpoint["metrics"], Mapping):
            raise ValueError("Checkpoint cannot be resumed: 'metrics' must be a mapping")
        if "best_validation_loss" not in checkpoint["metrics"]:
            raise ValueError(
                "Checkpoint cannot be resumed: missing metric: best_validation_loss"
            )
        if not isinstance(checkpoint["rng_state"], Mapping):
            raise ValueError("Checkpoint cannot be resumed: 'rng_state' must be a mapping")
        resume_rng_state = checkpoint["rng_state"]

        torch_rng_state = resume_rng_state.get("torch")
        if not isinstance(torch_rng_state, torch.Tensor):
            raise ValueError("Checkpoint cannot be resumed: missing torch RNG state")

        cuda_rng_states = resume_rng_state.get("cuda")
        if cuda_rng_states is not None and torch.cuda.is_available():
            if len(cuda_rng_states) != torch.cuda.device_count():
                raise ValueError(
                    "Checkpoint CUDA RNG state count does not match the current CUDA device count"
                )

    if model is not None:
        model.load_state_dict(checkpoint["model"], strict=strict)

    if resume:
        assert optimizer is not None and resume_rng_state is not None
        optimizer.load_state_dict(checkpoint["optimizer"])

        torch_rng_state = resume_rng_state["torch"]
        torch.set_rng_state(torch_rng_state.cpu())

        python_rng_state = resume_rng_state.get("python")
        if python_rng_state is not None:
            random.setstate(python_rng_state)

        cuda_rng_states = resume_rng_state.get("cuda")
        if cuda_rng_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_states])

        mps_rng_state = resume_rng_state.get("mps")
        if mps_rng_state is not None and torch.backends.mps.is_available():
            torch.mps.set_rng_state(mps_rng_state.cpu())

    return dict(checkpoint)
