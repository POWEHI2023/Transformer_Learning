from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def save_model(
    saved_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    global_step: int,
    validation_loss: float,
    best_validation_loss: float,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    tokenizer_config: Mapping[str, Any],
    device: torch.device,
) -> None:
    """Save a resumable model checkpoint and its training metadata."""
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
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "dataset_config": dict(dataset_config),
        "tokenizer_config": dict(tokenizer_config),
        "rng_state": rng_state,
        "runtime": {
            "torch_version": torch.__version__,
            "device_type": device.type,
        },
    }

    torch.save(checkpoint, saved_path)
