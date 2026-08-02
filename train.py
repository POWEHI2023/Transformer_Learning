from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from ztransformer import Transformer

IGNORE_INDEX = -100
TOKENIZER_PATH = Path("artifacts/tokenizer.json")
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# 从训练好的词表中加载 tokenizer
def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.padding_side = "right"
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, "PAD and EOS must use different token IDs"
    return tokenizer

class LMCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, context_length: int):
        self.tokenizer = tokenizer
        self.context_length = context_length
    def __call__(self, records: list[dict]) -> dict[str, torch.Tensor]:
        texts = [record["text"] for record in records]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.context_length + 1,
            return_attention_mask=True,
            return_tensors="pt"
        )
        full_token_ids = encoded["input_ids"]
        full_attention_mask = encoded["attention_mask"].to(dtype=torch.bool)
        input_ids = full_token_ids[:, :-1]
        attention_mask = full_attention_mask[:, :-1]
        labels = full_token_ids[:, 1:].clone()
        target_mask = full_attention_mask[:, 1:]
        labels.masked_fill_(~target_mask, IGNORE_INDEX)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def resolve_device() -> torch.device:
    try:
        import torch_npu # type: ignore
        if torch.npu.is_available():
            device = torch.device("npu")
        else:
            raise Exception("NPU is not available")
    except Exception:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dense Transformer language model on WikiText-2.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    tokenizer = load_tokenizer(str(TOKENIZER_PATH))
    tokenizer_sha256 = file_sha256(TOKENIZER_PATH)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG) \
        .filter(lambda record: bool(record["text"].strip()))
    collator = LMCollator(tokenizer=tokenizer, context_length=args.context_length)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Embedding weight: [vocab_size, d_model]
    # LM Head weight: [vocab_size, d_model]
    model = Transformer(
        vocab_size=len(tokenizer),
        n_layers=args.layers,
        n_heads=args.heads,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_causal_mask=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    device = resolve_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    global_step = 0
    best_validation_loss = float("inf")

    def save_model(
        saved_path: str | Path,
        *,
        completed_epoch: int,
        validation_loss: float,
    ) -> None:
        saved_path = Path(saved_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)

        rng_state: dict[str, object] = {
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

        torch.save(
            {
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
                "model_config": {
                    "vocab_size": len(tokenizer),
                    "n_layers": args.layers,
                    "n_heads": args.heads,
                    "d_model": args.d_model,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "use_causal_mask": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "tie_embedding": (
                        model.lm_head.weight is model.token_embedding.weight
                    ),
                },
                "training_config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "context_length": args.context_length,
                    "learning_rate": args.lr,
                    "max_grad_norm": 1.0,
                    "ignore_index": IGNORE_INDEX,
                    "optimizer": type(optimizer).__name__,
                },
                "dataset_config": {
                    "name": DATASET_NAME,
                    "config": DATASET_CONFIG,
                    "train_split": "train",
                    "validation_split": "validation",
                    "empty_texts_filtered": True,
                },
                "tokenizer_config": {
                    "path": str(TOKENIZER_PATH),
                    "sha256": tokenizer_sha256,
                    "vocab_size": len(tokenizer),
                    "pad_token_id": tokenizer.pad_token_id,
                    "bos_token_id": tokenizer.bos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "unk_token_id": tokenizer.unk_token_id,
                    "padding_side": tokenizer.padding_side,
                },
                "rng_state": rng_state,
                "runtime": {
                    "torch_version": torch.__version__,
                    "device_type": device.type,
                },
            },
            saved_path,
        )

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                input=logits.reshape(-1, logits.size(-1)),
                target=labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            if global_step % 20 == 0:
                print(f"step={global_step}, loss={loss.item():.4f}, grad_norm={grad_norm.item():.4f}")

        model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for batch in validation_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(token_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    input=logits.reshape(-1, logits.size(-1)),
                    target=labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )

                total_loss += loss.item()
                total_tokens += (labels != IGNORE_INDEX).sum().item()
        average_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(average_loss)).item()
        print(f"epoch={epoch} average_loss={average_loss:.4f} perplexity={perplexity:.4f}")

        if average_loss < best_validation_loss:
            best_validation_loss = average_loss
            save_model(
                "artifacts/model_best.pt",
                completed_epoch=epoch + 1,
                validation_loss=average_loss,
            )
 
    save_model(
        "artifacts/model.pt",
        completed_epoch=args.epochs,
        validation_loss=average_loss,
    )

if __name__ == "__main__":
    main()
