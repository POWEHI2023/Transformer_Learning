from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from model.mstore import save_model
from model.zmoe import MoEConfig
from model.ztransformer import Transformer, TransformerOutput

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def int_greater_than_one(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 1:
        raise argparse.ArgumentTypeError(f"expected an integer greater than 1, got {parsed_value}")
    return parsed_value


def format_router_stats(output: TransformerOutput) -> str:
    layer_stats: list[str] = []
    for layer_index, stats in enumerate(output.router_stats):
        token_counts = ",".join(str(count) for count in stats.tokens_per_expert.tolist())
        probabilities = ",".join(
            f"{probability:.3f}" for probability in stats.probability_per_expert.tolist()
        )
        layer_stats.append(
            f"layer_{layer_index}[tokens=({token_counts}), probs=({probabilities})]"
        )
    return " | ".join(layer_stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dense or MoE Transformer language model on WikiText-2.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--moe-auxloss-coeff", type=float, default=1e-3)
    parser.add_argument("--moe-zloss-coeff", type=float, default=1e-3)

    # MoE Config
    parser.add_argument("--expert-num", type=int_greater_than_one, default=4)
    parser.add_argument("--top-k", type=int_greater_than_one, default=2)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--use-moe-z-loss", action="store_true")
    args = parser.parse_args()
    if args.top_k > args.expert_num:
        parser.error(
            f"--top-k ({args.top_k}) must be less than or equal to --expert-num ({args.expert_num})"
        )
    return args

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

    moe_config = (
        MoEConfig(
            expert_num=args.expert_num,
            top_k=args.top_k,
            use_z_loss=args.use_moe_z_loss,
        )
        if args.use_moe
        else None
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
        moe_config=moe_config,
    )

    device = resolve_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    global_step = 0
    best_validation_loss = float("inf")

    model_config = {
        "vocab_size": len(tokenizer),
        "n_layers": args.layers,
        "n_heads": args.heads,
        "d_model": args.d_model,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "use_causal_mask": True,
        "pad_token_id": tokenizer.pad_token_id,
        "tie_embedding": model.lm_head.weight is model.token_embedding.weight,
        "ffn_type": "moe" if moe_config is not None else "dense",
        "moe_config": (
            {
                "expert_num": moe_config.expert_num,
                "top_k": moe_config.top_k,
                "use_z_loss": moe_config.use_z_loss,
            }
            if moe_config is not None
            else None
        ),
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

    print(
        f"training_config: device={device}, epochs={args.epochs}, "
        f"steps_per_epoch={len(train_loader)}, batch_size={args.batch_size}, "
        f"context_length={args.context_length}, lr={args.lr}"
    )
    print(
        f"model_config: layers={args.layers}, heads={args.heads}, "
        f"d_model={args.d_model}, hidden_dim={args.hidden_dim}, dropout={args.dropout}, "
        f"trainable_parameters={sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}"
    )
    if moe_config is not None:
        print(
            f"moe_config: experts={moe_config.expert_num}, top_k={moe_config.top_k}, "
            f"use_z_loss={moe_config.use_z_loss}, "
            f"aux_coeff={args.moe_auxloss_coeff}, z_coeff={args.moe_zloss_coeff}"
        )
    else:
        print("moe_config: disabled (using DenseFFN)")

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            _o: TransformerOutput = model(token_ids=input_ids, attention_mask=attention_mask)
            logits = _o.logits
            lm_loss = F.cross_entropy(
                input=logits.reshape(-1, logits.size(-1)),
                target=labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss = (
                lm_loss
                + args.moe_auxloss_coeff * _o.router_aux_loss
                + args.moe_zloss_coeff * _o.router_z_loss
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            if global_step % 20 == 0:
                print(
                    f"train: epoch={epoch + 1}/{args.epochs}, step={global_step}, "
                    f"loss={loss.item():.4f}, lm_loss={lm_loss.item():.4f}, "
                    f"aux_loss={_o.router_aux_loss.item():.4f}, "
                    f"z_loss={_o.router_z_loss.item():.4f}, "
                    f"grad_norm={grad_norm.item():.4f}"
                )
                if _o.router_stats:
                    print(f"router: {format_router_stats(_o)}")

        model.eval()
        total_loss, total_aux_loss, total_z_loss = 0.0, 0.0, 0.0
        total_tokens, validation_batches = 0, 0
        with torch.no_grad():
            for batch in validation_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                _o = model(token_ids=input_ids, attention_mask=attention_mask)
                logits = _o.logits
                lm_loss = F.cross_entropy(
                    input=logits.reshape(-1, logits.size(-1)),
                    target=labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                loss = (
                    lm_loss
                    + args.moe_auxloss_coeff * _o.router_aux_loss
                    + args.moe_zloss_coeff * _o.router_z_loss
                )

                total_loss += loss.item()
                total_aux_loss += _o.router_aux_loss.item()
                total_z_loss += _o.router_z_loss.item()
                total_tokens += (labels != IGNORE_INDEX).sum().item()
                validation_batches += 1
        average_loss = total_loss / total_tokens
        average_aux_loss = total_aux_loss / validation_batches
        average_z_loss = total_z_loss / validation_batches
        perplexity = torch.exp(torch.tensor(average_loss)).item()
        print(
            f"validation: epoch={epoch + 1}/{args.epochs}, "
            f"average_loss={average_loss:.4f}, perplexity={perplexity:.4f}, "
            f"aux_loss={average_aux_loss:.4f}, z_loss={average_z_loss:.4f}, "
            f"tokens={total_tokens}"
        )

        if average_loss < best_validation_loss:
            best_validation_loss = average_loss
            save_model(
                "artifacts/model_best.pt",
                model=model,
                optimizer=optimizer,
                completed_epoch=epoch + 1,
                global_step=global_step,
                validation_loss=average_loss,
                best_validation_loss=best_validation_loss,
                model_config=model_config,
                training_config=training_config,
                dataset_config=dataset_config,
                tokenizer_config=tokenizer_config,
                device=device,
            )
 
    save_model(
        "artifacts/model.pt",
        model=model,
        optimizer=optimizer,
        completed_epoch=args.epochs,
        global_step=global_step,
        validation_loss=average_loss,
        best_validation_loss=best_validation_loss,
        model_config=model_config,
        training_config=training_config,
        dataset_config=dataset_config,
        tokenizer_config=tokenizer_config,
        device=device,
    )

if __name__ == "__main__":
    main()
