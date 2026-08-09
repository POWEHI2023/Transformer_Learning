from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from model.mstore import record_model_config, save_model
from model.ztransformer import Transformer, TransformerOutput
from configs.zconfig import build_model_config
from configs.zparser import parse_args
from zlogger import log_configs

IGNORE_INDEX = -100
TOKENIZER_PATH = Path("artifacts/tokenizer.json")
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"

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

def main() -> None:
    args = parse_args()
    model_config = build_model_config(args)

    tokenizer = load_tokenizer(str(TOKENIZER_PATH))
    
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
        model_config=model_config,
        pad_token_id=tokenizer.pad_token_id,
    )

    device = resolve_device()
    model.to(device)

    # 先创建 Transformer 模型在创建 Optimizer, 在 Transformer 中可能会发生权重重绑定
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    record_model_config(
        tokenizer=tokenizer,
        model_config=model_config,
        args=args,
        optimizer=optimizer,
    )
    global_step = 0
    best_validation_loss = float("inf")

    log_configs(
        model_config=model_config,
        device=device,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        batch_size=args.batch_size,
        context_length=args.context_length,
        learning_rate=args.lr,
        trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        moe_aux_loss_coeff=args.moe_auxloss_coeff,
        moe_z_loss_coeff=args.moe_zloss_coeff,
    )

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
        device=device,
    )

if __name__ == "__main__":
    main()
