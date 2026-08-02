from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    decoders,
    normalizers,
    pre_tokenizers,
    processors,
)
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Byte-level BPE tokenizer on WikiText-2.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=8_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing tokenizer file.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Tokenizer file {args.output} already exists. Use --force to overwrite.")

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"].filter(lambda record: bool(record["text"].strip()))
    # lambda 只适用于 worker=0, 未来更改 worker 参数时需要改为模块级函数
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda records: [record["text"] for record in records]
    )

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    # 统一等价的 Unicode 表达形式
    tokenizer.normalizer = normalizers.NFC()
    # 将任意的 UTF-8 文本映射到 byte alphabet
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # 将 byte level token 恢复成文本
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        iterator=iter(train_dataloader),
        trainer=trainer,
        length=len(train_dataset),
    )

    bos_token_id = tokenizer.token_to_id("<bos>") 
    eos_token_id = tokenizer.token_to_id("<eos>")
    assert bos_token_id is not None, "Tokenizer must have a <bos> token."
    assert eos_token_id is not None, "Tokenizer must have a <eos> token."
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", bos_token_id),
            ("<eos>", eos_token_id),
        ],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output), pretty=True)


if __name__ == "__main__":
    main()