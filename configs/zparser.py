import argparse

def int_greater_than_one(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 1:
        raise argparse.ArgumentTypeError(f"expected an integer greater than 1, got {parsed_value}")
    return parsed_value

def positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed_value}")
    return parsed_value

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

    # Attention Config
    parser.add_argument("--attention-kind", choices=("mha", "mqa", "gqa"), default="gqa")
    parser.add_argument("--attention-backend", choices=("eager", "einsum"), default="einsum")
    parser.add_argument("--n-kv-heads", type=positive_int, default=2)
    parser.add_argument(
        "--tie-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # MoE Config
    parser.add_argument("--expert-num", type=int_greater_than_one, default=4)
    parser.add_argument("--top-k", type=int_greater_than_one, default=2)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--use-moe-z-loss", action="store_true")
    parser.add_argument("--moe-backend", choices=("eager", "gemm"), default="gemm")
    parser.add_argument("--gemm-mode", choices=("batched", "grouped"), default="batched")
    args = parser.parse_args()
    if args.top_k > args.expert_num:
        parser.error(
            f"--top-k ({args.top_k}) must be less than or equal to --expert-num ({args.expert_num})"
        )
    if args.attention_kind in ("mha", "mqa") and args.attention_backend != "eager":
        parser.error(
            f"--attention-kind={args.attention_kind} currently only supports "
            "--attention-backend=eager"
        )
    if args.attention_kind == "gqa" and args.heads % args.n_kv_heads != 0:
        parser.error(
            f"--heads ({args.heads}) must be divisible by --n-kv-heads ({args.n_kv_heads})"
        )
    if args.use_moe and args.moe_backend == "gemm" and args.gemm_mode == "grouped":
        parser.error("--gemm-mode=grouped is not implemented yet; use --gemm-mode=batched")
    return args