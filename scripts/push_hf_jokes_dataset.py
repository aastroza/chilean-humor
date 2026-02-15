#!/usr/bin/env python3
"""Push local jokes dataset files to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push jokes config to a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination dataset repo id, for example: astroza/chilean-humor-jokes",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("hf_dataset"),
        help="Root folder that contains jokes/train.parquet",
    )
    parser.add_argument(
        "--config-name",
        default="jokes",
        help="Dataset config name to push (default: jokes)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name to push (default: train)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not exist yet",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch or revision to push to",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Push changes to a PR instead of directly to the target revision",
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Max shard size for uploaded parquet files (default: 500MB)",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def main() -> None:
    args = parse_args()

    jokes_path = args.dataset_root / args.config_name / "train.parquet"
    require_file(jokes_path)

    token = args.token
    if not token:
        print(
            "HF token not provided. Using existing local HF CLI login credentials "
            "(if available)."
        )

    private_value = True if args.private else None

    print(f"Loading local dataset file: {jokes_path.resolve()}")
    jokes = Dataset.from_parquet(str(jokes_path))
    print(f"Loaded jokes rows: {len(jokes)}")

    print(f"Pushing config '{args.config_name}' to {args.repo_id}...")
    commit_info = jokes.push_to_hub(
        repo_id=args.repo_id,
        config_name=args.config_name,
        split=args.split,
        set_default=True,
        private=private_value,
        token=token,
        revision=args.revision,
        create_pr=args.create_pr,
        max_shard_size=args.max_shard_size,
    )
    print(f"Done: {commit_info}")
    print(f"Dataset available at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
