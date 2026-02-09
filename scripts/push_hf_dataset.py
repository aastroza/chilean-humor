#!/usr/bin/env python3
"""Push local Hugging Face dataset files to the Hub.

Expected local files:
  hf_dataset/routines/train.parquet
  hf_dataset/segments/train.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push routines + segments configs to a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination dataset repo id, for example: username/chilean-humor-raw-transcripts",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("hf_dataset"),
        help="Root folder that contains routines/train.parquet and segments/train.parquet",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name to push (default: train)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it doesn't exist yet",
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

    routines_path = args.dataset_root / "routines" / "train.parquet"
    segments_path = args.dataset_root / "segments" / "train.parquet"

    require_file(routines_path)
    require_file(segments_path)

    token = args.token
    if not token:
        print(
            "HF token not provided. Using existing local HF CLI login credentials "
            "(if available)."
        )

    private_value = True if args.private else None

    print(f"Loading local dataset files from: {args.dataset_root.resolve()}")
    routines = Dataset.from_parquet(str(routines_path))
    segments = Dataset.from_parquet(str(segments_path))
    print(f"Loaded routines rows: {len(routines)}")
    print(f"Loaded segments rows: {len(segments)}")

    print(f"Pushing config 'routines' to {args.repo_id}...")
    routines_commit = routines.push_to_hub(
        repo_id=args.repo_id,
        config_name="routines",
        split=args.split,
        set_default=True,
        private=private_value,
        token=token,
        revision=args.revision,
        create_pr=args.create_pr,
        max_shard_size=args.max_shard_size,
    )
    print(f"Done routines: {routines_commit}")

    print(f"Pushing config 'segments' to {args.repo_id}...")
    segments_commit = segments.push_to_hub(
        repo_id=args.repo_id,
        config_name="segments",
        split=args.split,
        set_default=False,
        private=private_value,
        token=token,
        revision=args.revision,
        create_pr=args.create_pr,
        max_shard_size=args.max_shard_size,
    )
    print(f"Done segments: {segments_commit}")

    print(f"Dataset available at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
