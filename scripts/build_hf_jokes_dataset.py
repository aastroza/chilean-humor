#!/usr/bin/env python3
"""Build Hugging Face-ready jokes dataset files from local jokes + metadata JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Value


JOKES_FEATURES = Features(
    {
        "id": Value("int64"),
        "routine_id": Value("int64"),
        "joke_idx": Value("int64"),
        "comedian": Value("string"),
        "year": Value("int64"),
        "date": Value("string"),
        "festival": Value("string"),
        "event_name": Value("string"),
        "youtube_url": Value("string"),
        "text": Value("string"),
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build HF dataset config 'jokes' from data/2026/*/jokes.json files "
            "plus routine metadata."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/2026"),
        help="Root directory containing <id>/jokes.json and <id>_metadata.json files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("hf_dataset"),
        help="Directory where jokes/train.parquet and jokes/train.jsonl are written.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a jokes.json file does not have matching metadata.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return int(stripped)
        except ValueError:
            try:
                return int(float(stripped))
            except ValueError:
                return None
    return None


def extract_joke_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("cleaned_transcript", "text", "transcript", "joke"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def choose_comedian(metadata: dict[str, Any]) -> str:
    return (metadata.get("show") or "").strip()


def find_metadata_path(jokes_path: Path, routine_id: int | None) -> Path | None:
    if routine_id is not None:
        candidate = jokes_path.parent / f"{routine_id}_metadata.json"
        if candidate.exists():
            return candidate

    candidates = sorted(jokes_path.parent.glob("*_metadata.json"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def build_rows(input_root: Path, strict: bool) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    missing_metadata: list[Path] = []

    jokes_paths = sorted(
        input_root.rglob("jokes.json"),
        key=lambda p: as_int(p.parent.name) or -1,
    )

    for jokes_path in jokes_paths:
        routine_id = as_int(jokes_path.parent.name)
        metadata_path = find_metadata_path(jokes_path, routine_id)

        if metadata_path is None:
            missing = jokes_path.parent / f"{jokes_path.parent.name}_metadata.json"
            missing_metadata.append(missing)
            if strict:
                raise FileNotFoundError(f"Missing metadata file for: {jokes_path}")
            continue

        jokes_payload = load_json(jokes_path)
        metadata = load_json(metadata_path)

        resolved_routine_id = as_int(metadata.get("id")) or routine_id
        comedian = choose_comedian(metadata)
        jokes_source = jokes_payload.get("jokes") or []

        for joke_idx, joke_item in enumerate(jokes_source):
            text = extract_joke_text(joke_item)
            if not text:
                continue

            rows.append(
                {
                    "id": 0,  # assigned after global ordering
                    "routine_id": resolved_routine_id,
                    "joke_idx": joke_idx,
                    "comedian": comedian,
                    "year": as_int(metadata.get("year")),
                    "date": metadata.get("date"),
                    "festival": metadata.get("festival"),
                    "event_name": metadata.get("event_name"),
                    "youtube_url": metadata.get("youtube_url"),
                    "text": text,
                }
            )

    rows.sort(
        key=lambda row: (
            row["routine_id"] if row["routine_id"] is not None else -1,
            row["joke_idx"],
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["id"] = idx

    return rows, missing_metadata


def save_dataset(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows, features=JOKES_FEATURES)
    dataset.to_parquet(str(output_dir / "train.parquet"))
    dataset.to_json(
        str(output_dir / "train.jsonl"), orient="records", lines=True, force_ascii=False
    )


def main() -> None:
    args = parse_args()

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    rows, missing_metadata = build_rows(
        input_root=args.input_root,
        strict=args.strict,
    )

    if not rows:
        raise RuntimeError(f"No jokes rows were built from input root: {args.input_root}")

    output_dir = args.output_root / "jokes"
    save_dataset(rows, output_dir)

    print(f"Created jokes rows: {len(rows)}")
    print(f"Output dir: {output_dir.resolve()}")

    if missing_metadata:
        print(f"Skipped missing metadata files: {len(missing_metadata)}")
        for path in missing_metadata[:10]:
            print(f"  - {path}")
        if len(missing_metadata) > 10:
            print("  - ...")


if __name__ == "__main__":
    main()
