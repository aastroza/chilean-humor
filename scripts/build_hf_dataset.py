#!/usr/bin/env python3
"""Build Hugging Face-ready dataset files from raw 2026 transcripts/metadata.

Creates two dataset configs under the output directory:
  - routines/train.parquet + routines/train.jsonl
  - segments/train.parquet + segments/train.jsonl

Expected source layout:
  data/2026/<id>/<id>_metadata.json
  data/2026/<id>/<id>_transcript.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Value


ROUTINES_FEATURES = Features(
    {
        "id": Value("int64"),
        "show_id": Value("int64"),
        "show": Value("string"),
        "year": Value("int64"),
        "date": Value("string"),
        "festival": Value("string"),
        "tv": Value("string"),
        "event_name": Value("string"),
        "youtube_url": Value("string"),
        "transcript": Value("string"),
        "segments": [
            {
                "start_time_seconds": Value("float64"),
                "end_time_seconds": Value("float64"),
                "text": Value("string"),
                "confidence": Value("float64"),
                "word_count": Value("int64"),
            }
        ],
    }
)

SEGMENTS_FEATURES = Features(
    {
        "id": Value("int64"),
        "segment_idx": Value("int64"),
        "show": Value("string"),
        "year": Value("int64"),
        "date": Value("string"),
        "festival": Value("string"),
        "event_name": Value("string"),
        "youtube_url": Value("string"),
        "start_time_seconds": Value("float64"),
        "end_time_seconds": Value("float64"),
        "duration_seconds": Value("float64"),
        "text": Value("string"),
        "confidence": Value("float64"),
        "word_count": Value("int64"),
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build two HF splits/configs (routines and segments) from raw "
            "metadata/transcript JSON files."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/2026"),
        help="Root directory containing <id>/<id>_metadata.json and <id>_transcript.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("hf_dataset"),
        help="Directory where routines/ and segments/ outputs will be created",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a metadata file does not have its matching transcript file",
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


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def extract_id_from_metadata_path(path: Path) -> int | None:
    suffix = "_metadata"
    stem = path.stem
    if not stem.endswith(suffix):
        return None
    raw_id = stem[: -len(suffix)]
    return as_int(raw_id)


def build_rows(
    input_root: Path, strict: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    routines_rows: list[dict[str, Any]] = []
    segments_rows: list[dict[str, Any]] = []
    missing_transcripts: list[Path] = []

    metadata_paths = sorted(
        input_root.rglob("*_metadata.json"),
        key=lambda p: extract_id_from_metadata_path(p) or -1,
    )

    for metadata_path in metadata_paths:
        routine_id = extract_id_from_metadata_path(metadata_path)
        if routine_id is None:
            continue

        transcript_path = metadata_path.with_name(f"{routine_id}_transcript.json")
        if not transcript_path.exists():
            missing_transcripts.append(transcript_path)
            if strict:
                raise FileNotFoundError(f"Missing transcript file: {transcript_path}")
            continue

        metadata = load_json(metadata_path)
        transcript = load_json(transcript_path)

        segments_source = transcript.get("segments") or []
        routine_segments: list[dict[str, Any]] = []

        for segment_idx, segment in enumerate(segments_source):
            segment_metadata = segment.get("metadata") or {}
            start = as_float(segment.get("start_time_seconds"))
            end = as_float(segment.get("end_time_seconds"))
            text = segment.get("text") or ""
            confidence = as_float(segment_metadata.get("confidence"))
            word_count = as_int(segment_metadata.get("word_count"))

            routine_segment = {
                "start_time_seconds": start,
                "end_time_seconds": end,
                "text": text,
                "confidence": confidence,
                "word_count": word_count,
            }
            routine_segments.append(routine_segment)

            duration = None
            if start is not None and end is not None:
                duration = end - start

            segments_rows.append(
                {
                    "id": as_int(metadata.get("id")) or routine_id,
                    "segment_idx": segment_idx,
                    "show": metadata.get("show"),
                    "year": as_int(metadata.get("year")),
                    "date": metadata.get("date"),
                    "festival": metadata.get("festival"),
                    "event_name": metadata.get("event_name"),
                    "youtube_url": metadata.get("youtube_url"),
                    "start_time_seconds": start,
                    "end_time_seconds": end,
                    "duration_seconds": duration,
                    "text": text,
                    "confidence": confidence,
                    "word_count": word_count,
                }
            )

        routines_rows.append(
            {
                "id": as_int(metadata.get("id")) or routine_id,
                "show_id": as_int(metadata.get("show_id")),
                "show": metadata.get("show"),
                "year": as_int(metadata.get("year")),
                "date": metadata.get("date"),
                "festival": metadata.get("festival"),
                "tv": metadata.get("tv"),
                "event_name": metadata.get("event_name"),
                "youtube_url": metadata.get("youtube_url"),
                "transcript": transcript.get("transcript") or "",
                "segments": routine_segments,
            }
        )

    routines_rows.sort(key=lambda row: (row["id"], row["show"] or ""))
    segments_rows.sort(key=lambda row: (row["id"], row["segment_idx"]))

    return routines_rows, segments_rows, missing_transcripts


def save_dataset(
    rows: list[dict[str, Any]], output_dir: Path, features: Features
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows, features=features)

    parquet_path = output_dir / "train.parquet"
    jsonl_path = output_dir / "train.jsonl"

    dataset.to_parquet(str(parquet_path))
    dataset.to_json(str(jsonl_path), orient="records", lines=True, force_ascii=False)


def main() -> None:
    args = parse_args()
    input_root: Path = args.input_root
    output_root: Path = args.output_root

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    routines_rows, segments_rows, missing_transcripts = build_rows(
        input_root=input_root, strict=args.strict
    )

    if not routines_rows:
        raise RuntimeError(
            f"No valid metadata/transcript pairs found in input root: {input_root}"
        )

    save_dataset(routines_rows, output_root / "routines", ROUTINES_FEATURES)
    save_dataset(segments_rows, output_root / "segments", SEGMENTS_FEATURES)

    print(f"Created routines rows: {len(routines_rows)}")
    print(f"Created segments rows: {len(segments_rows)}")
    print(f"Output root: {output_root.resolve()}")

    if missing_transcripts:
        print(f"Skipped missing transcripts: {len(missing_transcripts)}")
        for path in missing_transcripts[:10]:
            print(f"  - {path}")
        if len(missing_transcripts) > 10:
            print("  - ...")


if __name__ == "__main__":
    main()
