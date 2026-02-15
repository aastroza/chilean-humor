from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset

from .config import TopicModelingConfig


ONLY_NUMERIC_RE = re.compile(r"^[\s0-9,.;:/+\-()\[\]{}]+$")
# Any Unicode letter character (excluding digits/underscore).
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def is_numeric_noise(text: str) -> bool:
    """Return True when text only contains punctuation/numbers and no letters."""
    if not text:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    if LETTER_RE.search(stripped):
        return False

    return bool(ONLY_NUMERIC_RE.fullmatch(stripped))


def parse_decade(date_value: str) -> int | None:
    """Extract decade from a YYYY-MM-DD-like string."""
    if not date_value:
        return None

    prefix = date_value[:4]
    if not prefix.isdigit():
        return None

    year = int(prefix)
    return (year // 10) * 10


def load_clean_documents(
    config: TopicModelingConfig,
) -> tuple[list[str], list[int], list[dict[str, Any]], dict[str, Any]]:
    """
    Load dataset from Hugging Face and filter out numeric-only segments.

    Returns:
        documents: cleaned text segments
        decades: decade labels aligned with documents
        cleaned_rows: original dataset rows (plus modeling metadata) aligned with documents
        stats: summary counters useful for logging
    """
    dataset = load_dataset(
        path=config.repo_id,
        name=config.config_name,
        split=config.split,
    )

    documents: list[str] = []
    decades: list[int] = []
    cleaned_rows: list[dict[str, Any]] = []
    skipped_empty = 0
    skipped_too_short = 0
    skipped_numeric_noise = 0
    skipped_invalid_date = 0

    for row_index, row in enumerate(dataset):
        text = str(row.get(config.text_column, "")).strip()
        if not text:
            skipped_empty += 1
            continue
        if len(text) < config.min_text_chars:
            skipped_too_short += 1
            continue
        if len(text.split()) < config.min_text_tokens:
            skipped_too_short += 1
            continue
        if is_numeric_noise(text):
            skipped_numeric_noise += 1
            continue

        raw_date = str(row.get(config.date_column, "")).strip()
        decade = parse_decade(raw_date)
        if decade is None:
            skipped_invalid_date += 1
            continue

        documents.append(text)
        decades.append(decade)
        modeled_row = dict(row)
        if "show" not in modeled_row and "comedian" in modeled_row:
            modeled_row["show"] = modeled_row.get("comedian")
        modeled_row["source_row_index"] = row_index
        modeled_row["model_text"] = text
        modeled_row["model_decade"] = decade
        cleaned_rows.append(modeled_row)

    if config.sample_size is not None:
        documents = documents[: config.sample_size]
        decades = decades[: config.sample_size]
        cleaned_rows = cleaned_rows[: config.sample_size]

    stats = {
        "total_rows": len(dataset),
        "kept_rows": len(documents),
        "skipped_empty": skipped_empty,
        "skipped_too_short": skipped_too_short,
        "skipped_numeric_noise": skipped_numeric_noise,
        "skipped_invalid_date": skipped_invalid_date,
    }
    return documents, decades, cleaned_rows, stats
