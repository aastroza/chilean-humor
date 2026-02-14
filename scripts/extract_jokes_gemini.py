#!/usr/bin/env python3
"""Extract jokes from transcript.json files using Gemini structured output."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

load_dotenv()


class JokeItem(BaseModel):
    kind: Literal["joke", "non_joke"] = Field(
        description=(
            "Use 'joke' ONLY if this is actual comedic material with setup and "
            "punchline. Use 'non_joke' for introductions, awards, applause, host "
            "speech, promos, music, etc."
        )
    )
    transcript: str = Field(
        description="Original joke transcript exactly as spoken, without timestamps."
    )
    cleaned_transcript: str = Field(
        description=(
            "Lightly cleaned version of transcript with minimal edits only. "
            "Apply minor spelling/punctuation fixes when clearly needed, but do "
            "not paraphrase, summarize, neutralize slang, or change regional words."
        )
    )


class Repertoire(BaseModel):
    jokes: List[JokeItem]


SYSTEM_INSTRUCTIONS = """You are a stand-up comedy writer and transcript analyst.
Your task is to extract JOKES (bits) from a noisy script/transcript.

Rules:
- Return ONLY valid JSON that matches the provided schema. No extra text.
- Do not invent anything: use only content from the input.
- Do NOT include timestamps.
- EXCLUDE as non_joke (not jokes):
  * Comedian introductions (e.g., "please welcome..."), host greetings, award thank-yous.
  * Award segments, applause-only moments, credits, promos, commercial breaks.
  * Music or singing that is not explicitly part of the joke.
- Include ONLY completed jokes. If the punchline is cut off, do NOT include it as a joke.
- cleaned_transcript must be MINIMALLY edited:
  * Keep local/regional vocabulary exactly as spoken (e.g., Chilean slang/chilenismos).
  * Do NOT paraphrase, summarize, shorten, or merge dialogues.
  * Keep character voices and sentence structure.
  * Only fix minor spelling/punctuation issues when clearly appropriate.
  * If in doubt, keep cleaned_transcript equal to transcript.
- To keep JSON valid:
  * Prefer guillemets « » for quoted dialogue inside transcript fields.
  * Avoid raw double quotes inside transcript strings; if unavoidable, escape them as \\".
  * Return strictly valid JSON (parseable without fixes).

Respond in Spanish.
"""

USER_TASK = """Extract the jokes from the following text. Remember to mark as non_joke anything that is intro/awards/host/promo/applause.
For cleaned_transcript, make only minimal corrections and preserve local wording and full dialogue.
Prefer « » for dialogue quotes so JSON stays valid.
Text:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract jokes from transcript JSON files under data/2026 using Gemini "
            "structured output."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/2026"),
        help="Root folder containing routine subfolders with *_transcript.json files.",
    )
    parser.add_argument(
        "--glob",
        default="*_transcript.json",
        help="Glob pattern used with recursive search inside --input-root.",
    )
    parser.add_argument(
        "--ids",
        default="",
        help=(
            "Comma-separated routine ids to process (matches folder name or filename "
            "prefix). Example: 136,99,171"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files to process after filtering.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Gemini generation.",
    )
    parser.add_argument(
        "--json-retries",
        type=int,
        default=2,
        help=(
            "Number of retries for a window when Gemini returns invalid JSON "
            "(default: 2)."
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=9000,
        help="Window size in characters per request.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=250,
        help="Character overlap added from the next segment between windows.",
    )
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help=(
            "Environment variable name for Gemini API key. Falls back to "
            "GOOGLE_API_KEY if empty."
        ),
    )
    parser.add_argument(
        "--output-filename",
        default="jokes.json",
        help=(
            "Output file name written next to each transcript file "
            "(default: jokes.json)."
        ),
    )
    parser.add_argument(
        "--no-write-output",
        action="store_true",
        help="Do not write JSON output files; print results only.",
    )
    parser.add_argument(
        "--max-jokes-print",
        type=int,
        default=5,
        help="Maximum number of joke previews printed per routine.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between files to avoid API burst limits.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first file error.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess routines even if output file already exists.",
    )
    parser.add_argument(
        "--disable-locking",
        action="store_true",
        help="Disable per-file lock claiming (not recommended for parallel runs).",
    )
    parser.add_argument(
        "--lock-stale-seconds",
        type=float,
        default=21600.0,
        help=(
            "Consider lock files older than this as stale and reclaim them "
            "(default: 21600 = 6 hours). Set 0 to never reclaim."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call Gemini. Only validate files and window partitioning.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    step = max(1, max_chars - max(0, overlap_chars))
    start = 0
    while start < len(text):
        chunk = text[start : start + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        if start + max_chars >= len(text):
            break
        start += step
    return chunks


def build_segment_windows(
    segments: List[Dict[str, Any]],
    max_chars: int = 9000,
    overlap_chars: int = 250,
) -> List[str]:
    texts = [normalize_text(seg.get("text", "")) for seg in segments if seg.get("text")]

    windows: List[str] = []
    buffer_parts: List[str] = []
    buffer_len = 0

    for segment in texts:
        if len(segment) > max_chars:
            if buffer_parts:
                overlap = segment[:overlap_chars].strip()
                window = " ".join(buffer_parts).strip()
                if overlap:
                    window = f"{window}\n\n[OVERLAP]\n{overlap}"
                if window:
                    windows.append(window)
                buffer_parts = []
                buffer_len = 0
            windows.extend(split_long_text(segment, max_chars, overlap_chars))
            continue

        extra_len = len(segment) + (1 if buffer_parts else 0)
        if buffer_parts and (buffer_len + extra_len > max_chars):
            overlap = segment[:overlap_chars].strip()
            window = " ".join(buffer_parts).strip()
            if overlap:
                window = f"{window}\n\n[OVERLAP]\n{overlap}"
            if window:
                windows.append(window)
            buffer_parts = [segment]
            buffer_len = len(segment)
            continue

        buffer_parts.append(segment)
        buffer_len += extra_len

    if buffer_parts:
        window = " ".join(buffer_parts).strip()
        if window:
            windows.append(window)

    return windows


def call_gemini_structured(
    text_window: str,
    client: Any,
    model: str,
    temperature: float,
    json_retries: int,
) -> Repertoire:
    prompt = f"{USER_TASK}\n\n{text_window}\n\nRespond in Spanish."
    retries = max(0, json_retries)
    last_error: Exception | None = None

    for attempt in range(1, retries + 2):
        contents = [
            {"role": "user", "parts": [{"text": SYSTEM_INSTRUCTIONS}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ]
        if attempt > 1:
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Your previous output was invalid JSON. Return ONLY valid JSON "
                                "for the same content. Prefer « » in dialogues and avoid raw "
                                "double quotes inside string values."
                            )
                        }
                    ],
                }
            )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config={
                "temperature": temperature,
                "response_mime_type": "application/json",
                "response_json_schema": Repertoire.model_json_schema(),
            },
        )

        response_text = response.text or ""
        if not response_text.strip():
            last_error = ValueError("Gemini returned an empty response.")
            continue

        try:
            return Repertoire.model_validate_json(response_text)
        except Exception as exc:
            last_error = exc
            continue

    assert last_error is not None
    raise last_error


def fingerprint(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    return text[:200]


def tokenize_compare(text: str) -> List[str]:
    return re.findall(r"[a-z0-9áéíóúüñ]+", text.lower())


def select_minimal_cleaned_text(item: JokeItem) -> str:
    original = normalize_text(item.transcript)
    cleaned = normalize_text(item.cleaned_transcript)

    if not cleaned:
        return original
    if not original:
        return cleaned
    if cleaned == original:
        return cleaned

    original_tokens = tokenize_compare(original)
    cleaned_tokens = tokenize_compare(cleaned)
    if not original_tokens or not cleaned_tokens:
        return original

    len_ratio = len(cleaned_tokens) / max(1, len(original_tokens))
    if len_ratio < 0.85 or len_ratio > 1.20:
        return original

    original_vocab = set(original_tokens)
    cleaned_vocab = set(cleaned_tokens)
    vocab_retention = len(original_vocab & cleaned_vocab) / max(1, len(original_vocab))
    if vocab_retention < 0.72:
        return original

    similarity = difflib.SequenceMatcher(
        a=original.lower(),
        b=cleaned.lower(),
    ).ratio()
    if similarity < 0.78:
        return original

    return cleaned


def postprocess(items: List[JokeItem]) -> List[str]:
    seen = set()
    jokes: List[str] = []

    for item in items:
        if item.kind != "joke":
            continue
        cleaned = select_minimal_cleaned_text(item)
        if not cleaned:
            continue
        fp = fingerprint(cleaned)
        if fp in seen:
            continue
        seen.add(fp)
        jokes.append(cleaned)

    return jokes


def extract_jokes_from_transcript_json(
    transcript_json: Dict[str, Any],
    client: Any,
    model: str,
    max_chars: int,
    overlap_chars: int,
    temperature: float,
    json_retries: int,
) -> tuple[List[str], int, int]:
    segments = transcript_json.get("segments", [])
    windows = build_segment_windows(
        segments=segments,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    all_items: List[JokeItem] = []
    failed_windows = 0
    for window_idx, window in enumerate(windows, start=1):
        try:
            repertoire = call_gemini_structured(
                text_window=window,
                client=client,
                model=model,
                temperature=temperature,
                json_retries=json_retries,
            )
            all_items.extend(repertoire.jokes)
        except Exception as exc:
            failed_windows += 1
            print(
                f"  window_error | {window_idx}/{len(windows)} | {summarize_error(exc)}",
                file=sys.stderr,
            )

    jokes = postprocess(all_items)
    return jokes, len(windows), failed_windows


def parse_ids(raw_ids: str) -> set[str]:
    if not raw_ids.strip():
        return set()
    return {token.strip() for token in raw_ids.split(",") if token.strip()}


def discover_transcript_files(
    input_root: Path,
    pattern: str,
    routine_ids: set[str],
) -> List[Path]:
    files = sorted(input_root.rglob(pattern))
    if not routine_ids:
        return files
    filtered: List[Path] = []
    for path in files:
        stem_id = path.stem.split("_", 1)[0]
        parent_id = path.parent.name
        if stem_id in routine_ids or parent_id in routine_ids:
            filtered.append(path)
    return filtered


def shorten_preview(text: str, max_chars: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def summarize_error(exc: Exception, max_chars: int = 260) -> str:
    text = " ".join(str(exc).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def lock_file_for_output(source_file: Path, output_filename: str) -> Path:
    return source_file.parent / f".{output_filename}.lock"


def try_acquire_lock(lock_path: Path, stale_seconds: float) -> tuple[bool, str]:
    def create_lock_file() -> bool:
        try:
            fd = os.open(
                str(lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            return False

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "created_at_unix": time.time(),
                    }
                )
            )
        return True

    if create_lock_file():
        return True, "acquired"

    if stale_seconds > 0:
        try:
            age = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            if create_lock_file():
                return True, "acquired"
            return False, f"lock busy: {lock_path.name}"

        if age >= stale_seconds:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return False, f"lock busy: {lock_path.name}"
            if create_lock_file():
                return True, f"reclaimed stale lock ({age:.0f}s)"

    return False, f"lock busy: {lock_path.name}"


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def write_routine_output(
    source_file: Path,
    output_filename: str,
    model: str,
    window_count: int,
    failed_window_count: int,
    jokes: List[str],
) -> Path:
    output_path = source_file.parent / output_filename

    payload = {
        "source_file": str(source_file),
        "model": model,
        "window_count": window_count,
        "failed_window_count": failed_window_count,
        "joke_count": len(jokes),
        "jokes": jokes,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    args = parse_args()

    if args.max_chars <= 0:
        raise ValueError("--max-chars must be > 0")
    if args.json_retries < 0:
        raise ValueError("--json-retries must be >= 0")
    if args.lock_stale_seconds < 0:
        raise ValueError("--lock-stale-seconds must be >= 0")
    if args.overlap_chars < 0:
        raise ValueError("--overlap-chars must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0 when provided")

    input_root = args.input_root
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    routine_ids = parse_ids(args.ids)
    files = discover_transcript_files(
        input_root=input_root,
        pattern=args.glob,
        routine_ids=routine_ids,
    )
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print("No transcript files found with current filters.")
        return

    print(f"Found {len(files)} transcript file(s).")
    if routine_ids:
        print(f"Routine id filter: {sorted(routine_ids)}")

    client: Any = None
    if not args.dry_run and genai is None:
        raise RuntimeError(
            "google-genai is not installed. Run: pip install -r requirements.txt"
        )

    total_jokes = 0
    failed = 0
    skipped = 0
    locked = 0

    for idx, transcript_path in enumerate(files, start=1):
        abs_path = transcript_path.resolve()
        cwd = Path.cwd().resolve()
        try:
            rel_path: Path | str = abs_path.relative_to(cwd)
        except ValueError:
            rel_path = transcript_path
        print(f"\n[{idx}/{len(files)}] {rel_path}")
        lock_path = lock_file_for_output(
            source_file=transcript_path,
            output_filename=args.output_filename,
        )
        lock_acquired = False
        try:
            output_path = transcript_path.parent / args.output_filename
            if (
                not args.dry_run
                and not args.no_write_output
                and not args.force_reprocess
                and output_path.exists()
            ):
                skipped += 1
                print(f"  skipped | already exists: {output_path}")
                continue

            if not args.dry_run and not args.disable_locking:
                lock_acquired, lock_msg = try_acquire_lock(
                    lock_path=lock_path,
                    stale_seconds=args.lock_stale_seconds,
                )
                if not lock_acquired:
                    locked += 1
                    print(f"  skipped | {lock_msg}")
                    continue

            with transcript_path.open("r", encoding="utf-8") as f:
                transcript_data = json.load(f)

            segments = transcript_data.get("segments", [])
            windows = build_segment_windows(
                segments=segments,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
            )

            if args.dry_run:
                print(
                    "  dry-run | "
                    f"segments={len(segments)} windows={len(windows)}"
                )
                continue

            if client is None:
                api_key = os.getenv(args.api_key_env) or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        f"Missing API key. Set {args.api_key_env} (or GOOGLE_API_KEY) in .env."
                    )
                client = genai.Client(api_key=api_key)

            started = time.perf_counter()
            jokes, window_count, failed_window_count = extract_jokes_from_transcript_json(
                transcript_json=transcript_data,
                client=client,
                model=args.model,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
                temperature=args.temperature,
                json_retries=args.json_retries,
            )
            elapsed = time.perf_counter() - started

            if failed_window_count == window_count and window_count > 0:
                raise RuntimeError(
                    "All windows failed for this file; skipping output."
                )

            total_jokes += len(jokes)
            print(
                "  success | "
                f"segments={len(segments)} windows={window_count} jokes={len(jokes)} "
                f"failed_windows={failed_window_count} elapsed={elapsed:.1f}s"
            )

            for joke_idx, joke in enumerate(jokes[: args.max_jokes_print], start=1):
                print(f"  {joke_idx}. {shorten_preview(joke)}")
            remaining = len(jokes) - min(len(jokes), args.max_jokes_print)
            if remaining > 0:
                print(f"  ... +{remaining} joke(s) more")

            if not args.no_write_output:
                output_path = write_routine_output(
                    source_file=transcript_path,
                    output_filename=args.output_filename,
                    model=args.model,
                    window_count=window_count,
                    failed_window_count=failed_window_count,
                    jokes=jokes,
                )
                print(f"  wrote: {output_path}")

            if args.sleep_seconds > 0 and idx < len(files):
                time.sleep(args.sleep_seconds)

        except Exception as exc:
            failed += 1
            print(f"  error | {exc}", file=sys.stderr)
            if args.fail_fast:
                raise
        finally:
            if lock_acquired:
                release_lock(lock_path)

    processed = len(files) - failed - skipped - locked
    mode = "dry-run" if args.dry_run else "live"
    print(
        "\nFinished "
        f"({mode}): processed={processed} skipped={skipped} locked={locked} failed={failed} total_files={len(files)} "
        f"total_jokes={total_jokes}"
    )


if __name__ == "__main__":
    main()
