#!/usr/bin/env python3
"""Compute joke mentalizing analysis and persist results into jokes.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

load_dotenv()


MindVerb = Literal[
    "think",
    "believe",
    "know",
    "want",
    "intend",
    "assume",
    "hope",
    "feel",
    "pretend",
    "misunderstand",
    "other",
]


class Agent(BaseModel):
    id: str = Field(description="Stable short id, e.g., 'boy', 'barber', 'customer'.")
    name: str = Field(description="Display name as it appears in the joke.")


class MindState(BaseModel):
    id: str = Field(description="Unique id for this mindstate, e.g., 'm1'.")
    holder_id: str = Field(description="Agent id who holds the mental state.")
    target_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional agent id that is the target of the mental state "
            "(who the holder is thinking about)."
        ),
    )
    verb: MindVerb = Field(description="Type of mindstate verb (normalized).")
    content: str = Field(description="Short paraphrase of the mental state content.")
    parent_id: Optional[str] = Field(
        default=None,
        description=(
            "If embedded (A thinks [B thinks ...]), parent_id is the outer mindstate id."
        ),
    )
    story_depth: int = Field(
        ge=1,
        le=10,
        description="1 for first-order mindstate inside the story, 2 for embedded, etc.",
    )
    evidence_span: Optional[str] = Field(
        default=None,
        description="Short supporting snippet (<= 20 words).",
    )


class GraphEdge(BaseModel):
    source: str = Field(description="Node id, e.g., agent id 'boy' or mindstate id 'm1'.")
    target: str = Field(description="Node id.")
    label: str = Field(description="Edge label: holds/about/embedded_in.")


class JokeMentalizingAnalysis(BaseModel):
    agents: List[Agent] = Field(
        description="Main agents in the joke story (exclude comedian/audience)."
    )
    mindstates: List[MindState] = Field(
        description="Mindstates extracted from the story (exclude comedian/audience)."
    )
    summary: Dict[str, int] = Field(
        description=(
            "Must include integer keys: max_story_depth, total_intentionality_level, "
            "mindstate_count_story."
        )
    )
    edges: List[GraphEdge] = Field(description="Graph edges among agents and mindstates.")


SYSTEM_INSTRUCTIONS = """You analyze jokes for mentalizing complexity using orders of intentionality.

Counting rule (simple but consistent):
- The comedian-audience conversation frame ALWAYS counts as 3 levels of intentionality.
- Then add the MAX embedding depth of mindstates found inside the story of the joke.
  * A mindstate is a mental attitude: think/believe/know/want/intend/assume/hope/feel/pretend/misunderstand.
  * If A thinks (B thinks ...), that is embedded and increases depth.

Your tasks:
1) Identify the main agents/characters in the joke story.
2) Extract explicit or strongly implied mindstates:
   - holder: who has the mindstate
   - target: who it is about (optional)
   - verb: normalized mindstate type
   - content: brief paraphrase
   - parent_id: if this mindstate is embedded in another mindstate
   - story_depth: 1 for first-order, 2+ for embedded mindstates
   - evidence_span: <= 20 words supporting the extraction
3) Build a graph:
   - Nodes are agents and mindstates.
   - Edges:
     * holder -> mindstate (label='holds')
     * mindstate -> target (label='about') when target_id exists
     * mindstate -> parent mindstate (label='embedded_in') when parent_id exists
4) Compute summary (integers):
   - max_story_depth = maximum story_depth across extracted mindstates (0 if none)
   - mindstate_count_story = number of extracted mindstates
   - total_intentionality_level = 3 + max_story_depth

Constraints:
- Return ONLY valid JSON matching the provided schema.
- Do NOT include extra keys.

Respond in spanish.
"""


def build_prompt(joke_text: str) -> str:
    return f"""Analyze this joke:
\"\"\"{joke_text}\"\"\"

Respond in spanish.
""".strip()


def call_gemini_analysis(
    *,
    client: Any,
    model: str,
    joke_text: str,
    temperature: float,
    json_retries: int,
) -> JokeMentalizingAnalysis:
    prompt = build_prompt(joke_text)
    retries = max(0, json_retries)
    last_error: Exception | None = None
    schema = JokeMentalizingAnalysis.model_json_schema()

    for attempt in range(1, retries + 2):
        contents: List[Any] = [
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
                                "matching the schema."
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
                "response_json_schema": schema,
            },
        )

        response_text = (response.text or "").strip()
        if not response_text:
            last_error = ValueError("Gemini returned an empty response.")
            continue

        try:
            return JokeMentalizingAnalysis.model_validate_json(response_text)
        except Exception as exc:
            last_error = exc
            continue

    assert last_error is not None
    raise last_error


def compute_summary_from_mindstates(analysis: JokeMentalizingAnalysis) -> Dict[str, int]:
    max_story_depth = max((ms.story_depth for ms in analysis.mindstates), default=0)
    mindstate_count_story = len(analysis.mindstates)
    total_intentionality_level = 3 + max_story_depth
    return {
        "max_story_depth": int(max_story_depth),
        "mindstate_count_story": int(mindstate_count_story),
        "total_intentionality_level": int(total_intentionality_level),
    }


def analysis_to_dot(analysis: JokeMentalizingAnalysis) -> str:
    lines = ["digraph MentalizingGraph {", "  rankdir=LR;"]

    for a in analysis.agents:
        safe_label = a.name.replace('"', '\\"')
        lines.append(f'  "{a.id}" [shape=box, label="{safe_label}"];')

    for m in analysis.mindstates:
        safe_content = m.content.replace('"', '\\"')
        label = f"{m.id}\\n{m.verb}\\ndepth={m.story_depth}\\n{safe_content}"
        lines.append(f'  "{m.id}" [shape=ellipse, label="{label}"];')

    for e in analysis.edges:
        safe_label = e.label.replace('"', '\\"')
        lines.append(f'  "{e.source}" -> "{e.target}" [label="{safe_label}"];')

    s = compute_summary_from_mindstates(analysis)
    lines.append(
        f'  // max_story_depth={s["max_story_depth"]} '
        f'total_intentionality_level={s["total_intentionality_level"]} '
        f'mindstate_count_story={s["mindstate_count_story"]}'
    )
    lines.append("}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze each joke in jokes.json and persist mentalizing analysis "
            "into the same file."
        )
    )
    parser.add_argument("--input-root", type=Path, default=Path("data/2026"))
    parser.add_argument("--input-filename", default="jokes.json")
    parser.add_argument("--ids", default="", help="Optional comma-separated routine ids.")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--max-jokes", type=int, default=None)
    parser.add_argument("--model", default="gemini-3-pro-preview")
    parser.add_argument("--fallback-model", default="gemini-3-pro-preview")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json-retries", type=int, default=2)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--include-dot", action="store_true")
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--disable-locking", action="store_true")
    parser.add_argument("--lock-stale-seconds", type=float, default=21600.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_error(exc: Exception, max_chars: int = 320) -> str:
    text = " ".join(str(exc).split())
    return text if len(text) <= max_chars else (text[: max_chars - 3].rstrip() + "...")


def parse_ids(raw_ids: str) -> set[str]:
    if not raw_ids.strip():
        return set()
    return {tok.strip() for tok in raw_ids.split(",") if tok.strip()}


def discover_joke_files(input_root: Path, input_filename: str, routine_ids: set[str]) -> List[Path]:
    files = sorted(input_root.rglob(input_filename))
    if not routine_ids:
        return files
    return [p for p in files if p.parent.name in routine_ids]


def lock_file_for_jokes(jokes_path: Path) -> Path:
    return jokes_path.parent / f".{jokes_path.name}.mentalizing.lock"


def try_acquire_lock(lock_path: Path, stale_seconds: float) -> tuple[bool, str]:
    def create_lock_file() -> bool:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "created_at_unix": time.time()}))
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
            except OSError:
                return False, f"lock busy: {lock_path.name}"
            if create_lock_file():
                return True, f"reclaimed stale lock ({age:.0f}s)"

    return False, f"lock busy: {lock_path.name}"


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except OSError:
        return


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def normalize_mentalizing_results(raw: Any, joke_count: int) -> List[Any]:
    if isinstance(raw, list):
        out = list(raw[:joke_count])
        if len(out) < joke_count:
            out.extend([None] * (joke_count - len(out)))
        return out
    return [None] * joke_count


def is_success_entry(entry: Any) -> bool:
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        return False
    return isinstance(entry.get("total_intentionality_level"), int) or isinstance(
        entry.get("intentionality_depth"), int
    )


def should_process_entry(entry: Any, force_reprocess: bool) -> bool:
    return True if force_reprocess else (not is_success_entry(entry))


def is_probable_api_or_json_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = (
        "invalid json",
        "unterminated string",
        "eof while parsing",
        "json_invalid",
        "rate limit",
        "resource exhausted",
        "quota",
        "deadline exceeded",
        "service unavailable",
        "internal error",
        "timed out",
        "timeout",
        "too many requests",
        "503",
        "502",
        "504",
        "500",
        "429",
    )
    return any(m in msg for m in markers)


def count_pending_for_file(path: Path, force_reprocess: bool) -> tuple[int, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0

    jokes_raw = payload.get("jokes", [])
    if not isinstance(jokes_raw, list):
        return 0, 0

    results = normalize_mentalizing_results(payload.get("mentalizing_results"), len(jokes_raw))
    pending = 0
    for i in range(len(jokes_raw)):
        if should_process_entry(results[i], force_reprocess):
            pending += 1
    return pending, len(jokes_raw)


def update_summary_fields(payload: Dict[str, Any]) -> None:
    results = payload.get("mentalizing_results", [])
    if not isinstance(results, list):
        payload["mentalizing_completed_count"] = 0
        payload["mentalizing_error_count"] = 0
        return

    completed = 0
    errors = 0
    for entry in results:
        if isinstance(entry, dict):
            if entry.get("status") == "ok":
                completed += 1
            elif entry.get("status") == "error":
                errors += 1
    payload["mentalizing_completed_count"] = completed
    payload["mentalizing_error_count"] = errors


def main() -> None:
    args = parse_args()

    if args.json_retries < 0:
        raise ValueError("--json-retries must be >= 0")
    if args.limit_files is not None and args.limit_files <= 0:
        raise ValueError("--limit-files must be > 0 when provided")
    if args.max_jokes is not None and args.max_jokes <= 0:
        raise ValueError("--max-jokes must be > 0 when provided")
    if args.lock_stale_seconds < 0:
        raise ValueError("--lock-stale-seconds must be >= 0")

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    routine_ids = parse_ids(args.ids)
    files = discover_joke_files(args.input_root, args.input_filename, routine_ids)
    if args.limit_files is not None:
        files = files[: args.limit_files]

    if not files:
        print("No jokes files found with current filters.")
        return

    fallback_model = args.fallback_model.strip() if args.fallback_model else ""
    if fallback_model == args.model:
        fallback_model = ""

    pending_total = 0
    joke_total = 0
    pending_by_file: Dict[Path, int] = {}
    for jokes_path in files:
        pending, total = count_pending_for_file(jokes_path, args.force_reprocess)
        pending_by_file[jokes_path] = pending
        pending_total += pending
        joke_total += total

    print(f"Found {len(files)} jokes file(s).")
    if routine_ids:
        print(f"Routine id filter: {sorted(routine_ids)}")
    if fallback_model:
        print(f"Fallback model enabled: {fallback_model}")
    print(f"Snapshot: total_jokes={joke_total} pending_mentalizing={pending_total}")

    if not args.dry_run and genai is None:
        raise RuntimeError("google-genai is not installed. Run: pip install google-genai")

    client: Any = None
    stop_requested = False
    global_done = 0
    global_success = 0
    global_errors = 0
    files_locked = 0
    files_skipped = 0
    files_failed = 0

    for file_idx, jokes_path in enumerate(files, start=1):
        if stop_requested:
            break

        abs_path = jokes_path.resolve()
        cwd = Path.cwd().resolve()
        try:
            rel_path: Path | str = abs_path.relative_to(cwd)
        except ValueError:
            rel_path = jokes_path
        print(f"\n[{file_idx}/{len(files)}] {rel_path}")

        if pending_by_file.get(jokes_path, 0) == 0 and not args.force_reprocess:
            files_skipped += 1
            print("  skipped | no pending jokes")
            continue

        lock_path = lock_file_for_jokes(jokes_path)
        lock_acquired = False

        try:
            if not args.dry_run and not args.disable_locking:
                lock_acquired, lock_msg = try_acquire_lock(lock_path, args.lock_stale_seconds)
                if not lock_acquired:
                    files_locked += 1
                    print(f"  skipped | {lock_msg}")
                    continue

            try:
                payload: Dict[str, Any] = json.loads(jokes_path.read_text(encoding="utf-8"))
            except Exception as exc:
                files_failed += 1
                print(f"  error | invalid json file | {summarize_error(exc)}", file=sys.stderr)
                if args.fail_fast:
                    raise
                continue

            jokes_raw = payload.get("jokes")
            if not isinstance(jokes_raw, list):
                files_failed += 1
                print("  error | jokes field is missing or not a list", file=sys.stderr)
                if args.fail_fast:
                    raise RuntimeError("jokes field is missing or not a list")
                continue

            jokes: List[str] = [str(j).strip() for j in jokes_raw]
            results = normalize_mentalizing_results(payload.get("mentalizing_results"), len(jokes))
            pending_indices = [
                i for i in range(len(jokes)) if should_process_entry(results[i], args.force_reprocess)
            ]
            if not pending_indices:
                files_skipped += 1
                print("  skipped | no pending jokes")
                continue

            print(f"  plan | pending={len(pending_indices)}/{len(jokes)}")

            for local_pos, joke_idx in enumerate(pending_indices, start=1):
                if args.max_jokes is not None and global_done >= args.max_jokes:
                    stop_requested = True
                    break

                joke_text = jokes[joke_idx]
                global_pos = global_done + 1
                global_den = pending_total if pending_total > 0 else "?"
                print(
                    f"  progress | joke {local_pos}/{len(pending_indices)} | "
                    f"global {global_pos}/{global_den} | idx={joke_idx+1}/{len(jokes)}"
                )

                if args.dry_run:
                    global_done += 1
                    continue

                if client is None:
                    api_key = os.getenv(args.api_key_env) or os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        raise RuntimeError(
                            f"Missing API key. Set {args.api_key_env} (or GOOGLE_API_KEY) in .env."
                        )
                    client = genai.Client(api_key=api_key)

                used_model = args.model
                started = time.perf_counter()

                try:
                    analysis = call_gemini_analysis(
                        client=client,
                        model=args.model,
                        joke_text=joke_text,
                        temperature=args.temperature,
                        json_retries=args.json_retries,
                    )
                except Exception as primary_exc:
                    if fallback_model and is_probable_api_or_json_error(primary_exc):
                        try:
                            print(
                                f"  retry_fallback | idx={joke_idx+1} | model={fallback_model}",
                                file=sys.stderr,
                            )
                            analysis = call_gemini_analysis(
                                client=client,
                                model=fallback_model,
                                joke_text=joke_text,
                                temperature=args.temperature,
                                json_retries=args.json_retries,
                            )
                            used_model = fallback_model
                        except Exception as fallback_exc:
                            elapsed = time.perf_counter() - started
                            err_text = summarize_error(fallback_exc)
                            results[joke_idx] = {
                                "status": "error",
                                "joke_index": joke_idx,
                                "model_used": used_model,
                                "error": err_text,
                                "updated_at_utc": now_utc_iso(),
                            }
                            global_errors += 1
                            global_done += 1
                            print(
                                f"  error | idx={joke_idx+1} elapsed={elapsed:.1f}s | {err_text}",
                                file=sys.stderr,
                            )
                            if args.fail_fast:
                                raise
                            payload["mentalizing_results"] = results
                            payload["mentalizing_model"] = args.model
                            payload["mentalizing_fallback_model"] = fallback_model or None
                            payload["mentalizing_temperature"] = args.temperature
                            payload["mentalizing_last_updated_utc"] = now_utc_iso()
                            update_summary_fields(payload)
                            atomic_write_json(jokes_path, payload)
                            if args.sleep_seconds > 0:
                                time.sleep(args.sleep_seconds)
                            continue
                    else:
                        elapsed = time.perf_counter() - started
                        err_text = summarize_error(primary_exc)
                        results[joke_idx] = {
                            "status": "error",
                            "joke_index": joke_idx,
                            "model_used": used_model,
                            "error": err_text,
                            "updated_at_utc": now_utc_iso(),
                        }
                        global_errors += 1
                        global_done += 1
                        print(
                            f"  error | idx={joke_idx+1} elapsed={elapsed:.1f}s | {err_text}",
                            file=sys.stderr,
                        )
                        if args.fail_fast:
                            raise
                        payload["mentalizing_results"] = results
                        payload["mentalizing_model"] = args.model
                        payload["mentalizing_fallback_model"] = fallback_model or None
                        payload["mentalizing_temperature"] = args.temperature
                        payload["mentalizing_last_updated_utc"] = now_utc_iso()
                        update_summary_fields(payload)
                        atomic_write_json(jokes_path, payload)
                        if args.sleep_seconds > 0:
                            time.sleep(args.sleep_seconds)
                        continue

                summary_fixed = compute_summary_from_mindstates(analysis)
                analysis_dict = analysis.model_dump(mode="json")
                analysis_dict["summary"] = summary_fixed
                elapsed = time.perf_counter() - started

                result_entry: Dict[str, Any] = {
                    "status": "ok",
                    "joke_index": joke_idx,
                    "model_used": used_model,
                    "max_story_depth": summary_fixed["max_story_depth"],
                    "mindstate_count_story": summary_fixed["mindstate_count_story"],
                    "total_intentionality_level": summary_fixed["total_intentionality_level"],
                    "analysis": analysis_dict,
                    "updated_at_utc": now_utc_iso(),
                }
                if args.include_dot:
                    result_entry["graph_dot"] = analysis_to_dot(analysis)

                results[joke_idx] = result_entry
                global_success += 1
                global_done += 1
                print(
                    "  ok | "
                    f"idx={joke_idx+1} total_intentionality={summary_fixed['total_intentionality_level']} "
                    f"model={used_model} elapsed={elapsed:.1f}s"
                )

                payload["mentalizing_results"] = results
                payload["mentalizing_model"] = args.model
                payload["mentalizing_fallback_model"] = fallback_model or None
                payload["mentalizing_temperature"] = args.temperature
                payload["mentalizing_last_updated_utc"] = now_utc_iso()
                update_summary_fields(payload)
                atomic_write_json(jokes_path, payload)

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

        except Exception:
            files_failed += 1
            if args.fail_fast:
                raise
        finally:
            if lock_acquired:
                release_lock(lock_path)

    mode = "dry-run" if args.dry_run else "live"
    print(
        "\nFinished "
        f"({mode}): jokes_done={global_done} success={global_success} errors={global_errors} "
        f"files_locked={files_locked} files_skipped={files_skipped} files_failed={files_failed}"
    )


if __name__ == "__main__":
    main()
