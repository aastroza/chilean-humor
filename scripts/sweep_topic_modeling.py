#!/usr/bin/env python3
"""Run BERTopic hyperparameter sweeps and rank runs with comparable metrics."""

from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SweepConfig:
    umap_n_neighbors: int
    umap_min_dist: float
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int
    hdbscan_cluster_selection_method: str
    seed: int


def parse_int_list(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_float_list(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def parse_str_list(raw: str) -> list[str]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(token)
    if not values:
        raise ValueError("Expected at least one string value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep UMAP + HDBSCAN settings for topic modeling, then rank runs with "
            "metrics including multi-topic share."
        )
    )
    parser.add_argument(
        "--repo-id",
        default="astroza/chilean-humor-jokes",
        help="HF dataset repo id.",
    )
    parser.add_argument(
        "--config-name",
        default="jokes",
        help="Dataset config name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/topic_modeling_sweep"),
        help="Root output folder for sweep artifacts and per-run outputs.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        default="10,20,30",
        help="Comma-separated values for UMAP n_neighbors.",
    )
    parser.add_argument(
        "--umap-min-dist",
        default="0.0,0.05,0.1",
        help="Comma-separated values for UMAP min_dist.",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        default="10,15,22,30",
        help="Comma-separated values for HDBSCAN min_cluster_size.",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        default="1,3,6,10",
        help="Comma-separated values for HDBSCAN min_samples.",
    )
    parser.add_argument(
        "--hdbscan-cluster-selection-method",
        default="eom,leaf",
        help="Comma-separated values for HDBSCAN cluster_selection_method.",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on number of run configurations to execute.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip execution if a run already has run_report.json (still included in ranking).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop sweep on first failed run.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size to pass to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--multi-topic-thresholds",
        default="0.05,0.08,0.10",
        help="Comma-separated thresholds used for multi-topic metrics from segment_topic_probs_long.csv.",
    )
    parser.add_argument(
        "--rank-multi-topic-threshold",
        type=float,
        default=0.08,
        help="Threshold used as multi-topic signal in final score.",
    )
    parser.add_argument(
        "--topic-words-k",
        type=int,
        default=10,
        help="Number of top words used to estimate topic diversity.",
    )
    parser.add_argument(
        "--target-topic-count",
        type=float,
        default=30.0,
        help="Target number of topics for topic-count score component.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=10,
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--min-text-tokens",
        type=int,
        default=2,
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--use-jina-embeddings",
        action="store_true",
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--jina-provider",
        choices=["local", "api"],
        default="local",
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--jina-model-name",
        default="jinaai/jina-embeddings-v4",
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--jina-task",
        default="text-matching",
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--jina-truncate-dim",
        type=int,
        default=128,
        help="Forwarded to run_topic_modeling.py. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--jina-batch-size",
        type=int,
        default=16,
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--jina-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--jina-api-url",
        default="https://api.jina.ai/v1/embeddings",
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--jina-api-token-env",
        default="JINA_API_TOKEN",
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--jina-api-timeout-seconds",
        type=float,
        default=60.0,
        help="Forwarded to run_topic_modeling.py.",
    )
    parser.add_argument(
        "--jina-cache-dir",
        type=Path,
        default=None,
        help="Shared embedding cache dir across runs (recommended when sweeping).",
    )
    return parser.parse_args()


def threshold_key(value: float) -> str:
    return f"t{value:.3f}".rstrip("0").rstrip(".").replace(".", "_")


def run_slug(cfg: SweepConfig) -> str:
    min_dist = str(cfg.umap_min_dist).replace(".", "p")
    return (
        f"u{cfg.umap_n_neighbors}_d{min_dist}_"
        f"cs{cfg.hdbscan_min_cluster_size}_"
        f"ms{cfg.hdbscan_min_samples}_"
        f"{cfg.hdbscan_cluster_selection_method}_"
        f"s{cfg.seed}"
    )


def config_slug(cfg: SweepConfig) -> str:
    min_dist = str(cfg.umap_min_dist).replace(".", "p")
    return (
        f"u{cfg.umap_n_neighbors}_d{min_dist}_"
        f"cs{cfg.hdbscan_min_cluster_size}_"
        f"ms{cfg.hdbscan_min_samples}_"
        f"{cfg.hdbscan_cluster_selection_method}"
    )


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def extract_topic_words(row: pd.Series, top_k: int) -> list[str]:
    representation = row.get("Representation")
    if isinstance(representation, str) and representation.strip():
        try:
            parsed = ast.literal_eval(representation)
            if isinstance(parsed, list):
                words = [str(item).strip() for item in parsed if str(item).strip()]
                if words:
                    return words[:top_k]
        except Exception:
            pass

    name = str(row.get("Name", "")).strip()
    if "_" in name:
        parts = [token.strip() for token in name.split("_")[1:] if token.strip()]
        if parts:
            return parts[:top_k]

    return []


def compute_topic_diversity(topic_info_path: Path, top_k: int) -> float | None:
    if not topic_info_path.exists():
        return None
    df = pd.read_csv(topic_info_path)
    if "Topic" not in df.columns:
        return None
    df = df[pd.to_numeric(df["Topic"], errors="coerce").notna()].copy()
    df["Topic"] = df["Topic"].astype(int)
    df = df[df["Topic"] != -1].copy()
    if df.empty:
        return None

    words: list[str] = []
    for _, row in df.iterrows():
        words.extend(extract_topic_words(row, top_k))
    if not words:
        return None

    unique_words = len(set(words))
    total_words = len(words)
    if total_words == 0:
        return None
    return unique_words / total_words


def compute_multi_topic_metrics(
    probs_path: Path, thresholds: list[float]
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    if not probs_path.exists():
        return metrics

    probs = pd.read_csv(probs_path, usecols=["segment_idx", "p_topic"])
    probs["segment_idx"] = pd.to_numeric(probs["segment_idx"], errors="coerce")
    probs["p_topic"] = pd.to_numeric(probs["p_topic"], errors="coerce")
    probs = probs.dropna(subset=["segment_idx", "p_topic"]).copy()
    if probs.empty:
        return metrics

    probs["segment_idx"] = probs["segment_idx"].astype(int)
    for threshold in thresholds:
        key = threshold_key(threshold)
        tmp = probs[probs["p_topic"] >= float(threshold)].copy()
        n_soft = int(tmp["segment_idx"].nunique())
        n_multi = 0
        if n_soft > 0:
            counts = tmp.groupby("segment_idx").size()
            n_multi = int((counts >= 2).sum())
        share = float(n_multi / n_soft) if n_soft > 0 else 0.0
        metrics[f"n_soft_{key}"] = n_soft
        metrics[f"n_multi_{key}"] = n_multi
        metrics[f"multi_share_{key}"] = share
    return metrics


def run_single(
    cfg: SweepConfig,
    args: argparse.Namespace,
    threshold_values: list[float],
) -> dict[str, Any]:
    run_id = run_slug(cfg)
    run_dir = args.output_root / "runs" / run_id
    report_path = run_dir / "run_report.json"
    started = time.perf_counter()

    status = "success"
    error_text = ""
    reused_existing = False

    if args.skip_existing and report_path.exists():
        reused_existing = True
    else:
        cmd = [
            sys.executable,
            "scripts/run_topic_modeling.py",
            "--repo-id",
            args.repo_id,
            "--config-name",
            args.config_name,
            "--split",
            args.split,
            "--output-dir",
            str(args.output_root),
            "--run-id",
            run_id,
            "--seed",
            str(cfg.seed),
            "--min-text-chars",
            str(args.min_text_chars),
            "--min-text-tokens",
            str(args.min_text_tokens),
            "--embedding-model",
            args.embedding_model,
            "--umap-n-neighbors",
            str(cfg.umap_n_neighbors),
            "--umap-min-dist",
            str(cfg.umap_min_dist),
            "--hdbscan-min-cluster-size",
            str(cfg.hdbscan_min_cluster_size),
            "--hdbscan-min-samples",
            str(cfg.hdbscan_min_samples),
            "--hdbscan-cluster-selection-method",
            cfg.hdbscan_cluster_selection_method,
            "--selected-topics",
            "",
            "--quiet",
        ]
        if args.sample_size is not None:
            cmd.extend(["--sample-size", str(args.sample_size)])

        if args.use_jina_embeddings:
            cmd.extend(
                [
                    "--use-jina-embeddings",
                    "--jina-provider",
                    args.jina_provider,
                    "--jina-model-name",
                    args.jina_model_name,
                    "--jina-task",
                    args.jina_task,
                    "--jina-batch-size",
                    str(args.jina_batch_size),
                    "--jina-device",
                    args.jina_device,
                    "--jina-api-url",
                    args.jina_api_url,
                    "--jina-api-token-env",
                    args.jina_api_token_env,
                    "--jina-api-timeout-seconds",
                    str(args.jina_api_timeout_seconds),
                ]
            )
            if args.jina_truncate_dim > 0:
                cmd.extend(["--jina-truncate-dim", str(args.jina_truncate_dim)])
            if args.jina_cache_dir is not None:
                cmd.extend(["--jina-cache-dir", str(args.jina_cache_dir)])

        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            status = "failed"
            tail = (completed.stderr or completed.stdout or "").splitlines()[-10:]
            error_text = " | ".join(line.strip() for line in tail if line.strip())
            if not error_text:
                error_text = f"run_topic_modeling.py exited with code {completed.returncode}"

    elapsed_seconds = time.perf_counter() - started
    row: dict[str, Any] = {
        "config_id": config_slug(cfg),
        "run_id": run_id,
        "status": status,
        "reused_existing": reused_existing,
        "elapsed_seconds": elapsed_seconds,
        "umap_n_neighbors": cfg.umap_n_neighbors,
        "umap_min_dist": cfg.umap_min_dist,
        "hdbscan_min_cluster_size": cfg.hdbscan_min_cluster_size,
        "hdbscan_min_samples": cfg.hdbscan_min_samples,
        "hdbscan_cluster_selection_method": cfg.hdbscan_cluster_selection_method,
        "seed": cfg.seed,
        "run_dir": str(run_dir),
        "error": error_text,
    }
    if status == "failed":
        return row

    if not report_path.exists():
        row["status"] = "failed_missing_report"
        row["error"] = f"Missing run report: {report_path}"
        return row

    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifacts = report.get("artifacts", {})
    diagnostics = report.get("diagnostics", {})

    topic_info_path = Path(artifacts.get("topic_info_csv") or run_dir / "tables" / "topic_info.csv")
    probs_path = Path(
        artifacts.get("segment_topic_probs_long_csv")
        or run_dir / "tables" / "segment_topic_probs_long.csv"
    )

    row["num_documents_modeled"] = safe_int(report.get("num_documents_modeled"))
    row["num_topics_found_raw"] = safe_int(report.get("num_topics_found"))
    row["n_topics_final"] = safe_int(diagnostics.get("n_topics_final"))
    row["median_topic_size"] = safe_float(diagnostics.get("median_topic_size"))
    row["p90_topic_size"] = safe_float(diagnostics.get("p90_topic_size"))
    row["outlier_rate_initial"] = safe_float(diagnostics.get("outlier_rate_initial"))
    row["outlier_rate_final"] = safe_float(diagnostics.get("outlier_rate_final"))
    row["topic_diversity_topk"] = compute_topic_diversity(
        topic_info_path=topic_info_path, top_k=args.topic_words_k
    )

    row.update(compute_multi_topic_metrics(probs_path, threshold_values))
    return row


def minmax_scale(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series([math.nan] * len(series), index=series.index)
    min_v = values.min()
    max_v = values.max()
    if pd.isna(min_v) or pd.isna(max_v):
        return pd.Series([math.nan] * len(series), index=series.index)
    if math.isclose(float(min_v), float(max_v), rel_tol=1e-12, abs_tol=1e-12):
        return pd.Series([0.5] * len(series), index=series.index)
    return (values - min_v) / (max_v - min_v)


def topic_count_score(series: pd.Series, target: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = []
    for value in values:
        if pd.isna(value):
            out.append(math.nan)
            continue
        distance = abs(float(value) - target) / max(1.0, target)
        out.append(math.exp(-distance))
    return pd.Series(out, index=series.index)


def aggregate_by_config(
    ranked_runs: pd.DataFrame,
    rank_multi_topic_threshold: float,
    target_topic_count: float,
) -> pd.DataFrame:
    rt_key = threshold_key(rank_multi_topic_threshold)
    multi_col = f"multi_share_{rt_key}"

    successful = ranked_runs[ranked_runs["status"] == "success"].copy()
    if successful.empty:
        return pd.DataFrame()

    group_cols = [
        "config_id",
        "umap_n_neighbors",
        "umap_min_dist",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "hdbscan_cluster_selection_method",
    ]

    agg_spec: dict[str, list[str]] = {
        "seed": ["count", "nunique"],
        "score": ["mean", "std"],
        "topic_diversity_topk": ["mean", "std"],
        "outlier_rate_final": ["mean", "std"],
        "n_topics_final": ["mean", "std"],
        multi_col: ["mean", "std"],
        "elapsed_seconds": ["mean"],
    }

    grouped = successful.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
    grouped.columns = [
        "_".join(part for part in col if part).rstrip("_") for col in grouped.columns
    ]
    grouped = grouped.rename(
        columns={
            "seed_count": "n_runs_success",
            "seed_nunique": "n_seeds_success",
            "score_mean": "score_mean_runs",
            "score_std": "score_std_runs",
            "topic_diversity_topk_mean": "topic_diversity_mean",
            "topic_diversity_topk_std": "topic_diversity_std",
            "outlier_rate_final_mean": "outlier_rate_final_mean",
            "outlier_rate_final_std": "outlier_rate_final_std",
            "n_topics_final_mean": "n_topics_final_mean",
            "n_topics_final_std": "n_topics_final_std",
            f"{multi_col}_mean": f"{multi_col}_mean",
            f"{multi_col}_std": f"{multi_col}_std",
            "elapsed_seconds_mean": "elapsed_seconds_mean",
        }
    )

    grouped["score_topic_diversity"] = minmax_scale(grouped["topic_diversity_mean"])
    grouped["score_multi_topic"] = minmax_scale(grouped[f"{multi_col}_mean"])
    grouped["score_outlier"] = 1.0 - minmax_scale(grouped["outlier_rate_final_mean"])
    grouped["score_topic_count"] = topic_count_score(
        grouped["n_topics_final_mean"], target_topic_count
    )

    score_std_series = pd.to_numeric(grouped["score_std_runs"], errors="coerce")
    grouped["score_stability"] = 1.0 - minmax_scale(score_std_series.fillna(score_std_series.max()))
    grouped["score_stability"] = grouped["score_stability"].fillna(0.5)

    grouped["score_config"] = (
        0.30 * grouped["score_topic_diversity"].fillna(0.0)
        + 0.25 * grouped["score_multi_topic"].fillna(0.0)
        + 0.20 * grouped["score_outlier"].fillna(0.0)
        + 0.15 * grouped["score_topic_count"].fillna(0.0)
        + 0.10 * grouped["score_stability"].fillna(0.0)
    )

    grouped = grouped.sort_values(
        by=["score_config", "topic_diversity_mean", f"{multi_col}_mean"],
        ascending=[False, False, False],
        na_position="last",
    )
    return grouped


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    neighbors = parse_int_list(args.umap_n_neighbors)
    min_dists = parse_float_list(args.umap_min_dist)
    cluster_sizes = parse_int_list(args.hdbscan_min_cluster_size)
    min_samples = parse_int_list(args.hdbscan_min_samples)
    methods = parse_str_list(args.hdbscan_cluster_selection_method)
    seeds = parse_int_list(args.seeds)
    threshold_values = parse_float_list(args.multi_topic_thresholds)

    configs = [
        SweepConfig(*values)
        for values in product(
            neighbors,
            min_dists,
            cluster_sizes,
            min_samples,
            methods,
            seeds,
        )
    ]
    if args.max_runs is not None:
        configs = configs[: args.max_runs]

    print(f"Planned runs: {len(configs)}")
    rows: list[dict[str, Any]] = []
    for index, cfg in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] "
            f"u={cfg.umap_n_neighbors} md={cfg.umap_min_dist} "
            f"cs={cfg.hdbscan_min_cluster_size} ms={cfg.hdbscan_min_samples} "
            f"method={cfg.hdbscan_cluster_selection_method} seed={cfg.seed}"
        )
        row = run_single(cfg=cfg, args=args, threshold_values=threshold_values)
        rows.append(row)
        if row["status"].startswith("failed"):
            print(f"  failed: {row.get('error', '')}")
            if args.fail_fast:
                break
        else:
            rt_key = threshold_key(args.rank_multi_topic_threshold)
            multi_share = row.get(f"multi_share_{rt_key}")
            print(
                "  success: "
                f"topics={row.get('n_topics_final')} "
                f"outliers={row.get('outlier_rate_final')} "
                f"multi_share@{args.rank_multi_topic_threshold}={multi_share}"
            )

    df = pd.DataFrame(rows)
    leaderboard_path = args.output_root / "leaderboard.csv"

    if df.empty:
        df.to_csv(leaderboard_path, index=False)
        print(f"No rows collected. Wrote empty leaderboard: {leaderboard_path}")
        return

    rt_key = threshold_key(args.rank_multi_topic_threshold)
    multi_col = f"multi_share_{rt_key}"
    if multi_col not in df.columns:
        df[multi_col] = math.nan

    successful = df["status"] == "success"
    df["score_topic_diversity"] = minmax_scale(df["topic_diversity_topk"])
    df["score_multi_topic"] = minmax_scale(df[multi_col])
    df["score_outlier"] = 1.0 - minmax_scale(df["outlier_rate_final"])
    df["score_topic_count"] = topic_count_score(df["n_topics_final"], args.target_topic_count)

    df["score"] = (
        0.35 * df["score_topic_diversity"].fillna(0.0)
        + 0.30 * df["score_multi_topic"].fillna(0.0)
        + 0.20 * df["score_outlier"].fillna(0.0)
        + 0.15 * df["score_topic_count"].fillna(0.0)
    )
    df.loc[~successful, "score"] = math.nan

    ranked = df.sort_values(
        by=["score", "topic_diversity_topk", multi_col],
        ascending=[False, False, False],
        na_position="last",
    )
    ranked.to_csv(leaderboard_path, index=False)

    config_leaderboard = aggregate_by_config(
        ranked_runs=ranked,
        rank_multi_topic_threshold=args.rank_multi_topic_threshold,
        target_topic_count=args.target_topic_count,
    )
    config_leaderboard_path = args.output_root / "leaderboard_configs.csv"
    config_leaderboard.to_csv(config_leaderboard_path, index=False)

    best_path = args.output_root / "best_config.json"
    best_row = ranked[ranked["status"] == "success"].head(1)
    if not best_row.empty:
        best_dict = best_row.iloc[0].to_dict()
        best_path.write_text(json.dumps(best_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Best run: {best_dict.get('run_id')} (score={best_dict.get('score'):.4f})")
        print(f"Best config saved to: {best_path}")
    else:
        print("No successful runs to rank.")

    best_cfg_path = args.output_root / "best_config_aggregated.json"
    if not config_leaderboard.empty:
        best_cfg = config_leaderboard.head(1).iloc[0].to_dict()
        best_cfg_path.write_text(
            json.dumps(best_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            "Best aggregated config: "
            f"{best_cfg.get('config_id')} (score_config={best_cfg.get('score_config'):.4f})"
        )
        print(f"Best aggregated config saved to: {best_cfg_path}")
    else:
        print("No successful configs to aggregate.")

    print(f"Leaderboard saved to: {leaderboard_path}")
    print(f"Config leaderboard saved to: {config_leaderboard_path}")


if __name__ == "__main__":
    main()
