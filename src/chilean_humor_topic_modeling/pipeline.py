from __future__ import annotations

import json
import logging
import platform
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import TopicModelingConfig
from .deterministic import set_global_seed
from .embeddings import load_or_compute_jina_embeddings
from .modeling import build_topic_model
from .preprocessing import load_clean_documents
from .visualization import save_plotly_figure

logger = logging.getLogger(__name__)


def _serialize_for_table(value: object) -> object:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _safe_package_version(package_name: str) -> str | None:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return None


def run_topic_modeling_pipeline(
    config: TopicModelingConfig,
    output_dir: Path,
    save_model: bool = False,
) -> dict[str, Any]:
    """Run complete BERTopic analysis and persist tabular + visual outputs."""
    logger.info("Setting global seed to %d", config.random_seed)
    set_global_seed(config.random_seed)

    logger.info("Loading and cleaning documents from dataset...")
    documents, decades, cleaned_rows, data_stats = load_clean_documents(config)
    if not documents:
        raise RuntimeError("No documents available after preprocessing.")
    logger.info(
        "Loaded %d documents for modeling. Data stats: %s",
        len(documents),
        data_stats,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    precomputed_embeddings = None
    embeddings_cache_path: Path | None = None
    embeddings_loaded_from_cache: bool | None = None
    if config.use_jina_embeddings:
        logger.info("Jina embeddings enabled. Preparing precomputed embeddings...")
        precomputed_embeddings, embeddings_cache_path, embeddings_loaded_from_cache = (
            load_or_compute_jina_embeddings(
                documents=documents,
                config=config,
                output_dir=output_dir,
            )
        )
        logger.info(
            "Embeddings ready: cache=%s loaded_from_cache=%s",
            embeddings_cache_path,
            embeddings_loaded_from_cache,
        )

    topic_model = (
        build_topic_model(config, embedding_model=None)
        if config.use_jina_embeddings
        else build_topic_model(config)
    )
    logger.info("Fitting BERTopic model...")
    topics, probabilities = topic_model.fit_transform(
        documents,
        embeddings=precomputed_embeddings,
    )
    initial_topics = list(topics)
    logger.info("BERTopic fit_transform completed.")

    if config.reduce_outliers:
        logger.info(
            "Running outlier reassignment with strategy=%s threshold=%s",
            config.reduce_outliers_strategy,
            config.reduce_outliers_threshold,
        )
        strategy = config.reduce_outliers_strategy
        if strategy == "probabilities" and probabilities is None:
            warnings.append(
                "Outlier reassignment requested with 'probabilities' but probabilities "
                "are unavailable; falling back to 'distributions'."
            )
            strategy = "distributions"

        try:
            reduce_kwargs: dict[str, Any] = {
                "strategy": strategy,
                "threshold": config.reduce_outliers_threshold,
            }
            if strategy == "probabilities" and probabilities is not None:
                reduce_kwargs["probabilities"] = probabilities
            if strategy == "embeddings" and precomputed_embeddings is not None:
                reduce_kwargs["embeddings"] = precomputed_embeddings

            topics = topic_model.reduce_outliers(
                documents,
                topics,
                **reduce_kwargs,
            )
            topic_model.update_topics(
                documents,
                topics=topics,
                vectorizer_model=topic_model.vectorizer_model,
                ctfidf_model=topic_model.ctfidf_model,
            )
        except Exception as exc:
            warnings.append(f"Skipping outlier reassignment due to error: {exc}")
            logger.warning("Skipping outlier reassignment due to error: %s", exc)

    topic_info = topic_model.get_topic_info()
    topic_name_map = {
        int(row["Topic"]): str(row["Name"])
        for _, row in topic_info.iterrows()
        if pd.notna(row["Topic"]) and pd.notna(row["Name"])
    }
    topic_info_path = tables_dir / "topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False)
    logger.info("Saved topic info table to %s", topic_info_path)

    topics_over_time = topic_model.topics_over_time(
        docs=documents,
        topics=topics,
        timestamps=decades,
        global_tuning=config.global_tuning,
        evolution_tuning=config.evolution_tuning,
    )
    topics_over_time_path = tables_dir / "topics_over_time.csv"
    topics_over_time.to_csv(topics_over_time_path, index=False)
    logger.info("Saved topics-over-time table to %s", topics_over_time_path)

    max_probabilities: list[float | None] = [None] * len(documents)
    probability_columns_path: Path | None = None
    segment_topic_probs_long_path: Path | None = None
    segment_topic_probs_dense_path: Path | None = None
    probability_matrix_np: np.ndarray | None = None
    if probabilities is not None:
        try:
            probability_matrix_np = np.asarray(probabilities)
            if probability_matrix_np.ndim == 1:
                probability_matrix_np = probability_matrix_np.reshape(-1, 1)
            if probability_matrix_np.shape[0] == len(documents):
                max_probabilities = probability_matrix_np.max(axis=1).astype(float).tolist()
            else:
                warnings.append(
                    "Probability matrix row count does not match documents; "
                    "max_topic_probability values may be incomplete."
                )

            non_outlier_topic_ids = sorted(
                topic
                for topic in pd.to_numeric(topic_info["Topic"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
                if topic != -1
            )
            if len(non_outlier_topic_ids) == probability_matrix_np.shape[1]:
                inferred_topic_ids: list[int | None] = non_outlier_topic_ids
                inference_method = "sorted_non_outlier_topic_ids"
            else:
                inferred_topic_ids = [None] * probability_matrix_np.shape[1]
                inference_method = "unresolved"
                warnings.append(
                    "Could not resolve probability matrix columns to BERTopic topic IDs; "
                    "saved matrix-index mapping only."
                )

            probability_columns = pd.DataFrame(
                {
                    "topic_id_matrix": np.arange(probability_matrix_np.shape[1], dtype=int),
                    "topic_id_assumed": inferred_topic_ids,
                    "inference_method": inference_method,
                }
            )
            probability_columns_path = tables_dir / "topic_probability_columns.csv"
            probability_columns.to_csv(probability_columns_path, index=False)

            prob_threshold = float(getattr(config, "probability_export_threshold", 0.01))
            records: list[dict[str, object]] = []
            for segment_idx, row in enumerate(probability_matrix_np):
                topic_indices = np.where(row >= prob_threshold)[0]
                for topic_idx in topic_indices:
                    records.append(
                        {
                            "segment_idx": int(segment_idx),
                            "topic_id_matrix": int(topic_idx),
                            "topic_id_assumed": (
                                int(inferred_topic_ids[topic_idx])
                                if inferred_topic_ids[topic_idx] is not None
                                else None
                            ),
                            "p_topic": float(row[topic_idx]),
                        }
                    )

            segment_topic_probs_long_path = tables_dir / "segment_topic_probs_long.csv"
            pd.DataFrame.from_records(records).to_csv(
                segment_topic_probs_long_path,
                index=False,
                columns=[
                    "segment_idx",
                    "topic_id_matrix",
                    "topic_id_assumed",
                    "p_topic",
                ],
            )

            segment_topic_probs_dense_path = tables_dir / "segment_topic_probs_dense.npz"
            np.savez_compressed(
                segment_topic_probs_dense_path,
                probs=probability_matrix_np,
            )
        except Exception as exc:
            warnings.append(
                f"Could not export topic probability artifacts; continuing without them: {exc}"
            )
            logger.warning("Could not export probability artifacts: %s", exc)

    segments_topics_records: list[dict[str, object]] = []
    for row, initial_topic, final_topic, max_probability in zip(
        cleaned_rows,
        initial_topics,
        topics,
        max_probabilities,
        strict=False,
    ):
        record = {key: _serialize_for_table(value) for key, value in row.items()}
        record["topic_initial"] = int(initial_topic)
        record["topic_final"] = int(final_topic)
        record["topic_name_initial"] = topic_name_map.get(int(initial_topic))
        record["topic_name_final"] = topic_name_map.get(int(final_topic))
        record["is_outlier_initial"] = int(initial_topic) == -1
        record["is_outlier_final"] = int(final_topic) == -1
        record["max_topic_probability"] = max_probability
        segments_topics_records.append(record)

    segments_topics_df = pd.DataFrame(segments_topics_records)
    segments_topics_path = tables_dir / "segments_topics.csv"
    segments_topics_df.to_csv(segments_topics_path, index=False)
    logger.info("Saved segment-topic table to %s", segments_topics_path)

    topic_size_distribution_path: Path | None = None
    coverage_by_year_path: Path | None = None
    coverage_by_show_path: Path | None = None
    show_year_topic_distribution_path: Path | None = None
    diagnostics = {
        "n_docs": int(len(segments_topics_df)),
        "outlier_rate_initial": float((np.asarray(initial_topics) == -1).mean()),
        "outlier_rate_final": float((segments_topics_df["topic_final"] == -1).mean()),
    }
    try:
        size_df = (
            segments_topics_df[segments_topics_df["topic_final"] != -1]
            .groupby("topic_final")
            .size()
            .rename("n")
            .reset_index()
            .sort_values("n", ascending=False)
        )
        topic_size_distribution_path = tables_dir / "topic_size_distribution.csv"
        size_df.to_csv(topic_size_distribution_path, index=False)
        diagnostics["n_topics_final"] = int(size_df.shape[0])
        diagnostics["median_topic_size"] = float(size_df["n"].median()) if len(size_df) else 0.0
        diagnostics["p90_topic_size"] = float(size_df["n"].quantile(0.9)) if len(size_df) else 0.0
    except Exception as exc:
        warnings.append(
            f"Could not compute topic size diagnostics; continuing without them: {exc}"
        )
        logger.warning("Could not compute topic size diagnostics: %s", exc)
    diagnostics.setdefault("n_topics_final", 0)
    diagnostics.setdefault("median_topic_size", 0.0)
    diagnostics.setdefault("p90_topic_size", 0.0)

    if "year" in segments_topics_df.columns:
        try:
            coverage_by_year = (
                segments_topics_df.groupby("year")
                .size()
                .rename("n_segments")
                .reset_index()
                .sort_values("year")
            )
            coverage_by_year_path = tables_dir / "coverage_by_year.csv"
            coverage_by_year.to_csv(coverage_by_year_path, index=False)
        except Exception as exc:
            warnings.append(f"Could not export coverage_by_year table: {exc}")
            logger.warning("Could not export coverage_by_year table: %s", exc)
    else:
        warnings.append("Skipping coverage_by_year export because 'year' column is missing.")

    if "show" in segments_topics_df.columns:
        try:
            coverage_by_show = (
                segments_topics_df.groupby("show")
                .size()
                .rename("n_segments")
                .reset_index()
                .sort_values("n_segments", ascending=False)
            )
            coverage_by_show_path = tables_dir / "coverage_by_show.csv"
            coverage_by_show.to_csv(coverage_by_show_path, index=False)
        except Exception as exc:
            warnings.append(f"Could not export coverage_by_show table: {exc}")
            logger.warning("Could not export coverage_by_show table: %s", exc)
    else:
        warnings.append("Skipping coverage_by_show export because 'show' column is missing.")

    if {"show", "year"}.issubset(segments_topics_df.columns):
        try:
            show_year_topic = segments_topics_df[segments_topics_df["topic_final"] != -1].copy()
            if "max_topic_probability" in show_year_topic.columns:
                show_year_topic["topic_weight"] = (
                    pd.to_numeric(show_year_topic["max_topic_probability"], errors="coerce")
                    .fillna(1.0)
                    .clip(lower=0.0, upper=1.0)
                )
            else:
                show_year_topic["topic_weight"] = 1.0

            show_year_topic_distribution = (
                show_year_topic.groupby(["show", "year", "topic_final"], as_index=False)[
                    "topic_weight"
                ]
                .sum()
                .rename(columns={"topic_final": "topic_id", "topic_weight": "topic_mass"})
            )
            totals = (
                show_year_topic_distribution.groupby(["show", "year"], as_index=False)[
                    "topic_mass"
                ]
                .sum()
                .rename(columns={"topic_mass": "total_mass"})
            )
            show_year_topic_distribution = show_year_topic_distribution.merge(
                totals, on=["show", "year"], how="left"
            )
            show_year_topic_distribution["p_topic_show_year"] = (
                show_year_topic_distribution["topic_mass"]
                / show_year_topic_distribution["total_mass"]
            )
            show_year_topic_distribution_path = (
                tables_dir / "show_year_topic_distribution.csv"
            )
            show_year_topic_distribution.to_csv(
                show_year_topic_distribution_path, index=False
            )
        except Exception as exc:
            warnings.append(
                f"Could not export show_year_topic_distribution table: {exc}"
            )
            logger.warning(
                "Could not export show_year_topic_distribution table: %s", exc
            )
    else:
        warnings.append(
            "Skipping show_year_topic_distribution export because show/year columns are missing."
        )

    segment_embeddings_path: Path | None = None
    segment_embeddings_index_path: Path | None = None
    show_year_embedding_centroids_path: Path | None = None
    show_year_embedding_centroids_csv_path: Path | None = None
    segment_embeddings: np.ndarray | None = None
    if precomputed_embeddings is not None:
        segment_embeddings = np.asarray(precomputed_embeddings)
    else:
        try:
            segment_embeddings = np.asarray(
                topic_model._extract_embeddings(  # noqa: SLF001
                    documents,
                    method="document",
                    verbose=False,
                )
            )
        except Exception as exc:
            warnings.append(
                f"Could not extract segment embeddings from BERTopic model: {exc}"
            )
            logger.warning("Could not extract segment embeddings: %s", exc)

    if segment_embeddings is not None:
        try:
            if segment_embeddings.shape[0] != len(documents):
                warnings.append(
                    "Skipping segment embedding artifacts because number of embeddings "
                    "does not match number of documents."
                )
            else:
                segment_embeddings_path = tables_dir / "segment_embeddings.npz"
                np.savez_compressed(segment_embeddings_path, embeddings=segment_embeddings)

                segment_embeddings_index = pd.DataFrame(
                    {
                        "segment_idx": np.arange(len(documents), dtype=int),
                        "show": [row.get("show") for row in cleaned_rows],
                        "year": [row.get("year") for row in cleaned_rows],
                    }
                )
                segment_embeddings_index_path = (
                    tables_dir / "segment_embeddings_index.csv"
                )
                segment_embeddings_index.to_csv(
                    segment_embeddings_index_path, index=False
                )

                if {"show", "year"}.issubset(segment_embeddings_index.columns):
                    centroid_frame = pd.DataFrame(segment_embeddings)
                    centroid_frame.insert(0, "year", segment_embeddings_index["year"].values)
                    centroid_frame.insert(0, "show", segment_embeddings_index["show"].values)
                    centroids = (
                        centroid_frame.groupby(["show", "year"], as_index=False)
                        .mean(numeric_only=True)
                    )
                    show_year_embedding_centroids_path = (
                        tables_dir / "show_year_embedding_centroids.parquet"
                    )
                    try:
                        centroids.to_parquet(
                            show_year_embedding_centroids_path, index=False
                        )
                    except Exception as exc:
                        warnings.append(
                            "Could not write show-year centroids as parquet; wrote CSV "
                            f"instead: {exc}"
                        )
                        show_year_embedding_centroids_path = None
                        show_year_embedding_centroids_csv_path = (
                            tables_dir / "show_year_embedding_centroids.csv"
                        )
                        centroids.to_csv(
                            show_year_embedding_centroids_csv_path, index=False
                        )
        except Exception as exc:
            warnings.append(
                f"Could not export segment embedding artifacts; continuing without them: {exc}"
            )
            logger.warning("Could not export segment embedding artifacts: %s", exc)

    hierarchical_topics_path: Path | None = None
    hierarchy_html_path: Path | None = None
    hierarchy_png_path: Path | None = None
    try:
        hierarchical_topics = topic_model.hierarchical_topics(documents)
        hierarchical_topics_path = tables_dir / "hierarchical_topics.csv"
        hierarchical_topics.to_csv(hierarchical_topics_path, index=False)
        hierarchy_figure = topic_model.visualize_hierarchy(
            hierarchical_topics=hierarchical_topics
        )
        hierarchy_html_path, hierarchy_png_path = save_plotly_figure(
            hierarchy_figure,
            figures_dir,
            "topic_hierarchy",
        )
        logger.info("Saved hierarchical topics artifacts.")
    except Exception as exc:
        warnings.append(f"Skipping hierarchical topics artifacts due to error: {exc}")
        logger.warning("Skipping hierarchical topics artifacts due to error: %s", exc)

    topics_html_path: Path | None = None
    topics_png_path: Path | None = None
    non_outlier_topics = topic_info[topic_info["Topic"] != -1]
    if len(non_outlier_topics) >= 2:
        try:
            topics_figure = topic_model.visualize_topics()
            topics_html_path, topics_png_path = save_plotly_figure(
                topics_figure, figures_dir, "topics"
            )
        except Exception as exc:
            warnings.append(f"Skipping topic-map figure due to error: {exc}")
            logger.warning("Skipping topic-map figure due to error: %s", exc)
    else:
        warnings.append(
            "Skipping visualize_topics because fewer than two non-outlier topics were found."
        )

    over_time_html_path: Path | None = None
    over_time_png_path: Path | None = None
    try:
        over_time_figure = topic_model.visualize_topics_over_time(
            topics_over_time, top_n_topics=config.top_n_topics
        )
        over_time_html_path, over_time_png_path = save_plotly_figure(
            over_time_figure, figures_dir, "topics_over_time_top_n"
        )
    except Exception as exc:
        warnings.append(f"Skipping topics-over-time figure due to error: {exc}")
        logger.warning("Skipping topics-over-time figure due to error: %s", exc)

    selected_html_path: Path | None = None
    selected_png_path: Path | None = None
    if config.selected_topics:
        available_topics = set(topic_info["Topic"].tolist())
        filtered_selected_topics = [
            topic_id
            for topic_id in config.selected_topics
            if topic_id in available_topics and topic_id != -1
        ]
        if filtered_selected_topics:
            try:
                selected_topics_figure = topic_model.visualize_topics_over_time(
                    topics_over_time,
                    topics=filtered_selected_topics,
                )
                selected_html_path, selected_png_path = save_plotly_figure(
                    selected_topics_figure,
                    figures_dir,
                    "topics_over_time_selected",
                )
            except Exception as exc:
                warnings.append(f"Skipping selected-topics figure due to error: {exc}")
                logger.warning("Skipping selected-topics figure due to error: %s", exc)
        else:
            warnings.append(
                "Skipping selected-topics figure because none of the requested topic IDs exist."
            )

    model_dir: Path | None = None
    if save_model:
        model_dir = output_dir / "model"
        topic_model.save(str(model_dir), serialization="safetensors", save_ctfidf=True)
        logger.info("Saved BERTopic model to %s", model_dir)

    package_versions = {
        "bertopic": _safe_package_version("bertopic"),
        "umap_learn": _safe_package_version("umap-learn"),
        "hdbscan": _safe_package_version("hdbscan"),
        "scikit_learn": _safe_package_version("scikit-learn"),
        "pandas": _safe_package_version("pandas"),
        "numpy": _safe_package_version("numpy"),
    }
    run_id = output_dir.name if output_dir.parent.name == "runs" else None
    run_report = {
        "run_id": run_id,
        "seed": int(config.random_seed),
        "config": config.to_dict(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "package_versions": package_versions,
        },
        "params": {
            "embedding_model_name": (
                config.jina_model_name if config.use_jina_embeddings else config.embedding_model
            ),
            "use_jina_embeddings": bool(config.use_jina_embeddings),
            "umap": {
                "n_neighbors": int(config.umap_n_neighbors),
                "n_components": int(config.umap_n_components),
                "min_dist": float(config.umap_min_dist),
                "metric": str(config.umap_metric),
                "random_state": int(config.random_seed),
            },
            "hdbscan": {
                "min_cluster_size": int(config.hdbscan_min_cluster_size),
                "min_samples": (
                    int(config.hdbscan_min_samples)
                    if config.hdbscan_min_samples is not None
                    else None
                ),
                "metric": str(config.hdbscan_metric),
                "cluster_selection_method": str(config.hdbscan_cluster_selection_method),
            },
            "vectorizer": {
                "ngram_range": [int(config.ngram_range_min), int(config.ngram_range_max)],
                "min_df": int(config.min_df),
                "max_df": None,
                "max_features": None,
                "token_pattern": config.token_pattern,
            },
            "probability_export_threshold": float(
                getattr(config, "probability_export_threshold", 0.01)
            ),
        },
        "data_stats": data_stats,
        "num_topics_found": int(topic_info.shape[0]),
        "num_documents_modeled": len(documents),
        "diagnostics": diagnostics,
        "warnings": warnings,
        "artifacts": {
            "topic_info_csv": str(topic_info_path.resolve()),
            "topics_over_time_csv": str(topics_over_time_path.resolve()),
            "segments_topics_csv": str(segments_topics_path.resolve()),
            "topic_probability_columns_csv": (
                str(probability_columns_path.resolve())
                if probability_columns_path
                else None
            ),
            "segment_topic_probs_long_csv": (
                str(segment_topic_probs_long_path.resolve())
                if segment_topic_probs_long_path
                else None
            ),
            "segment_topic_probs_dense_npz": (
                str(segment_topic_probs_dense_path.resolve())
                if segment_topic_probs_dense_path
                else None
            ),
            "topic_size_distribution_csv": (
                str(topic_size_distribution_path.resolve())
                if topic_size_distribution_path
                else None
            ),
            "coverage_by_year_csv": (
                str(coverage_by_year_path.resolve()) if coverage_by_year_path else None
            ),
            "coverage_by_show_csv": (
                str(coverage_by_show_path.resolve()) if coverage_by_show_path else None
            ),
            "show_year_topic_distribution_csv": (
                str(show_year_topic_distribution_path.resolve())
                if show_year_topic_distribution_path
                else None
            ),
            "segment_embeddings_npz": (
                str(segment_embeddings_path.resolve()) if segment_embeddings_path else None
            ),
            "segment_embeddings_index_csv": (
                str(segment_embeddings_index_path.resolve())
                if segment_embeddings_index_path
                else None
            ),
            "show_year_embedding_centroids_parquet": (
                str(show_year_embedding_centroids_path.resolve())
                if show_year_embedding_centroids_path
                else None
            ),
            "show_year_embedding_centroids_csv": (
                str(show_year_embedding_centroids_csv_path.resolve())
                if show_year_embedding_centroids_csv_path
                else None
            ),
            "hierarchical_topics_csv": str(hierarchical_topics_path.resolve())
            if hierarchical_topics_path
            else None,
            "topics_html": str(topics_html_path.resolve()) if topics_html_path else None,
            "topics_png": str(topics_png_path.resolve()) if topics_png_path else None,
            "topic_hierarchy_html": str(hierarchy_html_path.resolve())
            if hierarchy_html_path
            else None,
            "topic_hierarchy_png": str(hierarchy_png_path.resolve())
            if hierarchy_png_path
            else None,
            "topics_over_time_html": str(over_time_html_path.resolve())
            if over_time_html_path
            else None,
            "topics_over_time_png": str(over_time_png_path.resolve())
            if over_time_png_path
            else None,
            "selected_topics_html": str(selected_html_path.resolve())
            if selected_html_path
            else None,
            "selected_topics_png": str(selected_png_path.resolve())
            if selected_png_path
            else None,
            "jina_embeddings_cache": str(embeddings_cache_path.resolve())
            if embeddings_cache_path
            else None,
            "jina_embeddings_loaded_from_cache": embeddings_loaded_from_cache,
            "model_dir": str(model_dir.resolve()) if model_dir else None,
        },
    }

    report_path = output_dir / "run_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(run_report, file, ensure_ascii=False, indent=2)
    logger.info("Saved run report to %s", report_path)

    result = {
        "topic_model": topic_model,
        "topics": topics,
        "probabilities": probabilities,
        "topics_over_time": topics_over_time,
        "topic_info": topic_info,
        "run_report_path": report_path,
    }
    return result
