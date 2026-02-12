from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import TopicModelingConfig
from .deterministic import set_global_seed
from .embeddings import load_or_compute_jina_embeddings
from .modeling import build_topic_model
from .preprocessing import load_clean_documents
from .visualization import save_plotly_figure

logger = logging.getLogger(__name__)


def run_topic_modeling_pipeline(
    config: TopicModelingConfig,
    output_dir: Path,
    save_model: bool = False,
) -> dict[str, Any]:
    """Run complete BERTopic analysis and persist tabular + visual outputs."""
    logger.info("Setting global seed to %d", config.random_seed)
    set_global_seed(config.random_seed)

    logger.info("Loading and cleaning documents from dataset...")
    documents, decades, data_stats = load_clean_documents(config)
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

    run_report = {
        "config": config.to_dict(),
        "data_stats": data_stats,
        "num_topics_found": int(topic_info.shape[0]),
        "num_documents_modeled": len(documents),
        "warnings": warnings,
        "artifacts": {
            "topic_info_csv": str(topic_info_path.resolve()),
            "topics_over_time_csv": str(topics_over_time_path.resolve()),
            "topics_html": str(topics_html_path.resolve()) if topics_html_path else None,
            "topics_png": str(topics_png_path.resolve()) if topics_png_path else None,
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
