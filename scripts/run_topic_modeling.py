#!/usr/bin/env python3
"""Run deterministic BERTopic analysis for Chilean humor transcripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chilean_humor_topic_modeling.config import TopicModelingConfig


def parse_selected_topics(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    return tuple(int(token.strip()) for token in raw.split(",") if token.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a BERTopic pipeline with fixed preprocessing and deterministic settings."
        )
    )
    parser.add_argument(
        "--repo-id",
        default="astroza/chilean-humor-raw-transcripts",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--config-name",
        default="segments",
        help="Dataset config name inside the HF repo.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing segment text.",
    )
    parser.add_argument(
        "--date-column",
        default="date",
        help="Column containing date values.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional cap on number of cleaned documents used for fitting.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=25,
        help="Drop segments shorter than this number of characters.",
    )
    parser.add_argument(
        "--min-text-tokens",
        type=int,
        default=5,
        help="Drop segments shorter than this number of whitespace tokens.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/topic_modeling"),
        help="Directory where charts, tables and run report will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic behavior.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model used by BERTopic.",
    )
    parser.add_argument(
        "--use-jina-embeddings",
        action="store_true",
        help=(
            "Precompute Jina embeddings and pass them to BERTopic.fit_transform. "
            "When enabled, BERTopic is initialized without an internal embedding model."
        ),
    )
    parser.add_argument(
        "--jina-model-name",
        default="jinaai/jina-embeddings-v4",
        help="Jina model id loaded via transformers.AutoModel.from_pretrained.",
    )
    parser.add_argument(
        "--jina-task",
        default="text-matching",
        help="Task argument passed to model.encode_text(...).",
    )
    parser.add_argument(
        "--jina-truncate-dim",
        type=int,
        default=128,
        help="Embedding dimension truncation for Jina; set <= 0 to disable.",
    )
    parser.add_argument(
        "--jina-batch-size",
        type=int,
        default=64,
        help="Batch size used while computing Jina embeddings.",
    )
    parser.add_argument(
        "--jina-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for Jina embeddings computation.",
    )
    parser.add_argument(
        "--jina-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for embedding cache files. "
            "If omitted, uses <output-dir>/embeddings_cache."
        ),
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency in CountVectorizer.",
    )
    parser.add_argument(
        "--token-pattern",
        default=None,
        help=(
            "Optional custom regex for CountVectorizer tokenization. "
            "If omitted, scikit-learn default tokenization is used."
        ),
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=8,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--umap-n-components",
        type=int,
        default=10,
        help="UMAP n_components parameter.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.0,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=8,
        help="HDBSCAN min_cluster_size parameter.",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=3,
        help="Optional HDBSCAN min_samples parameter.",
    )
    parser.add_argument(
        "--hdbscan-cluster-selection-method",
        choices=["eom", "leaf"],
        default="leaf",
        help="HDBSCAN cluster_selection_method.",
    )
    parser.add_argument(
        "--top-n-topics",
        type=int,
        default=20,
        help="Number of topics to show in topics-over-time chart.",
    )
    parser.add_argument(
        "--selected-topics",
        default="11,12,13,18",
        help="Comma-separated topic IDs for selected-topics chart.",
    )
    parser.add_argument(
        "--disable-outlier-reassignment",
        action="store_true",
        help="Disable BERTopic outlier reassignment after fit_transform.",
    )
    parser.add_argument(
        "--outlier-strategy",
        choices=["probabilities", "distributions", "c-tf-idf", "embeddings"],
        default="distributions",
        help="Strategy used by BERTopic.reduce_outliers.",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=0.0,
        help="Threshold used by BERTopic.reduce_outliers.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Persist trained BERTopic model under output_dir/model.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose BERTopic logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from chilean_humor_topic_modeling.pipeline import run_topic_modeling_pipeline

    config = TopicModelingConfig(
        repo_id=args.repo_id,
        config_name=args.config_name,
        split=args.split,
        text_column=args.text_column,
        date_column=args.date_column,
        sample_size=args.sample_size,
        min_text_chars=args.min_text_chars,
        min_text_tokens=args.min_text_tokens,
        embedding_model=args.embedding_model,
        use_jina_embeddings=args.use_jina_embeddings,
        jina_model_name=args.jina_model_name,
        jina_task=args.jina_task,
        jina_truncate_dim=(
            args.jina_truncate_dim if args.jina_truncate_dim > 0 else None
        ),
        jina_batch_size=args.jina_batch_size,
        jina_device=args.jina_device,
        jina_cache_dir=str(args.jina_cache_dir) if args.jina_cache_dir else None,
        verbose=not args.quiet,
        min_df=args.min_df,
        token_pattern=args.token_pattern,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_n_components=args.umap_n_components,
        umap_min_dist=args.umap_min_dist,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_cluster_selection_method=args.hdbscan_cluster_selection_method,
        top_n_topics=args.top_n_topics,
        selected_topics=parse_selected_topics(args.selected_topics),
        reduce_outliers=not args.disable_outlier_reassignment,
        reduce_outliers_strategy=args.outlier_strategy,
        reduce_outliers_threshold=args.outlier_threshold,
        random_seed=args.seed,
    )

    result = run_topic_modeling_pipeline(
        config=config,
        output_dir=args.output_dir,
        save_model=args.save_model,
    )

    print("Pipeline finished successfully.")
    print(f"Run report: {result['run_report_path']}")
    print(f"Output dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
