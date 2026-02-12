from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TopicModelingConfig:
    """Configuration for the BERTopic pipeline."""

    repo_id: str = "astroza/chilean-humor-raw-transcripts"
    config_name: str = "segments"
    split: str = "train"
    text_column: str = "text"
    date_column: str = "date"
    sample_size: int | None = None
    min_text_chars: int = 25
    min_text_tokens: int = 5

    language: str = "multilingual"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    use_jina_embeddings: bool = False
    jina_provider: str = "local"
    jina_model_name: str = "jinaai/jina-embeddings-v4"
    jina_task: str = "text-matching"
    jina_truncate_dim: int | None = 128
    jina_batch_size: int = 16
    jina_device: str = "auto"
    jina_api_url: str = "https://api.jina.ai/v1/embeddings"
    jina_api_token_env: str = "JINA_API_TOKEN"
    jina_api_timeout_seconds: float = 60.0
    jina_cache_dir: str | None = None
    calculate_probabilities: bool = True
    verbose: bool = True

    min_df: int = 2
    ngram_range_min: int = 1
    ngram_range_max: int = 2
    token_pattern: str | None = None
    extra_stopwords: tuple[str, ...] = (
        "ah",
        "eh",
        "oye",
        "po",
        "gracias",
        "huevon",
        "weon",
        "cachai",
        "nomas",
    )
    ctfidf_reduce_frequent_words: bool = True
    ctfidf_bm25_weighting: bool = True

    umap_n_neighbors: int = 15
    umap_n_components: int = 10
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"

    hdbscan_min_cluster_size: int = 22
    hdbscan_min_samples: int | None = 6
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_method: str = "eom"

    global_tuning: bool = True
    evolution_tuning: bool = True
    top_n_topics: int = 20
    selected_topics: tuple[int, ...] = (11, 12, 13, 18)
    reduce_outliers: bool = True
    reduce_outliers_strategy: str = "distributions"
    reduce_outliers_threshold: float = 0.0

    random_seed: int = 42

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["selected_topics"] = list(self.selected_topics)
        data["extra_stopwords"] = list(self.extra_stopwords)
        return data
