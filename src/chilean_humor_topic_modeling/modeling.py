from __future__ import annotations

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import nltk

from .config import TopicModelingConfig

_DEFAULT_EMBEDDING_MODEL = object()


def ensure_stopwords_resource() -> None:
    """Download NLTK stopwords if not already available."""
    nltk.download("stopwords", quiet=True)


def build_vectorizer(config: TopicModelingConfig) -> CountVectorizer:
    """Build a Spanish-oriented vectorizer for BERTopic c-TF-IDF."""
    ensure_stopwords_resource()
    spanish_stopwords = stopwords.words("spanish")
    all_stopwords = sorted({*spanish_stopwords, *config.extra_stopwords})

    vectorizer_kwargs: dict[str, object] = {
        "stop_words": all_stopwords,
        "min_df": config.min_df,
        "ngram_range": (config.ngram_range_min, config.ngram_range_max),
    }
    if config.token_pattern is not None:
        vectorizer_kwargs["token_pattern"] = config.token_pattern

    return CountVectorizer(
        **vectorizer_kwargs,
    )


def build_umap_model(config: TopicModelingConfig) -> UMAP:
    """Build deterministic UMAP dimensionality reduction model."""
    return UMAP(
        n_neighbors=config.umap_n_neighbors,
        n_components=config.umap_n_components,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.random_seed,
    )


def build_hdbscan_model(config: TopicModelingConfig) -> HDBSCAN:
    """Build HDBSCAN clustering model used by BERTopic."""
    return HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric=config.hdbscan_metric,
        cluster_selection_method=config.hdbscan_cluster_selection_method,
        prediction_data=True,
    )


def build_topic_model(
    config: TopicModelingConfig,
    embedding_model: object = _DEFAULT_EMBEDDING_MODEL,
) -> BERTopic:
    """Create BERTopic model with deterministic sub-components."""
    effective_embedding_model = (
        config.embedding_model
        if embedding_model is _DEFAULT_EMBEDDING_MODEL
        else embedding_model
    )
    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=config.ctfidf_reduce_frequent_words,
        bm25_weighting=config.ctfidf_bm25_weighting,
    )

    return BERTopic(
        language=config.language,
        embedding_model=effective_embedding_model,
        vectorizer_model=build_vectorizer(config),
        ctfidf_model=ctfidf_model,
        umap_model=build_umap_model(config),
        hdbscan_model=build_hdbscan_model(config),
        calculate_probabilities=config.calculate_probabilities,
        verbose=config.verbose,
    )
