from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import TopicModelingConfig


def _compute_cache_key(documents: list[str], config: TopicModelingConfig) -> str:
    hasher = hashlib.sha256()
    fingerprint = {
        "model": config.jina_model_name,
        "task": config.jina_task,
        "truncate_dim": config.jina_truncate_dim,
        "num_documents": len(documents),
    }
    hasher.update(json.dumps(fingerprint, sort_keys=True).encode("utf-8"))
    for document in documents:
        hasher.update(b"\0")
        hasher.update(document.encode("utf-8"))
    return hasher.hexdigest()


def _to_numpy(embeddings: object) -> np.ndarray:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(embeddings, torch.Tensor):
        array = embeddings.detach().cpu().numpy()
    elif isinstance(embeddings, np.ndarray):
        array = embeddings
    else:
        vectors: list[np.ndarray] = []
        for embedding in embeddings:  # type: ignore[assignment]
            vector = embedding
            if hasattr(vector, "detach"):
                vector = vector.detach()
            if hasattr(vector, "cpu"):
                vector = vector.cpu()
            vectors.append(np.asarray(vector, dtype=np.float32))
        array = np.vstack(vectors)

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _resolve_device(torch_module: object, configured_device: str) -> str:
    if configured_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "jina_device must be one of: 'auto', 'cpu', 'cuda'. "
            f"Received: {configured_device!r}"
        )

    if configured_device == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"

    if configured_device == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(
            "jina_device='cuda' was requested but CUDA is not available."
        )
    return configured_device


def _compute_jina_embeddings(
    documents: list[str],
    config: TopicModelingConfig,
) -> np.ndarray:
    import torch
    import transformers
    from transformers import AutoModel

    if not hasattr(transformers.cache_utils, "SlidingWindowCache"):
        raise RuntimeError(
            "Incompatible transformers version detected for jina-embeddings-v4. "
            "Please install: transformers>=4.50,<5"
        )

    device = _resolve_device(torch, config.jina_device)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        model = AutoModel.from_pretrained(
            config.jina_model_name,
            trust_remote_code=True,
            dtype=torch_dtype,
        )
    except TypeError:
        # Backward-compatible fallback for older transformers releases.
        model = AutoModel.from_pretrained(
            config.jina_model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    model.to(device)

    if not hasattr(model, "encode_text"):
        raise RuntimeError(
            f"Model {config.jina_model_name!r} does not expose encode_text(...)."
        )

    batch_size = max(1, config.jina_batch_size)
    batches: list[np.ndarray] = []
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        encode_kwargs: dict[str, object] = {
            "texts": batch_docs,
            "task": config.jina_task,
        }
        if config.jina_truncate_dim is not None:
            encode_kwargs["truncate_dim"] = config.jina_truncate_dim

        batch_embeddings = model.encode_text(**encode_kwargs)
        batches.append(_to_numpy(batch_embeddings))

    if not batches:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = np.vstack(batches).astype(np.float32, copy=False)
    if embeddings.shape[0] != len(documents):
        raise RuntimeError(
            "Jina embedding count does not match number of documents: "
            f"{embeddings.shape[0]} != {len(documents)}"
        )
    return embeddings


def load_or_compute_jina_embeddings(
    documents: list[str],
    config: TopicModelingConfig,
    output_dir: Path,
) -> tuple[np.ndarray, Path, bool]:
    """
    Load cached Jina embeddings when available, otherwise compute and cache them.

    Returns:
        embeddings: 2D float32 array aligned with input documents
        cache_path: location of the .npy cache file
        loaded_from_cache: whether embeddings were reused from disk
    """
    cache_root = (
        Path(config.jina_cache_dir)
        if config.jina_cache_dir is not None
        else output_dir / "embeddings_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_key = _compute_cache_key(documents, config)
    embeddings_path = cache_root / f"{cache_key}.npy"
    metadata_path = cache_root / f"{cache_key}.meta.json"

    if embeddings_path.exists():
        cached_embeddings = np.load(embeddings_path)
        if cached_embeddings.ndim == 2 and cached_embeddings.shape[0] == len(documents):
            return cached_embeddings.astype(np.float32, copy=False), embeddings_path, True

    embeddings = _compute_jina_embeddings(documents, config)
    np.save(embeddings_path, embeddings)

    metadata = {
        "cache_key": cache_key,
        "model": config.jina_model_name,
        "task": config.jina_task,
        "truncate_dim": config.jina_truncate_dim,
        "batch_size": config.jina_batch_size,
        "device": config.jina_device,
        "num_documents": len(documents),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return embeddings, embeddings_path, False
