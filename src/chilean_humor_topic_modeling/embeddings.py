from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import TopicModelingConfig

logger = logging.getLogger(__name__)


def _compute_cache_key(documents: list[str], config: TopicModelingConfig) -> str:
    hasher = hashlib.sha256()
    fingerprint = {
        "provider": config.jina_provider,
        "model": config.jina_model_name,
        "task": config.jina_task,
        "truncate_dim": config.jina_truncate_dim,
        "api_url": config.jina_api_url.rstrip("/")
        if config.jina_provider == "api"
        else None,
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
    batch_callback: Callable[[int, np.ndarray], None] | None = None,
    start_index_offset: int = 0,
) -> np.ndarray:
    provider = config.jina_provider.strip().lower()
    if provider == "local":
        return _compute_jina_embeddings_locally(
            documents,
            config,
            batch_callback=batch_callback,
            start_index_offset=start_index_offset,
        )
    if provider == "api":
        return _compute_jina_embeddings_via_api(
            documents,
            config,
            batch_callback=batch_callback,
            start_index_offset=start_index_offset,
        )
    raise ValueError(
        "jina_provider must be one of: 'local', 'api'. "
        f"Received: {config.jina_provider!r}"
    )


def _compute_jina_embeddings_locally(
    documents: list[str],
    config: TopicModelingConfig,
    batch_callback: Callable[[int, np.ndarray], None] | None = None,
    start_index_offset: int = 0,
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
    global_total_docs = start_index_offset + len(documents)
    logger.info(
        "Computing local Jina embeddings: docs=%d model=%s task=%s batch_size=%d device=%s",
        global_total_docs,
        config.jina_model_name,
        config.jina_task,
        batch_size,
        device,
    )
    batches: list[np.ndarray] = []
    total_batches = (len(documents) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(documents), batch_size), start=1):
        batch_docs = documents[start : start + batch_size]
        global_start = start_index_offset + start + 1
        global_end = start_index_offset + start + len(batch_docs)
        logger.info(
            "Local embedding batch %d/%d (docs %d-%d, size=%d)",
            batch_index,
            total_batches,
            global_start,
            global_end,
            len(batch_docs),
        )
        encode_kwargs: dict[str, object] = {
            "texts": batch_docs,
            "task": config.jina_task,
        }
        if config.jina_truncate_dim is not None:
            encode_kwargs["truncate_dim"] = config.jina_truncate_dim

        batch_embeddings = model.encode_text(**encode_kwargs)
        batch_array = _to_numpy(batch_embeddings)
        if batch_callback is not None:
            batch_callback(start_index_offset + start, batch_array)
        batches.append(batch_array)

    if not batches:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = np.vstack(batches).astype(np.float32, copy=False)
    if embeddings.shape[0] != len(documents):
        raise RuntimeError(
            "Jina embedding count does not match number of documents: "
            f"{embeddings.shape[0]} != {len(documents)}"
        )
    return embeddings


def _load_api_token(token_env: str) -> str:
    token = os.getenv(token_env)
    if token:
        return token

    try:
        from dotenv import load_dotenv

        load_dotenv()
        token = os.getenv(token_env)
    except Exception:
        token = None

    if token:
        return token
    raise RuntimeError(
        f"Missing Jina API token. Set environment variable {token_env!r} "
        "or define it in a .env file."
    )


def _extract_response_error_message(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""

    for key in ("detail", "message", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested_message = value.get("message")
            if isinstance(nested_message, str) and nested_message.strip():
                return nested_message.strip()
    return ""


def _safe_json_response(response: object) -> object:
    if not hasattr(response, "json"):
        return {}
    try:
        return response.json()  # type: ignore[attr-defined]
    except Exception:
        return {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _post_jina_embeddings_request(
    requests_module: object,
    api_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
    max_retries: int = 3,
    initial_backoff_seconds: float = 1.5,
) -> tuple[object, object]:
    if max_retries < 1:
        max_retries = 1

    retryable_statuses = {429, 500, 502, 503, 504}
    backoff_seconds = max(0.0, initial_backoff_seconds)
    timeout_exception: Exception | None = None
    network_exception: Exception | None = None
    last_response: object | None = None
    last_payload: object = {}

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.warning(
                    "Retrying Jina API request attempt %d/%d",
                    attempt,
                    max_retries,
                )
            response = requests_module.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
        except requests_module.exceptions.Timeout as exc:
            timeout_exception = exc
            if attempt < max_retries and backoff_seconds > 0:
                logger.warning(
                    "Jina API request timed out on attempt %d/%d. Backing off %.1fs.",
                    attempt,
                    max_retries,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)
                backoff_seconds *= 2
                continue
            raise
        except requests_module.exceptions.RequestException as exc:
            network_exception = exc
            if attempt < max_retries and backoff_seconds > 0:
                logger.warning(
                    "Jina API request failed on attempt %d/%d: %s. Backing off %.1fs.",
                    attempt,
                    max_retries,
                    exc,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)
                backoff_seconds *= 2
                continue
            raise

        response_payload = _safe_json_response(response)
        if (
            response.status_code in {400, 422}
            and "dimensions" in payload
            and payload.get("dimensions") is not None
        ):
            fallback_payload = dict(payload)
            fallback_payload.pop("dimensions", None)
            fallback_response = requests_module.post(
                api_url,
                headers=headers,
                json=fallback_payload,
                timeout=timeout_seconds,
            )
            fallback_response_payload = _safe_json_response(fallback_response)
            if fallback_response.status_code < 400:
                response = fallback_response
                response_payload = fallback_response_payload

        if response.status_code in retryable_statuses and attempt < max_retries:
            logger.warning(
                "Jina API returned retryable status %s on attempt %d/%d.",
                response.status_code,
                attempt,
                max_retries,
            )
            if backoff_seconds > 0:
                time.sleep(backoff_seconds)
                backoff_seconds *= 2
            last_response = response
            last_payload = response_payload
            continue

        return response, response_payload

    if timeout_exception is not None:
        raise timeout_exception
    if network_exception is not None:
        raise network_exception
    if last_response is not None:
        return last_response, last_payload
    raise RuntimeError("Failed to complete Jina API request.")


def _compute_jina_embeddings_via_api(
    documents: list[str],
    config: TopicModelingConfig,
    batch_callback: Callable[[int, np.ndarray], None] | None = None,
    start_index_offset: int = 0,
) -> np.ndarray:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "The 'requests' package is required for jina_provider='api'. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    if config.jina_api_timeout_seconds <= 0:
        raise ValueError(
            "jina_api_timeout_seconds must be greater than 0. "
            f"Received: {config.jina_api_timeout_seconds!r}"
        )

    api_url = config.jina_api_url.strip()
    if not api_url:
        raise ValueError("jina_api_url cannot be empty when jina_provider='api'.")

    token = _load_api_token(config.jina_api_token_env)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    batch_size = max(1, config.jina_batch_size)
    model_name = config.jina_model_name.strip()
    if "/" in model_name:
        model_name = model_name.rsplit("/", maxsplit=1)[-1]

    completed_batches: list[tuple[int, np.ndarray]] = []
    pending_batches: deque[tuple[int, list[str]]] = deque(
        (start, documents[start : start + batch_size])
        for start in range(0, len(documents), batch_size)
    )
    global_total_docs = start_index_offset + len(documents)
    logger.info(
        "Computing API Jina embeddings: docs=%d model=%s task=%s initial_batch_size=%d timeout=%ss endpoint=%s",
        global_total_docs,
        model_name,
        config.jina_task,
        batch_size,
        config.jina_api_timeout_seconds,
        api_url,
    )
    processed_docs = 0

    while pending_batches:
        start, batch_docs = pending_batches.popleft()
        global_start = start_index_offset + start + 1
        global_end = start_index_offset + start + len(batch_docs)
        logger.info(
            "API embedding request for docs %d-%d (size=%d, pending_batches=%d)",
            global_start,
            global_end,
            len(batch_docs),
            len(pending_batches),
        )
        payload: dict[str, Any] = {
            "model": model_name,
            "task": config.jina_task,
            "input": [{"text": text} for text in batch_docs],
        }
        if config.jina_truncate_dim is not None:
            payload["dimensions"] = config.jina_truncate_dim

        try:
            response, response_payload = _post_jina_embeddings_request(
                requests_module=requests,
                api_url=api_url,
                headers=headers,
                payload=payload,
                timeout_seconds=config.jina_api_timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            if len(batch_docs) <= 1:
                raise RuntimeError(
                    "Jina API timed out even with batch size 1. "
                    "Increase --jina-api-timeout-seconds or retry later."
                ) from exc
            split_at = len(batch_docs) // 2
            first_half = batch_docs[:split_at]
            second_half = batch_docs[split_at:]
            logger.warning(
                "Timeout for docs %d-%d (size=%d). Splitting into %d + %d docs.",
                global_start,
                global_end,
                len(batch_docs),
                len(first_half),
                len(second_half),
            )
            pending_batches.appendleft((start + split_at, second_half))
            pending_batches.appendleft((start, first_half))
            continue
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Jina API network error: {exc}. Check connectivity and retry."
            ) from exc

        if response.status_code >= 400:
            error_message = _extract_response_error_message(response_payload)
            if not error_message:
                error_message = response.text.strip()
            raise RuntimeError(
                "Jina API request failed with status "
                f"{response.status_code}: {error_message}"
            )

        if not isinstance(response_payload, dict):
            raise RuntimeError("Jina API returned a non-JSON object.")

        data = response_payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Jina API response is missing a list in 'data'.")

        parsed_batch: list[tuple[int, np.ndarray]] = []
        for fallback_index, item in enumerate(data):
            if not isinstance(item, dict):
                raise RuntimeError("Jina API response 'data' contains a non-object item.")

            index_raw = item.get("index", fallback_index)
            try:
                index = int(index_raw)
            except (TypeError, ValueError):
                index = fallback_index

            embedding = item.get("embedding")
            if embedding is None:
                raise RuntimeError("Jina API response item is missing 'embedding'.")
            parsed_batch.append((index, np.asarray(embedding, dtype=np.float32)))

        if len(parsed_batch) != len(batch_docs):
            raise RuntimeError(
                "Jina API embedding count does not match batch size: "
                f"{len(parsed_batch)} != {len(batch_docs)}"
            )

        min_index = min(index for index, _ in parsed_batch)
        normalized = [(index - min_index, embedding) for index, embedding in parsed_batch]
        normalized_indices = sorted(index for index, _ in normalized)
        if normalized_indices != list(range(len(batch_docs))):
            raise RuntimeError(
                "Jina API response indices are not aligned with the request order."
            )

        ordered_vectors = [
            embedding for _, embedding in sorted(normalized, key=lambda item: item[0])
        ]
        batch_array = np.vstack(ordered_vectors)
        if batch_callback is not None:
            batch_callback(start_index_offset + start, batch_array)
        completed_batches.append((start, batch_array))
        processed_docs += len(batch_docs)
        global_processed_docs = start_index_offset + processed_docs
        logger.info(
            "API embeddings progress: %d/%d docs processed",
            global_processed_docs,
            global_total_docs,
        )

    if not completed_batches:
        return np.empty((0, 0), dtype=np.float32)

    completed_batches.sort(key=lambda item: item[0])
    embeddings = np.vstack([vectors for _, vectors in completed_batches]).astype(
        np.float32, copy=False
    )
    if embeddings.shape[0] != len(documents):
        raise RuntimeError(
            "Jina API embedding count does not match number of documents: "
            f"{embeddings.shape[0]} != {len(documents)}"
        )
    logger.info(
        "Finished API embeddings: docs=%d dim=%d",
        global_total_docs,
        embeddings.shape[1] if embeddings.ndim == 2 else -1,
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
    partial_embeddings_path = cache_root / f"{cache_key}.partial.npy"
    partial_metadata_path = cache_root / f"{cache_key}.partial.meta.json"
    logger.info(
        "Embedding cache lookup: provider=%s cache=%s",
        config.jina_provider,
        embeddings_path,
    )

    if embeddings_path.exists():
        cached_embeddings = np.load(embeddings_path)
        if cached_embeddings.ndim == 2 and cached_embeddings.shape[0] == len(documents):
            logger.info(
                "Loaded embeddings from cache: docs=%d dim=%d",
                cached_embeddings.shape[0],
                cached_embeddings.shape[1] if cached_embeddings.ndim == 2 else -1,
            )
            return cached_embeddings.astype(np.float32, copy=False), embeddings_path, True

    def _cleanup_partial_cache() -> None:
        for path in (partial_embeddings_path, partial_metadata_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                logger.warning("Failed to remove partial cache file %s: %s", path, exc)

    num_completed = 0
    initial_num_completed = 0
    partial_embeddings: np.ndarray | None = None
    checkpoint_embeddings: np.ndarray | None = None
    embedding_dim: int | None = None

    if partial_embeddings_path.exists() and partial_metadata_path.exists():
        try:
            checkpoint_metadata = json.loads(
                partial_metadata_path.read_text(encoding="utf-8")
            )
            checkpoint_embeddings = np.load(partial_embeddings_path, mmap_mode="r+")
            checkpoint_completed = int(checkpoint_metadata.get("num_completed", 0))
            if (
                checkpoint_embeddings.ndim == 2
                and checkpoint_embeddings.shape[0] == len(documents)
            ):
                num_completed = max(0, min(len(documents), checkpoint_completed))
                initial_num_completed = num_completed
                partial_embeddings = checkpoint_embeddings
                embedding_dim = int(checkpoint_embeddings.shape[1])
                logger.info(
                    "Resuming embeddings checkpoint: %d/%d docs already cached.",
                    num_completed,
                    len(documents),
                )
            else:
                logger.warning(
                    "Ignoring partial cache due to incompatible shape: %s",
                    checkpoint_embeddings.shape,
                )
        except Exception as exc:
            logger.warning(
                "Ignoring unreadable partial cache checkpoint due to error: %s",
                exc,
            )

    def _persist_partial_batch(start_index: int, batch_embeddings: np.ndarray) -> None:
        nonlocal partial_embeddings
        nonlocal num_completed
        nonlocal embedding_dim

        if batch_embeddings.ndim != 2:
            raise RuntimeError(
                "Batch callback received non-2D embeddings with shape "
                f"{batch_embeddings.shape!r}"
            )
        if start_index < 0:
            raise RuntimeError(f"Invalid batch start index: {start_index}")

        batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
        batch_rows, batch_dim = batch_embeddings.shape
        end_index = start_index + batch_rows
        if end_index > len(documents):
            raise RuntimeError(
                "Batch callback received out-of-range slice: "
                f"{start_index}:{end_index} for {len(documents)} docs"
            )

        if partial_embeddings is None:
            embedding_dim = int(batch_dim)
            partial_embeddings = np.lib.format.open_memmap(
                partial_embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(documents), embedding_dim),
            )
        elif embedding_dim != int(batch_dim):
            raise RuntimeError(
                "Inconsistent embedding dimension across batches: "
                f"{embedding_dim} != {batch_dim}"
            )

        partial_embeddings[start_index:end_index] = batch_embeddings
        if hasattr(partial_embeddings, "flush"):
            partial_embeddings.flush()

        num_completed = max(num_completed, end_index)
        checkpoint_metadata: dict[str, Any] = {
            "cache_key": cache_key,
            "provider": config.jina_provider,
            "model": config.jina_model_name,
            "task": config.jina_task,
            "truncate_dim": config.jina_truncate_dim,
            "batch_size": config.jina_batch_size,
            "num_documents": len(documents),
            "num_completed": num_completed,
            "embedding_dim": embedding_dim,
        }
        _write_json_atomic(partial_metadata_path, checkpoint_metadata)
        logger.info(
            "Checkpoint updated: %d/%d docs written to partial cache.",
            num_completed,
            len(documents),
        )

    if num_completed < len(documents):
        logger.info(
            "No complete embedding cache found. Computing docs %d-%d.",
            num_completed + 1,
            len(documents),
        )
        remaining_documents = documents[num_completed:]
        _compute_jina_embeddings(
            remaining_documents,
            config,
            batch_callback=_persist_partial_batch,
            start_index_offset=num_completed,
        )
    else:
        logger.info("All documents already present in partial checkpoint. Finalizing.")

    if not documents:
        embeddings = np.empty((0, 0), dtype=np.float32)
    else:
        if partial_embeddings is None:
            raise RuntimeError(
                "Embeddings computation finished without partial cache writes."
            )
        embeddings = np.array(partial_embeddings, dtype=np.float32, copy=True)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
            raise RuntimeError(
                "Final embeddings shape mismatch after checkpoint assembly: "
                f"{embeddings.shape} for {len(documents)} docs"
            )

    np.save(embeddings_path, embeddings)
    logger.info("Saved embeddings cache to %s", embeddings_path)
    partial_embeddings = None
    checkpoint_embeddings = None
    gc.collect()
    _cleanup_partial_cache()

    metadata = {
        "cache_key": cache_key,
        "provider": config.jina_provider,
        "model": config.jina_model_name,
        "task": config.jina_task,
        "truncate_dim": config.jina_truncate_dim,
        "batch_size": config.jina_batch_size,
        "device": config.jina_device if config.jina_provider == "local" else None,
        "api_url": config.jina_api_url if config.jina_provider == "api" else None,
        "num_documents": len(documents),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
    }
    _write_json_atomic(metadata_path, metadata)
    loaded_from_cache = initial_num_completed == len(documents) and len(documents) > 0
    return embeddings, embeddings_path, loaded_from_cache
