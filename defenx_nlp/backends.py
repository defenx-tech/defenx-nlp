"""
defenx_nlp.backends - Encoder backend strategies for DefenX-NLP V2.
"""

from __future__ import annotations

import logging
import os
import threading
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import torch

from .device import get_device
from .interfaces import BaseEncoder
from .schemas import EncoderConfig

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
for _log in ("sentence_transformers", "transformers", "huggingface_hub"):
    logging.getLogger(_log).setLevel(logging.ERROR)


class BaseEncoderBackend(BaseEncoder, ABC):
    """Strategy interface for concrete encoder backends."""

    def __init__(self, config: EncoderConfig) -> None:
        self._config = config

    @property
    def config(self) -> EncoderConfig:
        """Resolved backend configuration."""
        return self._config

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Stable backend identifier."""

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return self._config.model_name


class SentenceTransformerBackend(BaseEncoderBackend):
    """
    Thread-safe SentenceTransformer encoder backend.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__(config)
        self._device = get_device(config.device)
        self._model = None
        self._lock = threading.Lock()
        self._dim: int | None = None
        if not config.lazy:
            self._load_model()

    @property
    def backend_name(self) -> str:
        return "sentence-transformers"

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.debug(
            "Loading SentenceTransformer '%s' on %s",
            self._config.model_name,
            self._device,
        )
        try:
            self._model = SentenceTransformer(self._config.model_name, device=str(self._device))
        except Exception as exc:  # pragma: no cover - depends on local model cache/network
            raise RuntimeError(
                "Failed to load sentence-transformers model "
                f"'{self._config.model_name}'. Ensure the model is available locally "
                "or that this environment can reach Hugging Face, then try again."
            ) from exc
        with torch.inference_mode():
            probe = self._model.encode(
                "probe",
                convert_to_numpy=True,
                normalize_embeddings=self._config.normalize_embeddings,
                show_progress_bar=False,
            )
        self._dim = int(probe.shape[-1])

    def _ensure_loaded(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()

    def encode(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        with torch.inference_mode():
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self._config.normalize_embeddings,
                show_progress_bar=False,
            )
        return np.asarray(embedding, dtype=np.float32)

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        self._ensure_loaded()
        with torch.inference_mode():
            embeddings = self._model.encode(
                list(texts),
                batch_size=batch_size or self._config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self._config.normalize_embeddings,
                show_progress_bar=show_progress,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def warmup(self) -> None:
        self._ensure_loaded()
        _ = self.encode("warmup")

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        return int(self._dim)

    @property
    def device(self) -> torch.device:
        return self._device


class OnnxEncoderBackend(BaseEncoderBackend):
    """
    ONNX encoder backend stub.

    The class is fully integrated into the backend strategy layer but does
    not perform inference until an ONNX runtime implementation is provided.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__(config)
        self._device = get_device(config.device)
        self._embedding_dim = int(config.backend_options.get("embedding_dim", 0))
        self._model_path = config.backend_options.get("model_path")

    @property
    def backend_name(self) -> str:
        return "onnx"

    def _raise_not_ready(self) -> None:
        raise NotImplementedError(
            "ONNX backend is registered but not implemented. "
            "Provide an ONNX runtime strategy before calling encode()."
        )

    def encode(self, text: str) -> np.ndarray:
        self._raise_not_ready()

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        self._raise_not_ready()

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim <= 0:
            raise RuntimeError(
                "ONNX backend requires backend_options['embedding_dim'] to be configured."
            )
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device


class APIEncoderBackend(BaseEncoderBackend):
    """
    API encoder backend stub.

    The class defines the production contract for remote embedding services.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__(config)
        self._device = torch.device("cpu")
        self._embedding_dim = int(config.backend_options.get("embedding_dim", 0))
        self._endpoint = config.backend_options.get("endpoint")
        self._api_key_env = config.backend_options.get("api_key_env")

    @property
    def backend_name(self) -> str:
        return "api"

    def _raise_not_ready(self) -> None:
        raise NotImplementedError(
            "API backend is registered but not implemented. "
            "Provide a concrete remote embedding client before calling encode()."
        )

    def encode(self, text: str) -> np.ndarray:
        self._raise_not_ready()

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        self._raise_not_ready()

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim <= 0:
            raise RuntimeError(
                "API backend requires backend_options['embedding_dim'] to be configured."
            )
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device


class EncoderBackendFactory:
    """Factory for encoder backend strategies."""

    _REGISTRY: dict[str, type[BaseEncoderBackend]] = {
        "sentence-transformers": SentenceTransformerBackend,
        "sentence_transformers": SentenceTransformerBackend,
        "onnx": OnnxEncoderBackend,
        "api": APIEncoderBackend,
    }

    @classmethod
    def create(cls, config: EncoderConfig) -> BaseEncoderBackend:
        """
        Instantiate a backend implementation from configuration.
        """
        try:
            backend_cls = cls._REGISTRY[config.backend]
        except KeyError as exc:
            supported = ", ".join(sorted(set(cls._REGISTRY)))
            raise ValueError(
                f"Unsupported encoder backend '{config.backend}'. Supported: {supported}."
            ) from exc
        return backend_cls(config)

    @classmethod
    def register(cls, name: str, backend_cls: type[BaseEncoderBackend]) -> None:
        """Register a custom backend strategy."""
        cls._REGISTRY[name.strip().lower()] = backend_cls
