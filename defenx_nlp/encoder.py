"""
defenx_nlp.encoder - Backend-driven semantic encoder facade.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .backends import BaseEncoderBackend, EncoderBackendFactory
from .interfaces import BaseEncoder
from .schemas import EncoderConfig


class SemanticEncoder(BaseEncoder):
    """
    Production encoder facade with pluggable backend strategies.

    The default backend uses SentenceTransformers, while ONNX and API
    backends are exposed through the same interface for orchestration code.
    """

    _DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
        lazy: bool = True,
        *,
        backend: str = "sentence-transformers",
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        backend_options: Mapping[str, Any] | None = None,
        config: EncoderConfig | None = None,
        backend_instance: BaseEncoderBackend | None = None,
    ) -> None:
        if config is None:
            config = EncoderConfig(
                backend=backend,
                model_name=model_name,
                device=device,
                lazy=lazy,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                backend_options=dict(backend_options or {}),
            )
        if backend_instance is not None:
            backend_name = backend_instance.backend_name.replace("_", "-")
            config_name = config.backend.replace("_", "-")
            if config_name != backend_name:
                raise ValueError(
                    "backend_instance does not match EncoderConfig.backend: "
                    f"'{backend_name}' != '{config_name}'"
                )
        self._config = config
        self._backend = backend_instance or EncoderBackendFactory.create(config)

    @classmethod
    def from_config(cls, config: EncoderConfig) -> "SemanticEncoder":
        """Build a semantic encoder from a concrete config object."""
        return cls(config=config)

    @property
    def config(self) -> EncoderConfig:
        """Resolved encoder configuration."""
        return self._config

    @property
    def backend(self) -> BaseEncoderBackend:
        """Concrete backend strategy instance."""
        return self._backend

    @property
    def backend_name(self) -> str:
        """Stable backend identifier."""
        return self._backend.backend_name

    @property
    def model_name(self) -> str:
        """Underlying model identifier."""
        return self._backend.model_name

    def encode(self, text: str) -> np.ndarray:
        return self._backend.encode(text)

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        return self._backend.encode_batch(
            list(texts),
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def warmup(self) -> None:
        self._backend.warmup()

    @property
    def embedding_dim(self) -> int:
        return self._backend.embedding_dim

    @property
    def device(self) -> torch.device:
        return self._backend.device

    def __repr__(self) -> str:
        return (
            "SemanticEncoder("
            f"backend='{self.backend_name}', "
            f"model='{self.model_name}', "
            f"device={self.device}, "
            f"lazy={self.config.lazy})"
        )
