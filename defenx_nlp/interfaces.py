"""
defenx_nlp.interfaces - Extensible contracts for DefenX-NLP V2.

These interfaces define the stable boundaries between the encoder,
retrieval, inference, and pipeline layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import torch

from .schemas import DocumentRecord, Prediction, SearchResult


class BaseEncoder(ABC):
    """
    Abstract base class for all text encoders.

    Concrete implementations may use local transformer models, ONNX
    runtimes, remote APIs, or test doubles.
    """

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text into a float32 embedding.

        Parameters
        ----------
        text : str

        Returns
        -------
        np.ndarray
            Shape ``(embedding_dim,)``.
        """

    @abstractmethod
    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode multiple texts into a float32 embedding matrix.

        Parameters
        ----------
        texts : Sequence[str]
        batch_size : int | None
        show_progress : bool

        Returns
        -------
        np.ndarray
            Shape ``(N, embedding_dim)``.
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Execution device backing the encoder."""

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Return cosine similarity between two embeddings."""
        from .utils import cosine_similarity

        return cosine_similarity(a, b)

    def warmup(self) -> None:
        """Warm up the encoder implementation if needed."""
        self.encode("warmup")


class BaseInferenceEngine(ABC):
    """
    Abstract base class for downstream inference over embeddings.

    Inference engines consume encoder outputs and emit structured
    predictions suitable for production services.
    """

    @abstractmethod
    def infer(self, embedding: np.ndarray) -> Prediction:
        """
        Run inference for a single embedding.

        Parameters
        ----------
        embedding : np.ndarray
            Shape ``(embedding_dim,)``.

        Returns
        -------
        Prediction
        """

    @abstractmethod
    def infer_batch(self, embeddings: np.ndarray) -> list[Prediction]:
        """
        Run inference for a batch of embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(N, embedding_dim)``.

        Returns
        -------
        list[Prediction]
        """


class BaseVectorIndex(ABC):
    """
    Abstract vector-index contract for semantic retrieval backends.

    Implementations may wrap NumPy, FAISS, or any ANN engine.
    """

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """
        Build or replace the index with the provided vectors.

        Parameters
        ----------
        vectors : np.ndarray
            Shape ``(N, D)``.
        """

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """
        Search the index using a single query vector.

        Parameters
        ----------
        query_vector : np.ndarray
            Shape ``(D,)``.
        top_k : int

        Returns
        -------
        list[tuple[int, float]]
            Ordered as ``(row_index, score)``.
        """

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of indexed vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """Embedding dimensionality currently stored in the index."""


class BaseRetriever(ABC):
    """
    Abstract retrieval contract for semantic search systems.
    """

    @abstractmethod
    def index(self, documents: Sequence[str | DocumentRecord]) -> None:
        """Index a new document corpus."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the indexed corpus for semantically similar documents."""

    @property
    @abstractmethod
    def document_count(self) -> int:
        """Number of indexed documents."""
