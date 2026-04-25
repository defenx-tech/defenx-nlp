"""
defenx_nlp.retrieval - Semantic retrieval primitives and search engine.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from collections.abc import Sequence

import numpy as np

from .interfaces import BaseEncoder, BaseRetriever, BaseVectorIndex
from .schemas import DocumentRecord, SearchResult
from .utils import normalize_batch, normalize_embedding


class NumpyVectorIndex(BaseVectorIndex):
    """
    Thread-safe cosine-similarity vector index built on NumPy.
    """

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None
        self._dimension: int | None = None
        self._lock = threading.RLock()

    def build(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        if matrix.shape[0] == 0:
            raise ValueError("vectors must contain at least one row")
        with self._lock:
            self._matrix = normalize_batch(matrix).copy()
            self._dimension = int(matrix.shape[1])

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_vector must be 1-D")

        with self._lock:
            if self._matrix is None or self._matrix.shape[0] == 0:
                return []
            if self._dimension is not None and query.shape[0] != self._dimension:
                raise ValueError(
                    f"Expected query dimension {self._dimension}, got {query.shape[0]}"
                )
            normalized_query = normalize_embedding(query)
            scores = self._matrix @ normalized_query
            limit = min(top_k, scores.shape[0])
            top_indices = np.argpartition(scores, -limit)[-limit:]
            ordered = top_indices[np.argsort(scores[top_indices])[::-1]]
            return [(int(index), float(scores[index])) for index in ordered]

    @property
    def size(self) -> int:
        with self._lock:
            return 0 if self._matrix is None else int(self._matrix.shape[0])

    @property
    def dimension(self) -> int | None:
        with self._lock:
            return self._dimension


class FaissVectorIndex(BaseVectorIndex):
    """
    Optional FAISS-backed vector index.
    """

    def __init__(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "FAISS is not installed. Install faiss-cpu or faiss-gpu to use this index."
            ) from exc

        self._faiss = faiss
        self._index = None
        self._dimension: int | None = None
        self._size = 0
        self._lock = threading.RLock()

    def build(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        if matrix.shape[0] == 0:
            raise ValueError("vectors must contain at least one row")

        normalized = normalize_batch(matrix)
        with self._lock:
            self._dimension = int(normalized.shape[1])
            self._index = self._faiss.IndexFlatIP(self._dimension)
            self._index.add(normalized)
            self._size = int(normalized.shape[0])

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_vector must be 1-D")

        with self._lock:
            if self._index is None or self._size == 0:
                return []
            if self._dimension is not None and query.shape[0] != self._dimension:
                raise ValueError(
                    f"Expected query dimension {self._dimension}, got {query.shape[0]}"
                )
            normalized_query = normalize_embedding(query).reshape(1, -1)
            scores, indices = self._index.search(normalized_query, min(top_k, self._size))
            return [
                (int(indices[0, position]), float(scores[0, position]))
                for position in range(indices.shape[1])
                if int(indices[0, position]) >= 0
            ]

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    @property
    def dimension(self) -> int | None:
        with self._lock:
            return self._dimension


class SemanticSearchEngine(BaseRetriever):
    """
    Production semantic search engine integrated with a DefenX encoder.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        *,
        vector_index: BaseVectorIndex | None = None,
        vector_index_factory: Callable[[], BaseVectorIndex] | None = None,
        text_preprocessor: Callable[[str], str] | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._encoder = encoder
        self._vector_index = vector_index or NumpyVectorIndex()
        self._vector_index_factory = vector_index_factory
        self._documents: list[DocumentRecord] = []
        self._text_preprocessor = text_preprocessor
        self._batch_size = batch_size
        self._lock = threading.RLock()

    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        with self._lock:
            return len(self._documents)

    def index(self, documents: Sequence[str | DocumentRecord]) -> None:
        """
        Build a retrieval index from raw texts or ``DocumentRecord`` objects.
        """
        prepared = self._coerce_documents(documents)
        texts = [self._transform_text(doc.text) for doc in prepared]
        embeddings = self._encoder.encode_batch(texts, batch_size=self._batch_size)
        with self._lock:
            self._vector_index.build(embeddings)
            self._documents = prepared

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Execute semantic search against the indexed corpus.
        """
        with self._lock:
            if not self._documents:
                raise RuntimeError("Search index is empty. Call index() before search().")

        query_vector = self._encoder.encode(self._transform_text(query))
        with self._lock:
            documents = tuple(self._documents)
            matches = self._vector_index.search(query_vector, top_k)
            return [
                SearchResult(
                    document=documents[row_index],
                    score=score,
                    rank=rank,
                )
                for rank, (row_index, score) in enumerate(matches, start=1)
            ]

    def clear(self) -> None:
        """Remove all indexed documents and reset the vector index."""
        with self._lock:
            self._documents = []
            self._vector_index = self._make_empty_index()

    @staticmethod
    def _coerce_documents(documents: Sequence[str | DocumentRecord]) -> list[DocumentRecord]:
        prepared: list[DocumentRecord] = []
        for index, document in enumerate(documents):
            if isinstance(document, DocumentRecord):
                prepared.append(document)
                continue
            prepared.append(
                DocumentRecord(
                    document_id=f"doc-{index}",
                    text=str(document),
                    metadata={},
                )
            )
        if not prepared:
            raise ValueError("At least one document is required for indexing")
        ids = [document.document_id for document in prepared]
        if len(set(ids)) != len(ids):
            raise ValueError("DocumentRecord.document_id values must be unique")
        return prepared

    def _transform_text(self, text: str) -> str:
        if self._text_preprocessor is None:
            return text
        return self._text_preprocessor(text)

    def _make_empty_index(self) -> BaseVectorIndex:
        if self._vector_index_factory is not None:
            return self._vector_index_factory()
        try:
            return type(self._vector_index)()
        except TypeError as exc:
            raise RuntimeError(
                "Cannot clear SemanticSearchEngine because the vector index cannot be "
                "reconstructed without constructor arguments. Pass vector_index_factory="
                " when constructing the search engine."
            ) from exc
