"""
defenx_nlp.inference - Production inference engines for embeddings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .interfaces import BaseEncoder, BaseInferenceEngine
from .schemas import InferenceMode, LabelPrototype, Prediction
from .utils import normalize_batch, normalize_embedding


class PrototypeInferenceEngine(BaseInferenceEngine):
    """
    Prototype-based inference over semantic embeddings.

    Supports:
    - classification: returns probability-like scores via softmax
    - scoring: returns raw cosine scores against label prototypes
    """

    def __init__(
        self,
        prototypes: Sequence[LabelPrototype],
        *,
        mode: InferenceMode = "classification",
        temperature: float = 1.0,
    ) -> None:
        if not prototypes:
            raise ValueError("PrototypeInferenceEngine requires at least one prototype")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self._prototypes = list(prototypes)
        labels = [item.label for item in self._prototypes]
        if len(set(labels)) != len(labels):
            raise ValueError("PrototypeInferenceEngine labels must be unique")

        self._mode = mode
        self._temperature = float(temperature)
        self._labels = labels
        self._metadata = {item.label: dict(item.metadata) for item in self._prototypes}

        matrix = np.stack([item.embedding for item in self._prototypes], axis=0)
        if len({int(item.embedding.shape[0]) for item in self._prototypes}) != 1:
            raise ValueError("All prototype embeddings must have the same dimension")
        normalized = normalize_batch(matrix)
        self._prototype_matrix = normalized.astype(np.float32)
        self._embedding_dim = int(self._prototype_matrix.shape[1])

    @classmethod
    def from_embeddings(
        cls,
        label_embeddings: Mapping[str, np.ndarray],
        *,
        mode: InferenceMode = "classification",
        temperature: float = 1.0,
        metadata: Mapping[str, Mapping[str, object]] | None = None,
    ) -> "PrototypeInferenceEngine":
        """
        Build an inference engine from pre-computed label embeddings.
        """
        prototypes = [
            LabelPrototype(
                label=label,
                embedding=np.asarray(embedding, dtype=np.float32),
                metadata=dict((metadata or {}).get(label, {})),
            )
            for label, embedding in label_embeddings.items()
        ]
        return cls(prototypes, mode=mode, temperature=temperature)

    @classmethod
    def from_texts(
        cls,
        encoder: BaseEncoder,
        label_examples: Mapping[str, Sequence[str]],
        *,
        mode: InferenceMode = "classification",
        temperature: float = 1.0,
        batch_size: int | None = None,
        metadata: Mapping[str, Mapping[str, object]] | None = None,
    ) -> "PrototypeInferenceEngine":
        """
        Build label prototypes by averaging encoded example texts.
        """
        prototypes: list[LabelPrototype] = []
        for label, examples in label_examples.items():
            if not examples:
                raise ValueError(f"Label '{label}' requires at least one example text")
            embeddings = encoder.encode_batch(list(examples), batch_size=batch_size)
            centroid = embeddings.mean(axis=0).astype(np.float32)
            prototypes.append(
                LabelPrototype(
                    label=label,
                    embedding=centroid,
                    metadata=dict((metadata or {}).get(label, {})),
                )
            )
        return cls(prototypes, mode=mode, temperature=temperature)

    @property
    def mode(self) -> InferenceMode:
        """Inference mode: ``classification`` or ``scoring``."""
        return self._mode

    @property
    def embedding_dim(self) -> int:
        """Expected embedding dimension."""
        return self._embedding_dim

    def infer(self, embedding: np.ndarray) -> Prediction:
        scores = self._score_vector(embedding)
        return self._prediction_from_scores(scores)

    def infer_batch(self, embeddings: np.ndarray) -> list[Prediction]:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a 2-D array")
        if matrix.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self._embedding_dim}, got {matrix.shape[1]}"
            )
        normalized = normalize_batch(matrix)
        similarities = normalized @ self._prototype_matrix.T
        return [self._prediction_from_scores(row) for row in similarities]

    def _score_vector(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("embedding must be 1-D")
        if vector.shape[0] != self._embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self._embedding_dim}, got {vector.shape[0]}"
            )
        normalized = normalize_embedding(vector)
        return normalized @ self._prototype_matrix.T

    def _prediction_from_scores(self, raw_scores: np.ndarray) -> Prediction:
        if self._mode == "classification":
            scores = self._softmax(raw_scores / self._temperature)
        else:
            scores = raw_scores.astype(np.float32)

        best_index = int(np.argmax(scores))
        label = self._labels[best_index]
        score = float(scores[best_index])
        score_map = {self._labels[i]: float(scores[i]) for i in range(len(self._labels))}
        metadata = dict(self._metadata.get(label, {}))
        metadata["mode"] = self._mode
        return Prediction(label=label, score=score, scores=score_map, metadata=metadata)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exps = np.exp(shifted, dtype=np.float32)
        return exps / exps.sum()
