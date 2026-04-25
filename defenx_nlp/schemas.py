"""
defenx_nlp.schemas - Shared immutable-style data contracts for V2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

InferenceMode = Literal["classification", "scoring"]


@dataclass(slots=True)
class EncoderConfig:
    """Configuration for encoder backend construction."""

    backend: str = "sentence-transformers"
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "auto"
    lazy: bool = True
    batch_size: int = 32
    normalize_embeddings: bool = False
    backend_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend = self.backend.strip().lower()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.backend_options = dict(self.backend_options)


@dataclass(slots=True)
class PreprocessingConfig:
    """Configuration for the default preprocessing stage."""

    lowercase: bool = False
    remove_urls_flag: bool = False
    remove_emails_flag: bool = False
    remove_special: bool = False
    max_chars: int | None = None

    def to_clean_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments compatible with ``clean_text``."""
        return {
            "lowercase": self.lowercase,
            "remove_urls_flag": self.remove_urls_flag,
            "remove_emails_flag": self.remove_emails_flag,
            "remove_special": self.remove_special,
            "max_chars": self.max_chars,
        }


@dataclass(slots=True)
class Prediction:
    """Structured inference output for one embedding."""

    label: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LabelPrototype:
    """Prototype embedding associated with a prediction label."""

    label: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.embedding = np.asarray(self.embedding, dtype=np.float32)
        if self.embedding.ndim != 1:
            raise ValueError("LabelPrototype.embedding must be 1-D")


@dataclass(slots=True)
class DocumentRecord:
    """Document stored inside the semantic retrieval index."""

    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    """A ranked semantic-search match."""

    document: DocumentRecord
    score: float
    rank: int


@dataclass(slots=True)
class PipelineResult:
    """End-to-end NLP pipeline output for one input text."""

    raw_text: str
    processed_text: str
    embedding: np.ndarray
    prediction: Prediction | None = None
