"""
Production V2 pipeline example.

Run:
    python examples/v2_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defenx_nlp import (
    BaseEncoder,
    NLPipeline,
    PreprocessingConfig,
    PrototypeInferenceEngine,
    SemanticSearchEngine,
)


class DemoEncoder(BaseEncoder):
    """Small deterministic encoder used for local examples and tests."""

    def __init__(self, embedding_dim: int = 8) -> None:
        self._embedding_dim = embedding_dim
        self._device = torch.device("cpu")

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self._embedding_dim, dtype=np.float32)
        for index, token in enumerate(text.lower().split()):
            bucket = index % self._embedding_dim
            vector[bucket] += float(sum(ord(char) for char in token) % 97)
        return vector

    def encode_batch(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        return np.stack([self.encode(text) for text in texts], axis=0)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device


def main() -> None:
    encoder = DemoEncoder()

    inference = PrototypeInferenceEngine.from_texts(
        encoder,
        {
            "support": [
                "help me reset my password",
                "i need account assistance",
            ],
            "billing": [
                "refund my payment",
                "i was charged twice",
            ],
        },
    )

    pipeline = NLPipeline(
        encoder,
        inference_engine=inference,
        preprocessing_config=PreprocessingConfig(lowercase=True),
    )

    result = pipeline.run("Please help me with my account login")
    print(result.prediction)

    search = SemanticSearchEngine(encoder)
    search.index(
        [
            "Resetting your password",
            "Understanding your invoice",
            "Troubleshooting login failures",
        ]
    )

    matches = search.search("how do i sign in again", top_k=2)
    for match in matches:
        print(match.rank, match.score, match.document.text)


if __name__ == "__main__":
    main()
