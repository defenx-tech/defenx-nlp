"""
defenx_nlp.pipeline - Configurable end-to-end NLP pipeline orchestration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .interfaces import BaseEncoder, BaseInferenceEngine
from .preprocessing import batch_clean, clean_text
from .schemas import PipelineResult, PreprocessingConfig


class NLPipeline:
    """
    Configurable V2 pipeline:

        raw text -> preprocess -> encode -> infer -> structured output
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        *,
        inference_engine: BaseInferenceEngine | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        enable_preprocessing: bool = True,
        batch_size: int = 32,
        preprocessor: Callable[[str], str] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._encoder = encoder
        self._inference_engine = inference_engine
        self._preprocessing_config = preprocessing_config or PreprocessingConfig()
        self._enable_preprocessing = enable_preprocessing
        self._batch_size = batch_size
        self._preprocessor = preprocessor

    @property
    def batch_size(self) -> int:
        """Preferred encoder batch size for pipeline batch execution."""
        return self._batch_size

    def run(self, text: str) -> PipelineResult:
        """
        Execute the pipeline for a single text.
        """
        text = str(text)
        processed = self._preprocess_one(text)
        embedding = self._encoder.encode(processed)
        prediction = None
        if self._inference_engine is not None:
            prediction = self._inference_engine.infer(embedding)
        return PipelineResult(
            raw_text=text,
            processed_text=processed,
            embedding=embedding,
            prediction=prediction,
        )

    def run_batch(self, texts: Sequence[str]) -> list[PipelineResult]:
        """
        Execute the pipeline for multiple texts using batch encoding.
        """
        raw_texts = [str(text) for text in texts]
        if not raw_texts:
            return []

        processed_texts = self._preprocess_many(raw_texts)
        embeddings = self._encoder.encode_batch(
            processed_texts,
            batch_size=self._batch_size,
        )
        predictions = None
        if self._inference_engine is not None:
            predictions = self._inference_engine.infer_batch(embeddings)

        outputs: list[PipelineResult] = []
        for index, raw_text in enumerate(raw_texts):
            outputs.append(
                PipelineResult(
                    raw_text=raw_text,
                    processed_text=processed_texts[index],
                    embedding=embeddings[index],
                    prediction=None if predictions is None else predictions[index],
                )
            )
        return outputs

    def _preprocess_one(self, text: str) -> str:
        if not self._enable_preprocessing:
            return text
        if self._preprocessor is not None:
            return self._preprocessor(text)
        return clean_text(text, **self._preprocessing_config.to_clean_kwargs())

    def _preprocess_many(self, texts: Sequence[str]) -> list[str]:
        if not self._enable_preprocessing:
            return list(texts)
        if self._preprocessor is not None:
            return [self._preprocessor(text) for text in texts]
        return batch_clean(list(texts), **self._preprocessing_config.to_clean_kwargs())
