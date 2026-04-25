"""
Unit tests for DefenX-NLP V2 orchestration components.
"""

import unittest
from pathlib import Path

import numpy as np
import torch

from defenx_nlp import (
    APIEncoderBackend,
    BaseEncoder,
    BaseEncoderBackend,
    EncoderBackendFactory,
    EncoderConfig,
    NLPipeline,
    OnnxEncoderBackend,
    Prediction,
    PreprocessingConfig,
    PrototypeInferenceEngine,
    SemanticEncoder,
    SemanticSearchEngine,
)


class MockEncoder(BaseEncoder):
    """Deterministic CPU encoder for repeatable tests."""

    def __init__(self, embedding_dim: int = 6) -> None:
        self._embedding_dim = embedding_dim
        self._device = torch.device("cpu")
        self._keyword_weights = {
            "help": 0,
            "support": 0,
            "reset": 0,
            "password": 0,
            "login": 1,
            "account": 1,
            "refund": 2,
            "invoice": 3,
            "billing": 3,
            "charge": 4,
            "payment": 4,
            "anomaly": 5,
            "critical": 5,
            "failure": 5,
            "incident": 5,
            "healthy": 0,
            "normal": 0,
            "baseline": 0,
        }

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self._embedding_dim, dtype=np.float32)
        for raw_token in text.lower().split():
            token = "".join(char for char in raw_token if char.isalnum())
            if not token:
                continue
            if token in self._keyword_weights:
                vector[self._keyword_weights[token]] += 2.0
            seed = sum(ord(char) for char in token)
            vector[seed % self._embedding_dim] += 0.25
            vector[(seed // 11) % self._embedding_dim] += 0.1
        if not np.any(vector):
            vector[0] = 1.0
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


class RecordingBackend(BaseEncoderBackend):
    """Small backend used to test SemanticEncoder without external downloads."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__(config)
        self._device = torch.device("cpu")

    @property
    def backend_name(self) -> str:
        return "api"

    def encode(self, text: str) -> np.ndarray:
        return np.array([len(text), 1.0], dtype=np.float32)

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
        return 2

    @property
    def device(self) -> torch.device:
        return self._device


class TestV2(unittest.TestCase):
    def test_prototype_inference_engine_classification(self) -> None:
        encoder = MockEncoder()
        engine = PrototypeInferenceEngine.from_texts(
            encoder,
            {
                "support": ["help me log in", "reset my password"],
                "billing": ["issue refund", "incorrect invoice charge"],
            },
        )

        prediction = engine.infer(encoder.encode("please help me reset access"))

        self.assertIsInstance(prediction, Prediction)
        self.assertEqual(prediction.label, "support")
        self.assertGreater(prediction.score, 0.5)
        self.assertEqual(set(prediction.scores), {"support", "billing"})

    def test_prototype_inference_engine_scoring_mode(self) -> None:
        encoder = MockEncoder()
        engine = PrototypeInferenceEngine.from_texts(
            encoder,
            {
                "healthy": ["normal event baseline"],
                "anomaly": ["critical failure incident"],
            },
            mode="scoring",
        )

        prediction = engine.infer(encoder.encode("critical system incident"))

        self.assertEqual(prediction.label, "anomaly")
        self.assertGreater(prediction.score, prediction.scores["healthy"])

    def test_semantic_search_engine_returns_ranked_matches(self) -> None:
        encoder = MockEncoder()
        engine = SemanticSearchEngine(encoder)
        engine.index(
            [
                "Reset your password securely",
                "View your latest invoice",
                "Troubleshoot multi factor login issues",
            ]
        )

        matches = engine.search("i cannot log in to my account", top_k=2)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].rank, 1)
        self.assertIn("login", matches[0].document.text.lower())

    def test_pipeline_runs_preprocess_encode_and_infer(self) -> None:
        encoder = MockEncoder()
        inference = PrototypeInferenceEngine.from_texts(
            encoder,
            {
                "support": ["help reset account", "unlock my login"],
                "billing": ["refund this charge", "invoice problem"],
            },
        )
        pipeline = NLPipeline(
            encoder,
            inference_engine=inference,
            preprocessing_config=PreprocessingConfig(lowercase=True, remove_special=True),
        )

        result = pipeline.run("HELP! Reset my ACCOUNT please!!!")

        self.assertEqual(result.raw_text, "HELP! Reset my ACCOUNT please!!!")
        self.assertEqual(result.processed_text, "help! reset my account please!!!")
        self.assertEqual(result.embedding.shape, (encoder.embedding_dim,))
        self.assertIsNotNone(result.prediction)
        self.assertEqual(result.prediction.label, "support")

    def test_pipeline_batch_mode_preserves_order(self) -> None:
        encoder = MockEncoder()
        inference = PrototypeInferenceEngine.from_texts(
            encoder,
            {
                "support": ["help me login"],
                "billing": ["refund my invoice"],
            },
        )
        pipeline = NLPipeline(encoder, inference_engine=inference)

        results = pipeline.run_batch(
            [
                "help me login now",
                "refund my invoice today",
            ]
        )

        self.assertEqual(
            [result.prediction.label for result in results],
            ["support", "billing"],
        )

    def test_semantic_encoder_accepts_custom_backend_instance(self) -> None:
        config = EncoderConfig(backend="onnx", backend_options={"embedding_dim": 6})
        backend = OnnxEncoderBackend(config)
        encoder = SemanticEncoder(config=config, backend_instance=backend)

        self.assertEqual(encoder.backend_name, "onnx")
        self.assertEqual(encoder.embedding_dim, 6)
        with self.assertRaises(NotImplementedError):
            encoder.encode("hello")

    def test_backend_factory_creates_registered_stubs(self) -> None:
        onnx_backend = EncoderBackendFactory.create(
            EncoderConfig(backend="onnx", backend_options={"embedding_dim": 8})
        )
        api_backend = EncoderBackendFactory.create(
            EncoderConfig(backend="api", backend_options={"embedding_dim": 8})
        )

        self.assertIsInstance(onnx_backend, OnnxEncoderBackend)
        self.assertIsInstance(api_backend, APIEncoderBackend)

    def test_semantic_encoder_delegates_to_custom_backend(self) -> None:
        config = EncoderConfig(backend="api", backend_options={"embedding_dim": 2})
        backend = RecordingBackend(config)
        encoder = SemanticEncoder(config=config, backend_instance=backend)

        vector = encoder.encode("hello")
        matrix = encoder.encode_batch(["hi", "team"])

        np.testing.assert_array_equal(vector, np.array([5.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(
            matrix,
            np.array([[2.0, 1.0], [4.0, 1.0]], dtype=np.float32),
        )
