"""
DefenX-NLP public API.
"""

from .backends import (
    APIEncoderBackend,
    BaseEncoderBackend,
    EncoderBackendFactory,
    OnnxEncoderBackend,
    SentenceTransformerBackend,
)
from .device import device_info, get_device
from .encoder import SemanticEncoder
from .inference import PrototypeInferenceEngine
from .interfaces import BaseEncoder, BaseInferenceEngine, BaseRetriever, BaseVectorIndex
from .pipeline import NLPipeline
from .preprocessing import batch_clean, clean_text, truncate
from .retrieval import FaissVectorIndex, NumpyVectorIndex, SemanticSearchEngine
from .schemas import (
    DocumentRecord,
    EncoderConfig,
    LabelPrototype,
    PipelineResult,
    Prediction,
    PreprocessingConfig,
    SearchResult,
)
from .utils import (
    batch_cosine_similarity,
    cosine_similarity,
    normalize_batch,
    normalize_embedding,
    top_k_similar,
)

__version__ = "1.0.1"
__author__ = "DEFENX"
__email__ = "defenx@zohomail.in"
__license__ = "MIT"

__all__ = [
    "APIEncoderBackend",
    "BaseEncoder",
    "BaseEncoderBackend",
    "BaseInferenceEngine",
    "BaseRetriever",
    "BaseVectorIndex",
    "DocumentRecord",
    "EncoderBackendFactory",
    "EncoderConfig",
    "FaissVectorIndex",
    "LabelPrototype",
    "NLPipeline",
    "NumpyVectorIndex",
    "OnnxEncoderBackend",
    "PipelineResult",
    "Prediction",
    "PreprocessingConfig",
    "PrototypeInferenceEngine",
    "SearchResult",
    "SemanticEncoder",
    "SemanticSearchEngine",
    "SentenceTransformerBackend",
    "batch_clean",
    "batch_cosine_similarity",
    "clean_text",
    "cosine_similarity",
    "device_info",
    "get_device",
    "normalize_batch",
    "normalize_embedding",
    "top_k_similar",
    "truncate",
]
