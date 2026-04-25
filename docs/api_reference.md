# defenx-nlp API Reference

## Core Objects

### `SemanticEncoder`

Backend-driven embedding facade.

```python
from defenx_nlp import SemanticEncoder

encoder = SemanticEncoder(
    model_name="all-MiniLM-L6-v2",
    device="auto",
    lazy=True,
    backend="sentence-transformers",
    batch_size=32,
    normalize_embeddings=False,
)
```

Important constructor arguments:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `"all-MiniLM-L6-v2"` | Sentence-transformers model name |
| `device` | `str` | `"auto"` | Device preference such as `"auto"`, `"cpu"`, `"cuda"`, `"mps"` |
| `lazy` | `bool` | `True` | Load the backend only when first used |
| `backend` | `str` | `"sentence-transformers"` | Backend identifier |
| `batch_size` | `int` | `32` | Default batch size for encoding |
| `normalize_embeddings` | `bool` | `False` | Return normalized embeddings from supported backends |
| `backend_options` | `Mapping[str, Any] \| None` | `None` | Backend-specific options |
| `config` | `EncoderConfig \| None` | `None` | Optional prebuilt config object |
| `backend_instance` | `BaseEncoderBackend \| None` | `None` | Inject a custom backend instance |

Methods and properties:

- `encode(text: str) -> np.ndarray`
- `encode_batch(texts: Sequence[str], *, batch_size: int | None = None, show_progress: bool = False) -> np.ndarray`
- `warmup() -> None`
- `from_config(config: EncoderConfig) -> SemanticEncoder`
- `config`
- `backend`
- `backend_name`
- `model_name`
- `embedding_dim`
- `device`

### `SemanticSearchEngine`

Semantic document retrieval over an encoder and vector index.

```python
from defenx_nlp import SemanticEncoder, SemanticSearchEngine

encoder = SemanticEncoder()
search = SemanticSearchEngine(encoder)
```

Constructor:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `encoder` | `BaseEncoder` | required | Encoder used for documents and queries |
| `vector_index` | `BaseVectorIndex \| None` | `None` | Custom index instance |
| `vector_index_factory` | `Callable[[], BaseVectorIndex] \| None` | `None` | Factory used when resetting the index |
| `text_preprocessor` | `Callable[[str], str] \| None` | `None` | Optional transform applied to indexed/query text |
| `batch_size` | `int \| None` | `None` | Batch size forwarded to `encode_batch()` |

Methods and properties:

- `index(documents: Sequence[str | DocumentRecord]) -> None`
- `search(query: str, top_k: int = 5) -> list[SearchResult]`
- `clear() -> None`
- `document_count`

### `PrototypeInferenceEngine`

Prototype-based inference over embeddings.

```python
from defenx_nlp import PrototypeInferenceEngine
```

Constructor:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prototypes` | `Sequence[LabelPrototype]` | required | Label prototypes used for scoring |
| `mode` | `"classification" \| "scoring"` | `"classification"` | Output mode |
| `temperature` | `float` | `1.0` | Softmax temperature in classification mode |

Helpers:

- `from_embeddings(label_embeddings, *, mode="classification", temperature=1.0, metadata=None)`
- `from_texts(encoder, label_examples, *, mode="classification", temperature=1.0, batch_size=None, metadata=None)`

Methods and properties:

- `infer(embedding: np.ndarray) -> Prediction`
- `infer_batch(embeddings: np.ndarray) -> list[Prediction]`
- `mode`
- `embedding_dim`

### `NLPipeline`

Simple end-to-end orchestration for preprocessing, encoding, and inference.

```python
from defenx_nlp import NLPipeline
```

Constructor:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `encoder` | `BaseEncoder` | required | Encoder used by the pipeline |
| `inference_engine` | `BaseInferenceEngine \| None` | `None` | Optional inference engine |
| `preprocessing_config` | `PreprocessingConfig \| None` | `None` | Default cleaning config |
| `enable_preprocessing` | `bool` | `True` | Enable or bypass preprocessing |
| `batch_size` | `int` | `32` | Preferred batch size for batch runs |
| `preprocessor` | `Callable[[str], str] \| None` | `None` | Custom text preprocessor |

Methods and properties:

- `run(text: str) -> PipelineResult`
- `run_batch(texts: Sequence[str]) -> list[PipelineResult]`
- `batch_size`

## Configuration and Data Schemas

### `EncoderConfig`

```python
from defenx_nlp import EncoderConfig
```

Fields:

- `backend: str = "sentence-transformers"`
- `model_name: str = "all-MiniLM-L6-v2"`
- `device: str = "auto"`
- `lazy: bool = True`
- `batch_size: int = 32`
- `normalize_embeddings: bool = False`
- `backend_options: dict[str, Any] = {}`

### `PreprocessingConfig`

Fields:

- `lowercase: bool = False`
- `remove_urls_flag: bool = False`
- `remove_emails_flag: bool = False`
- `remove_special: bool = False`
- `max_chars: int | None = None`

Helper:

- `to_clean_kwargs() -> dict[str, Any]`

### `DocumentRecord`

Structured document stored inside retrieval indexes.

Fields:

- `document_id: str`
- `text: str`
- `metadata: dict[str, Any]`

### `SearchResult`

Fields:

- `document: DocumentRecord`
- `score: float`
- `rank: int`

### `LabelPrototype`

Fields:

- `label: str`
- `embedding: np.ndarray`
- `metadata: dict[str, Any]`

### `Prediction`

Fields:

- `label: str`
- `score: float`
- `scores: dict[str, float]`
- `metadata: dict[str, Any]`

### `PipelineResult`

Fields:

- `raw_text: str`
- `processed_text: str`
- `embedding: np.ndarray`
- `prediction: Prediction | None`

## Backends and Interfaces

### Encoder backends

- `SentenceTransformerBackend`: implemented default backend
- `OnnxEncoderBackend`: contract stub, not implemented yet
- `APIEncoderBackend`: contract stub, not implemented yet
- `EncoderBackendFactory`: backend registry/factory

### Base interfaces

- `BaseEncoder`
- `BaseInferenceEngine`
- `BaseRetriever`
- `BaseVectorIndex`
- `BaseEncoderBackend`

These interfaces are the extension points if you want to plug in a custom
encoder, inference engine, or vector index.

## Retrieval Backends

### `NumpyVectorIndex`

Thread-safe cosine similarity index backed by NumPy.

Methods and properties:

- `build(vectors: np.ndarray) -> None`
- `search(query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]`
- `size`
- `dimension`

### `FaissVectorIndex`

Optional FAISS-backed vector index. Requires a compatible FAISS installation.

Methods and properties:

- `build(vectors: np.ndarray) -> None`
- `search(query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]`
- `size`
- `dimension`

## Utilities

### Device helpers

- `get_device(preferred="auto") -> torch.device`
- `device_info() -> dict[str, str | int | float | bool]`

### Preprocessing helpers

- `clean_text(text, *, lowercase=False, remove_urls_flag=False, remove_emails_flag=False, remove_special=False, max_chars=None) -> str`
- `batch_clean(texts, **clean_kwargs) -> list[str]`
- `truncate(text, max_chars=512, ellipsis=True) -> str`

### Similarity helpers

- `cosine_similarity(a, b) -> float`
- `batch_cosine_similarity(query, matrix) -> np.ndarray`
- `top_k_similar(query, corpus, k=5) -> list[tuple[int, float]]`
- `normalize_embedding(emb) -> np.ndarray`
- `normalize_batch(matrix) -> np.ndarray`
