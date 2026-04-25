# defenx-nlp

Lightweight semantic NLP building blocks for Python.

`defenx-nlp` gives you one interface for text embeddings, semantic retrieval,
prototype-based inference, and simple end-to-end NLP pipelines. It is designed
for developers who want production-friendly primitives without wiring the same
boilerplate in every project.

[![PyPI version](https://img.shields.io/pypi/v/defenx-nlp)](https://pypi.org/project/defenx-nlp/)
[![Python](https://img.shields.io/pypi/pyversions/defenx-nlp)](https://pypi.org/project/defenx-nlp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What It Does

The package currently covers four layers:

- `SemanticEncoder`: a backend-driven embedding facade for local transformer models.
- `SemanticSearchEngine`: semantic indexing and retrieval over embedded documents.
- `PrototypeInferenceEngine`: lightweight embedding-based classification and scoring.
- `NLPipeline`: preprocessing -> encode -> infer orchestration with structured output.

This makes the project useful for:

- support ticket routing
- internal knowledge search
- FAQ and help center retrieval
- anomaly or incident scoring
- semantic deduplication and clustering
- retrieval-augmented backends

## Who Uses It

This is primarily a developer library, not a direct end-user application.

Typical users are:

- Python backend developers
- ML engineers building semantic features
- support tooling teams
- security/SOC teams experimenting with event similarity
- teams building internal search or classification workflows

End users would normally interact with it indirectly inside:

- a FastAPI or Flask service
- a chatbot or RAG system
- a support desk platform
- an admin dashboard
- a data processing or analytics job

## Architecture

![defenx-nlp architecture](docs/architecture.png)

## Installation

### Standard install

```bash
pip install defenx-nlp
```

This installs the package and its core dependencies for a normal CPU workflow.

### CUDA install

If you want a CUDA-enabled PyTorch build, reinstall `torch` with the matching
wheel after installing the package:

```bash
pip install defenx-nlp
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

### Development install

```bash
git clone https://github.com/defenx-sec/defenx-nlp.git
cd defenx-nlp
pip install -e ".[dev]"
```

## Quick Start

### 1. Encode text

```python
from defenx_nlp import SemanticEncoder

enc = SemanticEncoder()

embedding = enc.encode("Neural networks are useful for semantic search.")
print(embedding.shape)  # (384,)

embeddings = enc.encode_batch(["hello", "goodbye", "help me"])
print(embeddings.shape)  # (3, 384)
```

### 2. Semantic retrieval

```python
from defenx_nlp import SemanticEncoder, SemanticSearchEngine

enc = SemanticEncoder()
search = SemanticSearchEngine(enc)

search.index(
    [
        "Reset your password",
        "Check your latest invoice",
        "Troubleshoot login issues",
    ]
)

results = search.search("I cannot sign in to my account", top_k=2)
for match in results:
    print(match.rank, round(match.score, 3), match.document.text)
```

### 3. Prototype-based classification

```python
from defenx_nlp import SemanticEncoder, PrototypeInferenceEngine

enc = SemanticEncoder()
engine = PrototypeInferenceEngine.from_texts(
    enc,
    {
        "support": ["reset password", "cannot log in", "account help"],
        "billing": ["charged twice", "refund request", "invoice issue"],
    },
)

prediction = engine.infer(enc.encode("please help me reset my login"))
print(prediction.label)
print(prediction.score)
```

### 4. Run a simple pipeline

```python
from defenx_nlp import (
    NLPipeline,
    PreprocessingConfig,
    PrototypeInferenceEngine,
    SemanticEncoder,
)

enc = SemanticEncoder()
inference = PrototypeInferenceEngine.from_texts(
    enc,
    {
        "support": ["reset password", "login problem"],
        "billing": ["refund request", "invoice problem"],
    },
)

pipeline = NLPipeline(
    enc,
    inference_engine=inference,
    preprocessing_config=PreprocessingConfig(lowercase=True),
)

result = pipeline.run("HELP! I cannot access my account.")
print(result.processed_text)
print(result.prediction.label)
```

## Why Use This Instead Of Raw sentence-transformers?

You can absolutely use `sentence-transformers` directly. This project becomes
helpful when you want a cleaner application-facing layer around embeddings.

| Problem | Raw sentence-transformers | defenx-nlp |
| --- | --- | --- |
| Device selection | You handle CPU/CUDA/MPS decisions yourself | `get_device()` is built in |
| Service-friendly facade | Model code leaks into app logic | `SemanticEncoder` keeps a stable interface |
| Retrieval layer | You wire indexing and ranking yourself | `SemanticSearchEngine` is ready to use |
| Simple classifier | You build your own prototype scoring | `PrototypeInferenceEngine` is included |
| End-to-end flow | You orchestrate each step manually | `NLPipeline` returns structured results |
| Output consistency | Mix of tensors/arrays depending on flags | Returns `float32` NumPy arrays |

## API Summary

| Symbol | Description |
| --- | --- |
| `SemanticEncoder` | Main embedding facade |
| `SemanticSearchEngine` | Document indexing and semantic retrieval |
| `NumpyVectorIndex` | NumPy-based cosine similarity index |
| `FaissVectorIndex` | Optional FAISS-backed vector index |
| `PrototypeInferenceEngine` | Prototype-based classifier/scoring engine |
| `NLPipeline` | Preprocess -> encode -> infer pipeline |
| `EncoderConfig` | Backend configuration object |
| `PreprocessingConfig` | Cleaning/truncation config for the pipeline |
| `DocumentRecord` | Structured retrieval document |
| `SearchResult` | Ranked retrieval result |
| `Prediction` | Structured inference output |
| `PipelineResult` | Structured pipeline output |
| `clean_text`, `batch_clean`, `truncate` | Preprocessing helpers |
| `cosine_similarity`, `batch_cosine_similarity` | Similarity helpers |
| `normalize_embedding`, `normalize_batch` | L2 normalization helpers |

Full API docs: [docs/api_reference.md](docs/api_reference.md)

## Backends

The default backend is `sentence-transformers`.

The package also exports backend contracts for future extension:

- `SentenceTransformerBackend`: implemented and production-usable
- `OnnxEncoderBackend`: interface stub, not implemented yet
- `APIEncoderBackend`: interface stub, not implemented yet

If you expose ONNX or remote API backends publicly, label them as experimental
until they perform real inference.

## Examples

```bash
python examples/basic_usage.py
python examples/batch_encoding.py
python examples/v2_pipeline.py
```

## Testing

```bash
pytest tests -v
```

The test suite contains both:

- pure local unit tests for retrieval, inference, and pipeline logic
- integration-style encoder tests that require the default model to be locally
  available or downloadable

If the environment cannot reach Hugging Face and the model is not cached, the
integration tests skip instead of failing the entire local test run.

## Project Structure

```text
defenx-nlp/
|-- defenx_nlp/
|   |-- __init__.py
|   |-- backends.py
|   |-- device.py
|   |-- encoder.py
|   |-- inference.py
|   |-- interfaces.py
|   |-- pipeline.py
|   |-- preprocessing.py
|   |-- retrieval.py
|   |-- schemas.py
|   `-- utils.py
|-- docs/
|   |-- api_reference.md
|   `-- architecture.png
|-- examples/
|   |-- basic_usage.py
|   |-- batch_encoding.py
|   `-- v2_pipeline.py
|-- tests/
|   |-- test_encoder.py
|   `-- test_v2.py
|-- pyproject.toml
`-- README.md
```

## Roadmap

Good next milestones for the project:

- implement the ONNX backend
- implement a real API embedding backend
- add persistence helpers for vector indexes
- add FastAPI service examples
- expand benchmark coverage for CPU vs CUDA vs FAISS
- publish hosted documentation

## License

MIT. See [LICENSE](LICENSE).
