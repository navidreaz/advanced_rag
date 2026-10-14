# AdvancedRAG

A lightweight retrieval-augmented generation (RAG) project built on LangChain, Hugging Face embeddings, FAISS, and a local reader model. It loads a public Hugging Face knowledge base, splits it into chunks, embeds the chunks, stores them in a vector database, and answers questions by combining retrieval with reranking and generation.

## What this project does

The project demonstrates a complete local RAG flow:

- Ingests a knowledge dataset from Hugging Face
- Splits documents into token-aware chunks
- Embeds chunks with a sentence-transformer model
- Stores embeddings in FAISS
- Reranks the most relevant results with a cross-encoder model
- Builds a final answer with a reader LLM

This is a practical example for experimenting with local AI pipelines without depending on a remote hosted vector database or model endpoint.

## Repository layout

```text
AdvancedRAG/
├── README.md
├── pyproject.toml
├── notebooks/
├── src/
│   ├── main.py
│   └── advanced_rag/
│       ├── __init__.py
│       ├── generation/
│       │   ├── prompt.py
│       │   ├── reader_model.py
│       │   └── __pycache__/
│       ├── ingestion/
│       │   └── loaders.py
│       ├── rag/
│       │   └── pipeline.py
│       └── retrieval/
│           ├── embedding.py
│           ├── reranker.py
│           ├── retriever.py
│           └── vector_store.py
└── .gitignore
```

## Core components

- `src/advanced_rag/ingestion/loaders.py`
 Loads the Hugging Face knowledge dataset used as the source corpus.

- `src/advanced_rag/retrieval/embedding.py`
 Splits documents and prepares the embedding model.

- `src/advanced_rag/retrieval/vector_store.py`
 Creates the FAISS knowledge base using cosine similarity.

- `src/advanced_rag/retrieval/reranker.py`
 Uses a BAAI reranker to improve the retrieved document ranking.

- `src/advanced_rag/generation/reader_model.py`
 Loads the reader LLM used to generate answers from the retrieved context.

- `src/advanced_rag/rag/pipeline.py`
 Orchestrates retrieval, reranking, prompt construction, and answer generation.

## Tech stack

- Python 3.11
- LangChain + LangChain Community
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- PyTorch
- MLX for local model inference on Apple Silicon

## Installation

Create a virtual environment and install the project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Running the demo

From the project root:

```bash
PYTHONPATH=src python src/main.py
```

This will:

1. Build the knowledge index from the dataset
2. Retrieve relevant text for the sample prompt
3. Rerank the retrieved documents
4. Generate an answer using the reader model
5. Print the answer and source documents

## Example output behavior

The current sample question in `src/main.py` is:

```python
question = "how to create a pipeline object?"
```

The script prints:

- the generated answer
- the most relevant document snippets used as sources

## Notes and requirements

- Model downloads happen at runtime, so the first run can take a while.
- This project is optimized for local experimentation and may require a large amount of RAM/VRAM depending on the model you run.
- The default reader model is `mlx-community/Qwen2.5-7B-Instruct-4bit` and the embedding model is `thenlper/gte-small`.
- On macOS, the code sets MPS/CPU device handling and OpenMP environment variables to reduce compatibility issues.

## Future improvements

Possible next steps for this project include:

- adding an API layer with FastAPI
- supporting custom local document ingestion
- exposing configuration via environment variables or YAML
- adding evaluation scripts for retrieval and answer quality
- adding unit/integration tests

## License

This project does not currently declare a license. Add one if you plan to distribute or publish it publicly.

