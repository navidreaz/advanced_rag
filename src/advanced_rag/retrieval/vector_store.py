
from langchain_community.vectorstores import FAISS  # load faiss's libomp before torch's to avoid macOS segfaults
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from .embedding import EMBEDDING_MODEL_NAME, docs_processed
from langchain_community.vectorstores.utils import DistanceStrategy
import os # Two OpenMP runtimes loaded at once. FAISS and PyTorch each ship one (libomp and libiomp5)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

device = "mps" if torch.backends.mps.is_available() else "cpu"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=False,# for mac should be false
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}, # Set `True` for cosine similarity
)

KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
)