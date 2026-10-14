# My change for reranker
from sentence_transformers import CrossEncoder

RERANKER = CrossEncoder("BAAI/bge-reranker-v2-m3")
