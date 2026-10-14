import datasets
from langchain_core.documents import Document as LangchainDocument
from tqdm import tqdm

# Load knowledge base

ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in tqdm(ds)
]
