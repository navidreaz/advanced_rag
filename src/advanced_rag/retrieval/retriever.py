from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..ingestion.loaders import RAW_KNOWLEDGE_BASE
import re
from transformers import AutoTokenizer
from tqdm import tqdm

# We use a hierarchical list of separators specifically tailored for splitting Markdown documents
# This list is taken from LangChain's MarkdownTextSplitter class
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size =1000, # The maximum size of each chunk in characters: we selected this value arbitrarily.
    chunk_overlap = 100, # The number of characters to overlap between chunks: we selected this value arbitrarily.
    add_start_index = True, # if `True`, include the chunk's start index in metadata.
    # strip_multiple_whitespaces=True, # if `True`, strip multiple whitespaces from the start and end of every document.
    # separatevars= MARKDOWN_SEPARATORS,
    keep_separator=True,
    separators=MARKDOWN_SEPARATORS,
    length_function=len,
)

docs_processed = []
for doc in RAW_KNOWLEDGE_BASE:
    doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
    docs_processed+= text_splitter.split_documents([doc])



tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
