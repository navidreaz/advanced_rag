from transformers import Pipeline
from langchain_community.vectorstores import FAISS
from typing import Optional, List, Tuple
from sentence_transformers import CrossEncoder
from ..generation.prompt import get_prompt_template

def answer_with_rag(
        question: str,
        llm: Pipeline,
        knowledge_index: FAISS,
        # reranker: Optional[RAGPretrainedModel] = None,
        reranker: Optional[CrossEncoder] = None,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
) -> Tuple[str, List[str]]:
    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(
        query=question, k=num_retrieved_docs
    )
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # Optionally rerank results
    if reranker:
        print("=> Reranking documents...")
        # relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = reranker.rank(question, relevant_docs, top_k=num_docs_final, return_documents=True)
        relevant_docs = [doc["text"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "\n".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
    )
    RAG_PROMPT_TEMPLATE = get_prompt_template()
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs