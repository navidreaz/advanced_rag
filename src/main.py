from advanced_rag.rag.pipeline import answer_with_rag
from advanced_rag.retrieval.reranker import RERANKER
from advanced_rag.generation.reader_model import READER_LLM
from advanced_rag.retrieval.vector_store import KNOWLEDGE_VECTOR_DATABASE

if __name__ == "__main__":
    print("Hello Navid ... ")

    question = "how to create a pipeline object?"

    answer, relevant_docs = answer_with_rag(
        question, READER_LLM, KNOWLEDGE_VECTOR_DATABASE, reranker=RERANKER
    )

    print("==================================Answer==================================")
    print(f"{answer}")
    print("==================================Source docs==================================")
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i}------------------------------------------------------------")
        print(doc)