from typing import List

from src.vectorstore.query_store import query_vector_store
from src.rag.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
import ollama


def build_context(docs: List[str]) -> str:
    """Join retrieved document chunks into a single context string."""
    return "\n\n".join(docs)


def run_rag(query: str, top_k: int = 20) -> str:
    """Retrieve relevant documents and return a local model answer for the query.

    Queries the vector store, builds a context string, fills the prompt
    templates, and returns the model's response as a plain string.
    """
    results = query_vector_store(query, top_k=top_k)
    context = build_context(results["documents"])

    user_message = RAG_USER_PROMPT.format(context=context, question=query)

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response["message"]["content"]


