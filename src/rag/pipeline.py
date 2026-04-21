from typing import List

from src.vectorstore.query_store import query_vector_store
from src.rag.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
import ollama

# Keywords that indicate an analytical question requiring summary documents
ANALYTICAL_KEYWORDS = {
    "profit", "sales", "revenue", "region", "category", "sub-category", "subcategory",
    "state", "segment", "trend", "year", "annual", "total", "highest", "lowest",
    "top", "best", "worst", "most", "least", "compare", "comparison", "growth",
    "2014", "2015", "2016", "2017", "customer", "loss", "city", "product",
    "performance", "rank", "ranking", "profitable", "shipping", "ship",
}


def is_analytical(query: str) -> bool:
    """Return True if the query is analytical (should prioritise summary docs)."""
    tokens = query.lower().replace("-", " ").split()
    return bool(ANALYTICAL_KEYWORDS.intersection(tokens))


def reorder_docs(docs: List[str], ids: List[str], distances: List[float], analytical: bool):
    """Put summary documents first for analytical queries, row-level docs first otherwise."""
    combined = list(zip(docs, ids, distances))
    summaries = [(d, i, s) for d, i, s in combined if i.startswith("summary_")]
    rows      = [(d, i, s) for d, i, s in combined if not i.startswith("summary_")]
    ordered = (summaries + rows) if analytical else (rows + summaries)
    d, i, s = zip(*ordered) if ordered else ([], [], [])
    return list(d), list(i), list(s)


def build_context(docs: List[str]) -> str:
    return "\n\n---\n\n".join(docs)


def run_rag(query: str, top_k: int = 20) -> str:
    """Retrieve relevant documents and generate an answer using the local LLM."""
    results = query_vector_store(query, top_k=top_k)

    analytical = is_analytical(query)
    docs, ids, distances = reorder_docs(
        results["documents"], results["ids"], results["distances"], analytical
    )

    context = build_context(docs)
    user_message = RAG_USER_PROMPT.format(context=context, question=query)

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response["message"]["content"]


