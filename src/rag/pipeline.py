from functools import lru_cache
from typing import Iterator, List

import torch
from src.vectorstore.query_store import query_vector_store
from src.rag.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
import ollama

MODEL = "phi3:mini"
GPU_LAYERS = -1 if torch.cuda.is_available() else 0  # -1 = all layers on GPU

# Keywords that indicate an analytical question requiring summary documents
ANALYTICAL_KEYWORDS = {
    "profit", "sales", "revenue", "region", "category", "sub-category", "subcategory",
    "state", "segment", "trend", "year", "annual", "total", "highest", "lowest",
    "top", "best", "worst", "most", "least", "compare", "comparison", "growth",
    "2014", "2015", "2016", "2017", "customer", "loss", "city", "product",
    "performance", "rank", "ranking", "profitable", "shipping", "ship",
}


def is_analytical(query: str) -> bool:
    tokens = query.lower().replace("-", " ").split()
    return bool(ANALYTICAL_KEYWORDS.intersection(tokens))


def reorder_docs(docs: List[str], ids: List[str], distances: List[float], analytical: bool):
    combined = list(zip(docs, ids, distances))
    summaries = [(d, i, s) for d, i, s in combined if i.startswith("summary_")]
    rows      = [(d, i, s) for d, i, s in combined if not i.startswith("summary_")]
    ordered = (summaries + rows) if analytical else (rows + summaries)
    d, i, s = zip(*ordered) if ordered else ([], [], [])
    return list(d), list(i), list(s)


def build_context(docs: List[str]) -> str:
    return "\n\n---\n\n".join(docs)


def _make_messages(query: str, top_k: int) -> list:
    """Build the message list for the LLM from retrieved context."""
    results = query_vector_store(query, top_k=top_k)
    analytical = is_analytical(query)
    docs, ids, distances = reorder_docs(
        results["documents"], results["ids"], results["distances"], analytical
    )
    context = build_context(docs)
    user_message = RAG_USER_PROMPT.format(context=context, question=query)
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


@lru_cache(maxsize=128)
def run_rag(query: str, top_k: int = 10) -> str:
    """Return a complete answer. Repeated identical queries are served from cache."""
    response = ollama.chat(
        model=MODEL,
        messages=_make_messages(query, top_k),
        options={"num_predict": 256, "num_gpu": GPU_LAYERS},
    )
    return response["message"]["content"]


def stream_rag(query: str, top_k: int = 10) -> Iterator[str]:
    """Yield answer tokens one at a time for streaming responses."""
    for chunk in ollama.chat(
        model=MODEL,
        messages=_make_messages(query, top_k),
        stream=True,
        options={"num_gpu": GPU_LAYERS},
    ):
        token = chunk["message"]["content"]
        if token:
            yield token


