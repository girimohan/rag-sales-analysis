from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import chromadb

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Singletons — created once, reused on every call
_client = None
_model = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path="chroma_db")
    return _client


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=_DEVICE)
    return _model


@lru_cache(maxsize=256)
def _embed(query: str) -> list:
    """Embed a query string. Result is cached so repeated queries skip re-embedding."""
    return _get_model().encode(query).tolist()


def query_vector_store(query: str, collection_name: str = "superstore", top_k: int = 10) -> dict:
    """Retrieve the top_k most similar documents from ChromaDB for the given query."""
    collection = _get_client().get_collection(name=collection_name)
    results = collection.query(query_embeddings=[_embed(query)], n_results=top_k)
    return {
        "documents": results["documents"][0],
        "ids": results["ids"][0],
        "distances": results["distances"][0],
    }


_summaries_cache: dict | None = None


def get_all_summaries(collection_name: str = "superstore") -> dict:
    """Return all pre-computed summary documents from ChromaDB.

    Summaries are identified by their 'summary_' ID prefix (more reliable than
    metadata filtering, which requires a separate upsert step to set correctly).
    The result is cached in-process so repeated analytical queries pay no extra cost.
    """
    global _summaries_cache
    if _summaries_cache is not None:
        return _summaries_cache

    collection = _get_client().get_collection(name=collection_name)
    # Fetch only IDs first (lightweight), then retrieve matching summary docs
    all_ids: list[str] = collection.get(include=[])["ids"]
    summary_ids = [id_ for id_ in all_ids if id_.startswith("summary_")]
    if not summary_ids:
        _summaries_cache = {"documents": [], "ids": []}
        return _summaries_cache

    results = collection.get(ids=summary_ids, include=["documents"])
    _summaries_cache = {
        "documents": results["documents"],
        "ids": results["ids"],
    }
    return _summaries_cache


_summary_embeddings: np.ndarray | None = None


def get_relevant_summaries(query: str, top_k: int = 10) -> list:
    """Return the top_k most semantically relevant summary documents for the query.

    Embeddings for all summaries are computed once and cached in-process.
    This avoids context-window overload from feeding all 32 summaries to the LLM
    and prevents the model from being confused by unrelated summary documents.
    """
    global _summary_embeddings

    all_sums = get_all_summaries()
    docs = all_sums["documents"]
    if not docs:
        return []
    if len(docs) <= top_k:
        return docs

    # Build summary embedding matrix once and cache it
    if _summary_embeddings is None or _summary_embeddings.shape[0] != len(docs):
        _summary_embeddings = _get_model().encode(
            docs, convert_to_numpy=True, show_progress_bar=False
        )

    q_emb = _get_model().encode(query, convert_to_numpy=True)
    row_norms = np.linalg.norm(_summary_embeddings, axis=1)
    q_norm = np.linalg.norm(q_emb)
    denom = np.where(row_norms * q_norm == 0, 1.0, row_norms * q_norm)
    sims = _summary_embeddings @ q_emb / denom

    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [docs[i] for i in top_idx]

