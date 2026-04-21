from functools import lru_cache
from typing import List

import torch
from sentence_transformers import SentenceTransformer
import chromadb

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Singletons — created once, reused on every call
_client: chromadb.PersistentClient | None = None
_model: SentenceTransformer | None = None


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

