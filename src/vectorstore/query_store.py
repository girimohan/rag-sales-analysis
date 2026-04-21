from typing import List

from sentence_transformers import SentenceTransformer
import chromadb


def query_vector_store(query: str, collection_name: str = "superstore", top_k: int = 5) -> dict:
    """Embed a query and retrieve the top_k most similar documents from ChromaDB.

    Loads the persistent client from 'chroma_db', embeds the query using
    'all-MiniLM-L6-v2', and returns the closest matching documents along
    with their IDs and distances.
    """
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name=collection_name)

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    query_embedding = model.encode(query).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    return {
        "documents": results["documents"][0],
        "ids": results["ids"][0],
        "distances": results["distances"][0],
    }

