from typing import List

from sentence_transformers import SentenceTransformer
import chromadb


def build_vector_store(chunks: List[str], collection_name: str = "superstore"):
    """Embed text chunks and store them in a persistent ChromaDB collection.

    Creates (or opens) a ChromaDB client at 'chroma_db/', embeds each chunk
    using 'all-MiniLM-L6-v2', and inserts the embeddings with the original
    text stored as metadata. Returns the collection object.
    """
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name=collection_name)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"text": chunk} for chunk in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    return collection

