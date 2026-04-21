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

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"text": chunk} for chunk in chunks]

    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=chunks[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    return collection

