from typing import List


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of at most chunk_size characters."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks
